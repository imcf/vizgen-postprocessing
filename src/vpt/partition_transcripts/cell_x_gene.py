from typing import List, Optional

try:
    from vpt_core import log
except ImportError:
    import logging

    log = logging.getLogger("cell_x_gene")

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import shapely
from shapely import Polygon
from vpt_core.io.vzgfs import get_storage_options, retrying_attempts, vzg_open

from vpt.utils.boundaries import Boundaries

ROW_GROUP_SIZE = 10_000_000  # about 1 GB in memory


def read_parquet_chunked(f, chunksize: int):
    file = pq.ParquetFile(f)

    if file.metadata.num_rows == 0:
        yield file.read().to_pandas()
        return

    for chunk in file.iter_batches(chunksize):
        temp = chunk.to_pandas()
        if "" in temp.columns:
            yield temp.rename(columns={"": "Unnamed: 0"})
        else:
            yield temp.reset_index().rename(columns={"index": "Unnamed: 0"})


def get_chunks(input_transcripts: str, chunk_size: int):
    if input_transcripts.endswith(".csv"):
        with vzg_open(input_transcripts, "r") as f:
            yield from pd.read_csv(f, chunksize=chunk_size)
    elif input_transcripts.endswith(".parquet"):
        with vzg_open(input_transcripts, "rb") as f:
            yield from read_parquet_chunked(f, chunksize=chunk_size)
    else:
        raise NotImplementedError()


def write_detected_transcripts(
    transcripts_df, output_path: str, append: bool = False
) -> None:
    storage_options = get_storage_options(output_path)

    for attempt in retrying_attempts():
        with attempt:
            if output_path.endswith(".csv"):
                # Use Polars for CSV writing if possible
                if isinstance(transcripts_df, pl.DataFrame):
                    transcripts_df.write_csv(output_path)
                else:
                    pl.DataFrame(transcripts_df).write_csv(output_path)
            elif output_path.endswith(".parquet"):
                # Polars can write parquet, but keep pandas for compatibility
                if isinstance(transcripts_df, pl.DataFrame):
                    transcripts_df.write_parquet(output_path, compression="zstd")
                else:
                    transcripts_df.to_parquet(
                        output_path,
                        engine="fastparquet",
                        index=False,
                        append=append,
                        compression="zstd",
                        row_group_offsets=ROW_GROUP_SIZE,
                        storage_options=storage_options,
                    )
            else:
                raise NotImplementedError()


def process_chunk(
    chunk_df, shapely_list, z_planes_count, cell_id_list, needs_new_dt: bool = False
):
    """
    Process a chunk of transcript data and assign cell IDs.

    Parameters
    ----------
    chunk_df : pd.DataFrame
        Transcript data chunk.
    shapely_list : list
        List of shapely polygons per z-plane.
    z_planes_count : int
        Number of z-planes.
    cell_id_list : list
        List of cell IDs.
    needs_new_dt : bool, optional
        Whether to assign new cell IDs to transcripts.

    Returns
    -------
    cell_x_gene : pd.DataFrame
        Cell-by-gene matrix for the chunk.
    transcripts_df : pd.DataFrame
        Transcripts with assigned cell IDs.

    Examples
    --------
    >>> process_chunk(chunk_df, shapely_list, 3, cell_id_list)
    """
    log.info(
        f"Processing chunk with {len(chunk_df)} transcripts and {z_planes_count} z-planes."
    )
    genes_detected = chunk_df["gene"].unique()
    log.info(f"Detected {len(genes_detected)} genes in chunk.")
    grouped = chunk_df.groupby(chunk_df["gene"])

    gene_df_list = []
    transcripts_list = []
    for gene in genes_detected:
        log.info(f"Processing gene '{gene}' in chunk.")
        one_gene = grouped.get_group(gene)
        one_gene_partition_list = []
        for z in range(z_planes_count):
            one_gene_z = one_gene.loc[one_gene["global_z"] == z]
            points = shapely.points(one_gene_z["global_x"], one_gene_z["global_y"])
            one_gene_tree = shapely.STRtree(points)
            one_gene_partition_z = one_gene_tree.query(
                shapely_list[z], predicate="contains"
            )
            one_gene_partition_list.append(one_gene_partition_z)

            if needs_new_dt:
                out = np.full(len(one_gene_z), -1, dtype=np.int64)
                if len(one_gene_partition_z[0]) > 0:
                    cell_id_vectorize = np.vectorize(lambda t: cell_id_list[t])
                    out[one_gene_partition_z[1]] = cell_id_vectorize(
                        one_gene_partition_z[0]
                    )

                one_gene_z = one_gene_z.assign(cell_id=out)
                transcripts_list.append(one_gene_z)

        if needs_new_dt:
            unhandled_transcripts = one_gene.loc[
                (one_gene["global_z"] >= z_planes_count) | (one_gene["global_z"] < 0)
            ]
            unhandled_transcripts.assign(cell_id=-1)
            if len(unhandled_transcripts) > 0:
                transcripts_list.append(unhandled_transcripts)

        if one_gene_partition_list:
            one_gene_partition = np.concatenate(one_gene_partition_list, axis=1)
        else:
            one_gene_partition = np.array([[], []])

        # Use Polars for groupby/count
        df_one_gene = pl.DataFrame(
            {"cell_id": one_gene_partition[0], gene: one_gene_partition[1]}
        )
        cell_x_one_gene = df_one_gene.groupby("cell_id").agg([pl.count()])
        gene_df_list.append(cell_x_one_gene)

    # Use Polars for join/fillna
    cell_x_gene = (
        pl.DataFrame({"cell": cell_id_list})
        .join(gene_df_list, on=None, how="left")
        .fill_null(0)
    )
    cell_x_gene = cell_x_gene.with_column(pl.col("cell").cast(pl.Int64))

    if len(transcripts_list) > 0:
        transcripts_df = pl.concat([pl.DataFrame(t) for t in transcripts_list])
        transcripts_df = transcripts_df.filter(pl.col("index").is_in(chunk_df.index))
    else:
        transcripts_df = pl.DataFrame(
            {col: [] for col in list(chunk_df.columns) + ["cell_id"]}
        )

    log.info(
        "Finished processing chunk. Returning cell_x_gene matrix and transcripts_df."
    )
    return cell_x_gene, transcripts_df


def construct_cell_x_gene(
    transcripts,
    geometry_list,
    z_planes_count: int,
    cell_id_list,
    output_transcripts: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construct cell-by-gene matrix from transcript and geometry data.

    Parameters
    ----------
    transcripts : iterator or generator
        Transcript data chunks.
    geometry_list : np.ndarray
        Array of cell polygons per z-plane.
    z_planes_count : int
        Number of z-planes.
    cell_id_list : list
        List of cell IDs.
    output_transcripts : str, optional
        Output path for transcripts with cell IDs.

    Returns
    -------
    cell_x_gene : pd.DataFrame
        Cell-by-gene matrix.

    Examples
    --------
    >>> construct_cell_x_gene(transcripts, geometry_list, 3, cell_id_list)
    """
    log.info("Starting construct_cell_x_gene.")
    # import dask
    # import platform

    from dask.distributed import Client, LocalCluster, as_completed
    from tqdm.auto import tqdm

    # Convert transcripts (iterator/generator) to list for parallel processing
    chunk_list = list(transcripts)
    log.info(f"Loaded {len(chunk_list)} transcript chunks for parallel processing.")
    needs_new_dt = output_transcripts is not None

    # Use threads for Dask LocalCluster to ensure robust parallelism on all platforms
    use_processes = False
    n_workers = min(4, len(chunk_list)) if len(chunk_list) > 0 else 1
    log.info(f"Launching Dask LocalCluster with {n_workers} workers.")
    with LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=use_processes,
        dashboard_address=None,
    ) as cluster:
        with Client(cluster) as client:
            futures = [
                client.submit(
                    process_chunk,
                    chunk_df,
                    geometry_list,
                    z_planes_count,
                    cell_id_list,
                    needs_new_dt,
                )
                for chunk_df in chunk_list
            ]
            results = []
            with tqdm(
                total=len(futures),
                desc="Transcript chunks processed",
                unit="chunk",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                for fut in as_completed(futures):
                    results.append(fut.result())
                    pbar.update(1)
    log.info("Parallel chunk processing complete.")

    # Unpack results
    cell_by_gene = pl.DataFrame({"cell": cell_id_list})
    barcode_id_name_df = pl.DataFrame({"barcode_id": [], "gene": []})
    transcripts_df_list = []
    for i, (chunk_cell_by_gene, transcripts_df) in enumerate(results):
        log.info(f"Merging results from chunk {i + 1}/{len(results)}.")
        cell_by_gene = (
            pl.concat([cell_by_gene, chunk_cell_by_gene])
            .groupby("cell")
            .agg([pl.sum(pl.col(col)) for col in cell_by_gene.columns if col != "cell"])
            .fill_null(0)
        )
        chunk = chunk_list[i]
        barcode_id_name_df = pl.concat(
            [
                barcode_id_name_df,
                pl.DataFrame(
                    {"barcode_id": chunk["barcode_id"], "gene": chunk["gene"]}
                ),
            ]
        ).unique(subset=["barcode_id"])
        if needs_new_dt:
            transcripts_df_list.append(transcripts_df)

    # Write transcripts sequentially after parallel processing
    if needs_new_dt and output_transcripts is not None:
        log.info(f"Writing transcripts to {output_transcripts}.")
        first_chunk = True
        for transcripts_df in transcripts_df_list:
            # Polars does not support rename by index, so fallback to pandas for this step
            if isinstance(transcripts_df, pl.DataFrame):
                transcripts_df = transcripts_df.rename({transcripts_df.columns[0]: ""})
            else:
                transcripts_df = transcripts_df.rename(
                    columns={transcripts_df.columns[0]: ""}
                )
            write_detected_transcripts(
                transcripts_df, output_transcripts, append=not first_chunk
            )
            first_chunk = False

    cell_by_gene = cell_by_gene.sort("cell")
    barcode_id_name_df = barcode_id_name_df.sort("barcode_id")
    # Reindex columns to match barcode_id_name_df["gene"]
    gene_order = barcode_id_name_df["gene"].to_list()
    cell_by_gene = cell_by_gene.select(["cell"] + gene_order)
    log.info("Finished construct_cell_x_gene. Returning cell_x_gene matrix.")
    return cell_by_gene.with_columns(
        [pl.col(col).cast(pl.Int64) for col in cell_by_gene.columns if col != "cell"]
    )


def cell_by_gene_matrix(
    bnds: Boundaries,
    transcripts: pd.DataFrame,
    output_transcripts: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate cell-by-gene matrix from boundaries and transcript data.

    Parameters
    ----------
    bnds : Boundaries
        Cell boundary object.
    transcripts : pd.DataFrame
        Transcript data.
    output_transcripts : str, optional
        Output path for transcripts with cell IDs.

    Returns
    -------
    cell_x_gene : pd.DataFrame
        Cell-by-gene matrix.

    Examples
    --------
    >>> cell_by_gene_matrix(bnds, transcripts)
    """
    log.info("Starting cell_by_gene_matrix.")
    idList = []
    geomList: List[List[Polygon]] = []
    for z in range(bnds.get_z_planes_count()):
        geomList.append([])
    for feature in bnds.features:
        idList.append(np.int64(feature.get_feature_id()))
        for zIdx, poly in enumerate(feature.get_full_cell()):
            geomList[zIdx].append(poly)

    log.info(
        f"Prepared {len(idList)} cell IDs and {len(geomList)} z-planes for matrix construction."
    )
    cell_x_gene = construct_cell_x_gene(
        transcripts,
        np.array(geomList),
        bnds.get_z_planes_count(),
        idList,
        output_transcripts,
    )
    log.info("cell_by_gene_matrix complete. Returning cell_x_gene matrix.")
    return cell_x_gene
