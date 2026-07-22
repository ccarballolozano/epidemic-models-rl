"""Utilities for interacting with Azure ML — including fast artifact downloads
directly from blob storage, bypassing the slow AzureML Jobs API.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential
from azure.core.pipeline.transport import RequestsTransport
import requests
import requests.adapters
from tqdm import tqdm


TENANT_ID = os.environ["TENANT_ID"]
CONFIG_PATH = Path(__file__).parent.parent / "experiments" / "config.json"
CONTAINER_NAME = "azureml"
BLOB_PREFIX_TEMPLATE = "ExperimentRun/dcid.{run_name}/"
# Written inside a run folder after all its blobs are confirmed on disk.
# Presence means the run is complete and can be skipped without a network call.
_SENTINEL_FILENAME = ".complete"


def get_credential(interactive: bool = True):
    """Return an Azure credential for AzureML control-plane calls.

    Falls back to DefaultAzureCredential when *interactive* is False
    (useful in CI / headless environments).
    """
    if interactive:
        return InteractiveBrowserCredential(tenant_id=TENANT_ID)
    return DefaultAzureCredential()


def get_ml_client(credential=None) -> MLClient:
    """Create an MLClient from the project's config.json."""
    if credential is None:
        credential = get_credential()
    return MLClient.from_config(credential=credential, path=str(CONFIG_PATH))


def _make_container_client(
    storage_account_name: str,
    storage_account_key: str,
    container_name: str,
    max_workers: int,
):
    """Build a :class:`ContainerClient` with a connection pool sized for *max_workers*."""
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    # Build a requests Session with a pool large enough for the thread count.
    # The RequestsTransport constructor's connection_pool_maxsize kwarg does not
    # reliably propagate to urllib3, so we wire it up via HTTPAdapter explicitly.
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers,
    )
    session.mount("https://", adapter)
    blob_service = BlobServiceClient(
        account_url=account_url,
        credential=AzureNamedKeyCredential(storage_account_name, storage_account_key),
        transport=RequestsTransport(session=session),
    )
    return blob_service.get_container_client(container_name)


def _download_blobs(
    container_client,
    blob_prefix: str,
    output_dir: Path,
    overwrite: bool,
    executor: ThreadPoolExecutor,
) -> tuple[Path, int, int]:
    """Download all blobs under *blob_prefix* into *output_dir* using *executor*.

    Returns
    -------
    tuple[Path, int, int]
        ``(resolved_output_dir, n_downloaded, n_skipped)``.
    """
    blobs = list(container_client.list_blobs(name_starts_with=blob_prefix))

    if not blobs:
        raise FileNotFoundError(
            f"No blobs found at {blob_prefix}. Check that the run_name is correct."
        )

    def _download(blob) -> str:
        """Return 'downloaded' or 'skipped'."""
        relative_path = blob.name[len(blob_prefix) :]
        local_path = output_dir / relative_path
        if local_path.exists() and not overwrite:
            return "skipped"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            container_client.get_blob_client(blob.name).download_blob().readinto(f)
        return "downloaded"

    futures = {executor.submit(_download, blob): blob.name for blob in blobs}
    n_downloaded = n_skipped = 0
    for future in as_completed(futures):
        status = future.result()
        if status == "skipped":
            n_skipped += 1
        else:
            n_downloaded += 1

    return output_dir.resolve(), n_downloaded, n_skipped


def download_run_artifacts(
    run_name: str,
    output_dir: str | Path,
    storage_account_name: str,
    storage_account_key: str,
    *,
    container_name: str = CONTAINER_NAME,
    overwrite: bool = False,
    skip_downloaded: bool = False,
    max_workers: int = 16,
) -> Path:
    """Download all artifacts of a single AzureML run directly from blob storage.

    Bypasses the slow AzureML Jobs API by using ``azure-storage-blob``
    authenticated with a storage account key.

    Parameters
    ----------
    run_name:
        Run ID as it appears in the blob path, e.g. ``"icy_calypso_44jyl8t5c4"``.
        The full blob prefix used is ``ExperimentRun/dcid.<run_name>/``.
    output_dir:
        Local directory where artifacts will be written.  The directory
        structure under the blob prefix is preserved.
    storage_account_name:
        Azure Storage account name.
        Find it in the Azure Portal → Storage account → Overview.
    storage_account_key:
        Storage account key (key1 or key2).
        Find it in the Azure Portal → Storage account → Security + networking → Access keys.
    container_name:
        Blob container name (default: ``"azureml"``).
    overwrite:
        ``True``  — always re-download every file, ignoring local copies.
        ``False`` — skip individual files that already exist locally (default).
    skip_downloaded:
        When ``True`` and ``overwrite`` is ``False``, skip the Azure blob listing
        entirely for runs that have a ``.complete`` sentinel file on disk.
        The sentinel is written automatically after every successful full download.
        Has no effect when ``overwrite=True``.
    max_workers:
        Number of parallel download threads (default: 16).  Blob downloads
        are network I/O-bound so threading gives a large speedup.

    Returns
    -------
    Path
        The resolved *output_dir*.
    """
    run_output_dir = Path(output_dir)
    sentinel = run_output_dir / _SENTINEL_FILENAME

    if skip_downloaded and not overwrite and sentinel.exists():
        print(f"{run_name}: skipped (sentinel present)")
        return run_output_dir.resolve()

    container_client = _make_container_client(
        storage_account_name, storage_account_key, container_name, max_workers
    )
    blob_prefix = BLOB_PREFIX_TEMPLATE.format(run_name=run_name)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        path, n_downloaded, n_skipped = _download_blobs(
            container_client, blob_prefix, run_output_dir, overwrite, executor
        )
    if not overwrite:
        sentinel.touch()
    print(f"{run_name}: {n_downloaded} downloaded, {n_skipped} skipped")
    return path


def download_runs_artifacts(
    run_names: list[str],
    output_dir: str | Path,
    storage_account_name: str,
    storage_account_key: str,
    *,
    container_name: str = CONTAINER_NAME,
    overwrite: bool = False,
    skip_downloaded: bool = False,
    max_workers: int = 16,
) -> dict[str, Path]:
    """Download artifacts for a list of AzureML runs, with a progress bar over runs.

    A single :class:`BlobServiceClient` (and its connection pool) is created
    once and reused across all runs, avoiding the per-run TCP handshake overhead.

    Parameters
    ----------
    run_names:
        List of run IDs to download.
    output_dir:
        Root directory.  Each run is saved under ``<output_dir>/<run_name>/``.
    storage_account_name:
        Azure Storage account name.
    storage_account_key:
        Storage account key (key1 or key2).
    container_name:
        Blob container name (default: ``"azureml"``).
    overwrite:
        ``True``  — always re-download every file, ignoring local copies.
        ``False`` — skip individual files that already exist locally (default).
    skip_downloaded:
        When ``True`` and ``overwrite`` is ``False``, skip the Azure blob listing
        entirely for runs that have a ``.complete`` sentinel file on disk.
        The sentinel is written automatically after every successful full download.
        Has no effect when ``overwrite=True``.
    max_workers:
        Parallel download threads per run (default: 16).

    Returns
    -------
    dict[str, Path]
        Mapping of ``run_name → local output path`` for every successfully
        downloaded run.  Runs that fail are logged and excluded from the result.
    """
    results: dict[str, Path] = {}
    errors: dict[str, Exception] = {}
    total_downloaded = total_skipped = total_run_skipped = 0

    # Runs that need an actual network call
    runs_to_fetch = []
    for run_name in run_names:
        run_output_dir = Path(output_dir) / run_name
        sentinel = run_output_dir / _SENTINEL_FILENAME
        if skip_downloaded and not overwrite and sentinel.exists():
            results[run_name] = run_output_dir.resolve()
            total_run_skipped += 1
        else:
            runs_to_fetch.append(run_name)

    if total_run_skipped:
        tqdm.write(f"Skipped {total_run_skipped} already-complete runs (sentinel present)")

    if runs_to_fetch:
        # Build the client and thread pool once — reused across all remaining runs.
        container_client = _make_container_client(
            storage_account_name, storage_account_key, container_name, max_workers
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for run_name in tqdm(runs_to_fetch, desc="Runs", unit="run"):
                run_output_dir = Path(output_dir) / run_name
                try:
                    blob_prefix = BLOB_PREFIX_TEMPLATE.format(run_name=run_name)
                    path, n_downloaded, n_skipped = _download_blobs(
                        container_client, blob_prefix, run_output_dir, overwrite, executor
                    )
                    results[run_name] = path
                    total_downloaded += n_downloaded
                    total_skipped += n_skipped
                    if not overwrite:
                        (run_output_dir / _SENTINEL_FILENAME).touch()
                    tqdm.write(
                        f"  {run_name}: {n_downloaded} downloaded, {n_skipped} skipped"
                    )
                except Exception as exc:
                    errors[run_name] = exc
                    tqdm.write(f"  {run_name}: FAILED — {exc}")

    print(
        f"\nDone — {len(results)}/{len(run_names)} runs succeeded "
        f"({total_downloaded} files downloaded, {total_skipped} files skipped, "
        f"{total_run_skipped} runs skipped entirely, "
        f"{len(errors)} runs failed) → {Path(output_dir).resolve()}"
    )
    return results
