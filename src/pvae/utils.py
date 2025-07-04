import hashlib
import re
from pathlib import Path

import requests
from tqdm import tqdm

PATTERN_SPACE = re.compile(r" +")
PATTERN_NOT_ALPHANUMERIC = re.compile(r"[^0-9a-zA-Z_]")
PATTERN_UNDERSCORE_DUPLICATED = re.compile(r"_{2,}")


def simplify_string(value: str) -> str:
    # replace spaces by _
    value = re.sub(PATTERN_SPACE, "_", value)

    # remove non-alphanumeric characters
    value = re.sub(PATTERN_NOT_ALPHANUMERIC, "", value)

    # replace spaces by _
    value = re.sub(PATTERN_UNDERSCORE_DUPLICATED, "_", value)

    return value


def download_file(
    url: str, output_file: str | Path, block_size: int = 1024, progress_bar: bool = True
):
    """Default function to download a file given an URL and output path"""
    # command = ["curl", "-s", "-L", url, "-o", output_file]
    # run(command)

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))

    with (
        open(output_file, "wb") as handle,
        tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            disable=not progress_bar,
        ) as pbar,
    ):
        for data in response.iter_content(block_size):
            pbar.update(len(data))
            handle.write(data)


def download(
    url: str,
    output_file: str | Path,
    md5hash: str = None,
    download_file_func=download_file,
    raise_on_md5hash_mismatch=True,
):
    """Downloads a file from an URL. If the md5hash option is specified, it checks
    if the file was successfully downloaded (whether MD5 matches).
    Before starting the download, it checks if output_file exists. If so, and md5hash
    is None, it quits without downloading again. If md5hash is not None, it checks if
    it matches the file.
    Args:
        url: URL of file to download.
        output_file: path of file to store content.
        md5hash: expected MD5 hash of file to download.
        download_file_func: a function that receives two arguments (a url to
            a file and an output file path). It is supposed to download the file
            pointed by the URL and save it to the specified path. This argument is
            mainly used for unit testing purposes.
        raise_on_md5hash_mismatch: if the method should raise an AssertionError
            if the downloaded file does not match the given md5 hash.
    """
    Path(output_file).resolve().parent.mkdir(parents=True, exist_ok=True)

    if Path(output_file).exists() and (
        md5hash is None or md5_matches(md5hash, output_file)
    ):
        print(f"File already downloaded: {output_file}")
        return

    download_file_func(url, output_file)

    if md5hash is not None and not md5_matches(md5hash, output_file):
        msg = "MD5 does not match"
        print(msg)

        if raise_on_md5hash_mismatch:
            raise AssertionError(msg)


def md5_matches(expected_md5: str, filepath: str) -> bool:
    """Checks the MD5 hash for a given filename and compares with the expected value.
    Args:
        expected_md5: expected MD5 hash.
        filepath: file for which MD5 will be computed.
    Returns:
        True if MD5 matches, False otherwise.
    """
    with open(filepath, "rb") as f:
        current_md5 = hashlib.md5(f.read()).hexdigest()
        return expected_md5 == current_md5
