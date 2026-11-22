"""Download the original study's data from its public Google Drive folder.

Source: the folder linked from https://github.com/E30895/economics-forecast-using-text
(Almeida, 2024). Run with --list first: it lists the files and downloads nothing.
Subfolders: 1_https_textos, 2_textos_completos (full PT text), 3_textos_tratados
(cleaned + translated), 4_sentiment_analysis (original indices), 5_dataset_economic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown

FOLDER_ID = "1Nody-yyQQBYg_QLQ0vPLtDvj9pD5EVa-"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
DEFAULT_OUT = Path("data/raw/original")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="list files, download nothing")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    files = gdown.download_folder(
        url=FOLDER_URL,
        output=str(args.out),
        quiet=args.list is False,
        skip_download=args.list,
        resume=True,
    )
    if args.list:
        for f in files or []:
            print(f.path)
    else:
        print(f"Downloaded {len(files or [])} files to {args.out}")


if __name__ == "__main__":
    main()
