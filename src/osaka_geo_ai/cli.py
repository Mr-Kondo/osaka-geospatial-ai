"""Shared command-line handling keeps every script small and consistent."""

import argparse
import logging

from osaka_geo_ai.config import ROOT, load_config
from osaka_geo_ai.pipeline.pipeline import run_pipeline, run_stage


def main(stage="pipeline"):
    parser = argparse.ArgumentParser(description=f"Osaka Geospatial AI: {stage}")
    parser.add_argument(
        "--config", default=str(ROOT / "configs/default.yaml"), help="YAML configuration path"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    if stage in {"pipeline", "download"}:
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(
            "--offline", action="store_true", help="Use checksum-verified cached archives only"
        )
        mode.add_argument(
            "--refresh", action="store_true", help="Redownload and verify configured archives"
        )
    if stage == "pipeline":
        parser.add_argument(
            "--phase",
            type=int,
            choices=[1, 2, 3, 4],
            default=4,
            help="Run through this implementation phase",
        )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    try:
        config = load_config(args.config)
        if stage == "pipeline":
            run_pipeline(config, phase=args.phase, offline=args.offline, refresh=args.refresh)
        else:
            run_stage(
                stage,
                config,
                **(
                    {"offline": args.offline, "refresh": args.refresh}
                    if stage == "download"
                    else {}
                ),
            )
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc, exc_info=args.log_level == "DEBUG")
        raise SystemExit(1) from exc
