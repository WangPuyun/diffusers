#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

from run_trial import load_context, read_json, validate_result


def parse_args():
    parser = argparse.ArgumentParser(description="Check Flux2 Klein loss-search trial integrity")
    parser.add_argument("--study", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        context = load_context(args.study, args.candidate)
        immutable_result = context["output_root"] / "trials" / context["fingerprint"] / "result.json"
        if immutable_result.is_file():
            validate_result(immutable_result, context)
            return 0

        lock_path = context["output_root"] / "locks" / f"{context['fingerprint']}.lock"
        if lock_path.exists():
            raise ValueError(f"Current candidate has an unresolved trial lock: {lock_path}")

        latest_path = Path(args.result)
        if latest_path.is_file():
            latest = read_json(latest_path)
            if latest.get("fingerprint") == context["fingerprint"]:
                if latest.get("status") == "failure":
                    raise ValueError(latest.get("error", "Current trial failed"))
                if latest.get("status") == "success":
                    raise ValueError("Success summary exists without its immutable trial result")
                elif latest.get("status") == "running":
                    raise ValueError("Current candidate has an unfinished running result")
                else:
                    raise ValueError("Current candidate has an unknown result status")
        return 0
    except Exception as error:
        print(f"guard_trial failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
