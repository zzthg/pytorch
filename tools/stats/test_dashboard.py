import argparse
from collections import defaultdict
from functools import lru_cache
import json
from multiprocessing import Pool
from pathlib import Path
import re
import sys
from urllib.request import urlopen

import requests
from tools.stats.upload_stats_lib import (
    _get_request_headers,
    download_gha_artifacts,
    download_s3_artifacts,
    unzip,
)
from tools.stats.upload_test_stats import get_tests, parse_xml_report
from typing import List, Dict, Any
import os
from tempfile import TemporaryDirectory


def get_tests(workflow_run_id: int, workflow_run_attempt: int) -> List[Dict[str, Any]]:
    temp_dir = f"/Users/csl/zzzzzzzz/tmp/{workflow_run_id}"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_paths = download_s3_artifacts(
            "test-report", workflow_run_id, workflow_run_attempt
        )
        mp = Pool(20)
        for path in s3_paths:
            mp.apply_async(unzip, args=(path,))
        mp.close()
        mp.join()

        artifact_paths = download_gha_artifacts(
            "test-report", workflow_run_id, workflow_run_attempt
        )
        for path in artifact_paths:
            unzip(path)
    os.chdir(temp_dir)

    # Parse the reports and transform them to JSON
    test_cases = []
    mp = Pool(20)
    for xml_report in Path(".").glob("**/*.xml"):
        test_cases.append(
            mp.apply_async(
                parse_xml_report,
                args=("testcase", xml_report, workflow_run_id, workflow_run_attempt),
            )
        )
    mp.close()
    mp.join()
    test_cases = [tc.get() for tc in test_cases]
    flattened = [item for sublist in test_cases for item in sublist]

    return flattened


@lru_cache(maxsize=1000)
def get_job_name(job_id):
    try:
        return requests.get(
            f"https://api.github.com/repos/pytorch/pytorch/actions/jobs/{job_id}",
            headers=_get_request_headers(),
        ).json()["name"]
    except Exception as e:
        print(f"Failed to get job name for job id {job_id}: {e}")
        return "NoJobName"


REGEX_JOB_INFO = r"(.*) \/ .*test \(([^,]*), .*\)"

@lru_cache(maxsize=1000)
def get_build_name(job_name: str) -> str:
    try:
        return re.match(REGEX_JOB_INFO, job_name).group(1)
    except AttributeError:
        print(f"Failed to match job name: {job_name}")
        return "NoBuildEnv"

@lru_cache(maxsize=1000)
def get_test_config(job_name: str) -> str:
    try:
        return re.match(REGEX_JOB_INFO, job_name).group(2)
    except AttributeError:
        print(f"Failed to match job name: {job_name}")
        return "NoTestConfig"


def get_per_job_summary(test_cases):
    grouped = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )

    for test_case in test_cases:
        job_name = get_job_name(test_case["job_id"])
        build_name = get_build_name(job_name)
        test_config = get_test_config(job_name)
        class_name = test_case.pop("classname", "NoClass")
        name = test_case.pop("name", "NoName")
        invoking_file = test_case.pop("invoking_file", "NoFile")
        grouped[build_name][test_config][invoking_file][class_name][name].append(
            test_case
        )
    with open("t.json", "w") as f:
        print(json.dumps(grouped, indent=2), file=f)
    couunt = 0
    for build_name, build in grouped.items():
        for test_config, test_config_data in build.items():
            for invoking_file, invoking_file_data in test_config_data.items():
                for class_name, class_data in invoking_file_data.items():
                    for test_name, test_data in class_data.items():
                        couunt += 1
    print(couunt)
    return grouped

def compare_build(job_summary, base_job_summary, build_name):
    # Compare the two summaries for a single build
    if build_name not in job_summary:
        print(f"Build {build_name} not found in job summary")
        return {
            "new_tests": {},
            "removed_tests": base_job_summary[build_name],
            "changed_tests": {},
        }
    if build_name not in base_job_summary:
        print(f"Build {build_name} not found in base job summary")
        return {
            "new_tests": job_summary[build_name],
            "removed_tests": {},
            "changed_tests": {},
        }
    build_summary = job_summary[build_name]
    base_build_summary = base_job_summary[build_name]

    return compare(build_summary, base_build_summary)


def compare(job_summary, base_job_summary):
    # Compare the two summaries
    diff = {}
    if isinstance(job_summary, list):
        return {}
    all_builds = set(job_summary.keys()) | set(base_job_summary.keys())
    for build_name in all_builds:
        diff[build_name] = compare_build(job_summary, base_job_summary, build_name)
    return diff

def get_parser():
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--base-workflow-run-id",
        type=int,
        help="id of the base workflow to get artifacts from",
        required=True,
    )


    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    print(f"Workflow id is: {args.workflow_run_id}")

    test_cases = get_tests(args.workflow_run_id, args.workflow_run_attempt)
    base_test_cases = get_tests(args.base_workflow_run_id, 1)
    job_summary = get_per_job_summary(test_cases)
    base_job_summary = get_per_job_summary(base_test_cases)
    with open("t.txt", "w") as f:
        print(json.dumps(compare(job_summary, base_job_summary), indent=2), file=f)

    # Flush stdout so that any errors in Rockset upload show up last in the logs.
    sys.stdout.flush()
