#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e -o pipefail

KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

# source hack/lib
source "${KUBE_ROOT}/hack/lib/init.sh"

FILE=${0}
BASELINE_BENCHMARK_FILE="${KUBE_ROOT}/_output/baseline.report"
TARGET_BENCHMARK_FILE="${KUBE_ROOT}/_output/target.report"

function help {
    {
        printf "\n"
        printf "This script is expected to run twice for comparing initial and final benchmarks."
        printf "\n\tFirst run for establishing a baseline benchmark this should be done by pointing"
        printf "\n\tthe repository to master/main branch of the upstream."
        printf "\n\tSecond run for generating benchmarks after code optimization and finally comparing "
        printf "\n\tthe target with baseline benchmark."
        printf "\n\nUsage: %s [-b] [-c count] [-f function] [-h]\n" "${FILE}"
        printf "\t-b generate baseline benchmark, generates target benchmark if '-b' not specified.\n"
        printf "\t-c count, run each benchmark n times.\n"
        printf "\t-f function, regular expression for the function to run benchmark on.\n"
        printf "\t-h help.\n"
        printf "\nExample:\n"
        printf "\t %s -b\n" "${FILE}"
        printf "\t %s\n" "${FILE}"
        printf "\t %s -b -c 25\n" "${FILE}"
        printf "\t %s -c 25\n" "${FILE}"
        printf "\t %s -b -c 25 -f BenchmarkWriter\n" "${FILE}"
    } >&2
    exit 1
}

function info_message(){
    echo -e "[ INFO ] ${1}" >&2
}

function error_message(){
    echo -e "[ ERROR ] ${1}" >&2
}

function benchmark() {
    mode=$1
    count=$2
    function=$3
    benchmark_file=""
    info_message "mode: ${mode}"

    # generate comparison if the mode is target
    if [[ "${mode}" == "target" ]]; then
        benchmark_file=$TARGET_BENCHMARK_FILE

        # exit if baseline benchmark doesn't exist
        if [[ ! -f $BASELINE_BENCHMARK_FILE ]]; then
            error_message "no baseline benchmark to compare with, run '${FILE} -b' to generate baseline benchmark for comparison\n"
            exit 1
        fi
    else
        benchmark_file=$BASELINE_BENCHMARK_FILE
    fi

    # run the benchmark
    (set -x; go test --bench="${function}" --benchmem --count="${count}" --run=^# > "${benchmark_file}")

    # generate comparison if the mode is target
    if [[ "${mode}" == "target" ]]; then
        # generate comparison
        info_message "running benchstat"
        (set -x; benchstat "${BASELINE_BENCHMARK_FILE}" "${TARGET_BENCHMARK_FILE}")
    fi
}

mode="target"
count=10
function="."

while getopts hbc:f: flag
do
    case "${flag}" in
        h) help ;;
        b) mode="baseline";;
        c) count="${OPTARG}";;
        f) function="${OPTARG}";;
        ? ) help ;;
    esac
done

# setup go env before running any benchmark
kube::golang::setup_env

benchmark "${mode}" "${count}" "${function}"

