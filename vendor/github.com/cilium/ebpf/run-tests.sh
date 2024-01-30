#!/usr/bin/env bash
# Test the current package under a different kernel.
# Requires virtme and qemu to be installed.
# Examples:
#     Run all tests on a 5.4 kernel
#     $ ./run-tests.sh 5.4
#     Run a subset of tests:
#     $ ./run-tests.sh 5.4 ./link

set -euo pipefail

script="$(realpath "$0")"
readonly script

# This script is a bit like a Matryoshka doll since it keeps re-executing itself
# in various different contexts:
#
#   1. invoked by the user like run-tests.sh 5.4
#   2. invoked by go test like run-tests.sh --exec-vm
#   3. invoked by init in the vm like run-tests.sh --exec-test
#
# This allows us to use all available CPU on the host machine to compile our
# code, and then only use the VM to execute the test. This is because the VM
# is usually slower at compiling than the host.
if [[ "${1:-}" = "--exec-vm" ]]; then
  shift

  input="$1"
  shift

  # Use sudo if /dev/kvm isn't accessible by the current user.
  sudo=""
  if [[ ! -r /dev/kvm || ! -w /dev/kvm ]]; then
    sudo="sudo"
  fi
  readonly sudo

  testdir="$(dirname "$1")"
  output="$(mktemp -d)"
  printf -v cmd "%q " "$@"

  if [[ "$(stat -c '%t:%T' -L /proc/$$/fd/0)" == "1:3" ]]; then
    # stdin is /dev/null, which doesn't play well with qemu. Use a fifo as a
    # blocking substitute.
    mkfifo "${output}/fake-stdin"
    # Open for reading and writing to avoid blocking.
    exec 0<> "${output}/fake-stdin"
    rm "${output}/fake-stdin"
  fi

  for ((i = 0; i < 3; i++)); do
    if ! $sudo virtme-run --kimg "${input}/bzImage" --memory 768M --pwd \
      --rwdir="${testdir}=${testdir}" \
      --rodir=/run/input="${input}" \
      --rwdir=/run/output="${output}" \
      --script-sh "PATH=\"$PATH\" CI_MAX_KERNEL_VERSION="${CI_MAX_KERNEL_VERSION:-}" \"$script\" --exec-test $cmd" \
      --kopt possible_cpus=2; then # need at least two CPUs for some tests
      exit 23
    fi

    if [[ -e "${output}/status" ]]; then
      break
    fi

    if [[ -v CI ]]; then
      echo "Retrying test run due to qemu crash"
      continue
    fi

    exit 42
  done

  rc=$(<"${output}/status")
  $sudo rm -r "$output"
  exit $rc
elif [[ "${1:-}" = "--exec-test" ]]; then
  shift

  mount -t bpf bpf /sys/fs/bpf
  mount -t tracefs tracefs /sys/kernel/debug/tracing

  if [[ -d "/run/input/bpf" ]]; then
    export KERNEL_SELFTESTS="/run/input/bpf"
  fi

  if [[ -f "/run/input/bpf/bpf_testmod/bpf_testmod.ko" ]]; then
    insmod "/run/input/bpf/bpf_testmod/bpf_testmod.ko"
  fi

  dmesg --clear
  rc=0
  "$@" || rc=$?
  dmesg
  echo $rc > "/run/output/status"
  exit $rc # this return code is "swallowed" by qemu
fi

readonly kernel_version="${1:-}"
if [[ -z "${kernel_version}" ]]; then
  echo "Expecting kernel version as first argument"
  exit 1
fi
shift

readonly kernel="linux-${kernel_version}.bz"
readonly selftests="linux-${kernel_version}-selftests-bpf.tgz"
readonly input="$(mktemp -d)"
readonly tmp_dir="${TMPDIR:-/tmp}"
readonly branch="${BRANCH:-master}"

fetch() {
    echo Fetching "${1}"
    pushd "${tmp_dir}" > /dev/null
    curl -s -L -O --fail --etag-compare "${1}.etag" --etag-save "${1}.etag" "https://github.com/cilium/ci-kernels/raw/${branch}/${1}"
    local ret=$?
    popd > /dev/null
    return $ret
}

fetch "${kernel}"
cp "${tmp_dir}/${kernel}" "${input}/bzImage"

if fetch "${selftests}"; then
  echo "Decompressing selftests"
  mkdir "${input}/bpf"
  tar --strip-components=4 -xf "${tmp_dir}/${selftests}" -C "${input}/bpf"
else
  echo "No selftests found, disabling"
fi

args=(-short -coverpkg=./... -coverprofile=coverage.out -count 1 ./...)
if (( $# > 0 )); then
  args=("$@")
fi

export GOFLAGS=-mod=readonly
export CGO_ENABLED=0
# LINUX_VERSION_CODE test compares this to discovered value.
export KERNEL_VERSION="${kernel_version}"

echo Testing on "${kernel_version}"
go test -exec "$script --exec-vm $input" "${args[@]}"
echo "Test successful on ${kernel_version}"

rm -r "${input}"
