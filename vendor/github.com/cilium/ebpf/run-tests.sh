#!/usr/bin/env bash
# Test the current package under a different kernel.
# Requires virtme and qemu to be installed.
# Examples:
#     Run all tests on a 5.4 kernel
#     $ ./run-tests.sh 5.4
#     Run a subset of tests:
#     $ ./run-tests.sh 5.4 ./link
#     Run using a local kernel image
#     $ ./run-tests.sh /path/to/bzImage

set -euo pipefail

script="$(realpath "$0")"
readonly script

source "$(dirname "$script")/testdata/sh/lib.sh"

quote_env() {
  for var in "$@"; do
    if [ -v "$var" ]; then
      printf "%s=%q " "$var" "${!var}"
    fi
  done
}

declare -a preserved_env=(
  PATH
  CI_MAX_KERNEL_VERSION
  TEST_SEED
  KERNEL_VERSION
)

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

  if ! $sudo virtme-run --kimg "${input}/boot/vmlinuz" --cpus 2 --memory 1G --pwd \
    --rwdir="${testdir}=${testdir}" \
    --rodir=/run/input="${input}" \
    --rwdir=/run/output="${output}" \
    --script-sh "$(quote_env "${preserved_env[@]}") \"$script\" \
    --exec-test $cmd"; then
    exit 23
  fi

  if ! [[ -e "${output}/status" ]]; then
    exit 42
  fi

  rc=$(<"${output}/status")
  $sudo rm -r "$output"
  exit "$rc"
elif [[ "${1:-}" = "--exec-test" ]]; then
  shift

  mount -t bpf bpf /sys/fs/bpf
  mount -t tracefs tracefs /sys/kernel/debug/tracing

  if [[ -d "/run/input/usr/src/linux/tools/testing/selftests/bpf" ]]; then
    export KERNEL_SELFTESTS="/run/input/usr/src/linux/tools/testing/selftests/bpf"
  fi

  if [[ -d "/run/input/lib/modules" ]]; then
    find /run/input/lib/modules -type f -name bpf_testmod.ko -exec insmod {} \;
  fi

  dmesg --clear
  rc=0
  "$@" || rc=$?
  dmesg
  echo $rc > "/run/output/status"
  exit $rc # this return code is "swallowed" by qemu
fi

if [[ -z "${1:-}" ]]; then
  echo "Expecting kernel version or path as first argument"
  exit 1
fi

input="$(mktemp -d)"
readonly input

if [[ -f "${1}" ]]; then
  # First argument is a local file.
  readonly kernel="${1}"
  cp "${1}" "${input}/boot/vmlinuz"
else
  readonly kernel="${1}"

  # LINUX_VERSION_CODE test compares this to discovered value.
  export KERNEL_VERSION="${1}"

  if ! extract_oci_image "ghcr.io/cilium/ci-kernels:${kernel}-selftests" "${input}"; then
    extract_oci_image "ghcr.io/cilium/ci-kernels:${kernel}" "${input}"
  fi
fi
shift

args=(-short -coverpkg=./... -coverprofile=coverage.out -count 1 ./...)
if (( $# > 0 )); then
  args=("$@")
fi

export GOFLAGS=-mod=readonly
export CGO_ENABLED=0

echo Testing on "${kernel}"
go test -exec "$script --exec-vm $input" "${args[@]}"
echo "Test successful on ${kernel}"

rm -r "${input}"
