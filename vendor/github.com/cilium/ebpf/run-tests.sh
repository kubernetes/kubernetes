#!/bin/bash
# Test the current package under a different kernel.
# Requires virtme and qemu to be installed.

set -eu
set -o pipefail

if [[ "${1:-}" = "--in-vm" ]]; then
  shift

  mount -t bpf bpf /sys/fs/bpf
  mount -t tracefs tracefs /sys/kernel/debug/tracing
  export CGO_ENABLED=0
  export GOFLAGS=-mod=readonly
  export GOPATH=/run/go-path
  export GOPROXY=file:///run/go-path/pkg/mod/cache/download
  export GOSUMDB=off
  export GOCACHE=/run/go-cache

  if [[ -d "/run/input/bpf" ]]; then
    export KERNEL_SELFTESTS="/run/input/bpf"
  fi

  readonly output="${1}"
  shift

  echo Running tests...
  go test -v -coverpkg=./... -coverprofile="$output/coverage.txt" -count 1 ./...
  touch "$output/success"
  exit 0
fi

# Pull all dependencies, so that we can run tests without the
# vm having network access.
go mod download

# Use sudo if /dev/kvm isn't accessible by the current user.
sudo=""
if [[ ! -r /dev/kvm || ! -w /dev/kvm ]]; then
  sudo="sudo"
fi
readonly sudo

readonly kernel_version="${1:-}"
if [[ -z "${kernel_version}" ]]; then
  echo "Expecting kernel version as first argument"
  exit 1
fi

readonly kernel="linux-${kernel_version}.bz"
readonly selftests="linux-${kernel_version}-selftests-bpf.bz"
readonly input="$(mktemp -d)"
readonly output="$(mktemp -d)"
readonly tmp_dir="${TMPDIR:-/tmp}"
readonly branch="${BRANCH:-master}"

fetch() {
    echo Fetching "${1}"
    wget -nv -N -P "${tmp_dir}" "https://github.com/cilium/ci-kernels/raw/${branch}/${1}"
}

fetch "${kernel}"

if fetch "${selftests}"; then
  mkdir "${input}/bpf"
  tar --strip-components=4 -xjf "${tmp_dir}/${selftests}" -C "${input}/bpf"
else
  echo "No selftests found, disabling"
fi

echo Testing on "${kernel_version}"
$sudo virtme-run --kimg "${tmp_dir}/${kernel}" --memory 512M --pwd \
  --rw \
  --rwdir=/run/input="${input}" \
  --rwdir=/run/output="${output}" \
  --rodir=/run/go-path="$(go env GOPATH)" \
  --rwdir=/run/go-cache="$(go env GOCACHE)" \
  --script-sh "PATH=\"$PATH\" $(realpath "$0") --in-vm /run/output" \
  --qemu-opts -smp 2 # need at least two CPUs for some tests

if [[ ! -e "${output}/success" ]]; then
  echo "Test failed on ${kernel_version}"
  exit 1
else
  echo "Test successful on ${kernel_version}"
  if [[ -v COVERALLS_TOKEN ]]; then
    goveralls -coverprofile="${output}/coverage.txt" -service=semaphore -repotoken "$COVERALLS_TOKEN"
  fi
fi

$sudo rm -r "${input}"
$sudo rm -r "${output}"
