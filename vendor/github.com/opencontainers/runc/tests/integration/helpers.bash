#!/bin/bash

# Root directory of integration tests.
INTEGRATION_ROOT=$(dirname "$(readlink -f "$BASH_SOURCE")")
RUNC="${INTEGRATION_ROOT}/../../runc"
RECVTTY="${INTEGRATION_ROOT}/../../contrib/cmd/recvtty/recvtty"
GOPATH="$(mktemp -d --tmpdir runc-integration-gopath.XXXXXX)"

# Test data path.
TESTDATA="${INTEGRATION_ROOT}/testdata"

# Busybox image
BUSYBOX_IMAGE="$BATS_TMPDIR/busybox.tar"
BUSYBOX_BUNDLE="$BATS_TMPDIR/busyboxtest"

# hello-world in tar format
HELLO_IMAGE="$TESTDATA/hello-world.tar"
HELLO_BUNDLE="$BATS_TMPDIR/hello-world"

# CRIU PATH
CRIU="$(which criu || true)"

# Kernel version
KERNEL_VERSION="$(uname -r)"
KERNEL_MAJOR="${KERNEL_VERSION%%.*}"
KERNEL_MINOR="${KERNEL_VERSION#$KERNEL_MAJOR.}"
KERNEL_MINOR="${KERNEL_MINOR%%.*}"

# Root state path.
ROOT=$(mktemp -d "$BATS_TMPDIR/runc.XXXXXX")

# Path to console socket.
CONSOLE_SOCKET="$BATS_TMPDIR/console.sock"

# Cgroup mount
CGROUP_MEMORY_BASE_PATH=$(grep "cgroup" /proc/self/mountinfo | gawk 'toupper($NF) ~ /\<MEMORY\>/ { print $5; exit }')
CGROUP_CPU_BASE_PATH=$(grep "cgroup" /proc/self/mountinfo | gawk 'toupper($NF) ~ /\<CPU\>/ { print $5; exit }')

# CONFIG_MEMCG_KMEM support
KMEM="${CGROUP_MEMORY_BASE_PATH}/memory.kmem.limit_in_bytes"
RT_PERIOD="${CGROUP_CPU_BASE_PATH}/cpu.rt_period_us"

# Check if we're in rootless mode.
ROOTLESS=$(id -u)

# Wrapper for runc.
function runc() {
	run __runc "$@"

	# Some debug information to make life easier. bats will only print it if the
	# test failed, in which case the output is useful.
	echo "runc $@ (status=$status):" >&2
	echo "$output" >&2
}

# Raw wrapper for runc.
function __runc() {
	"$RUNC" --log /proc/self/fd/2 --root "$ROOT" "$@"
}

# Wrapper for runc spec.
function runc_spec() {
	local args=""

	if [ "$ROOTLESS" -ne 0 ]; then
		args+="--rootless"
	fi

	runc spec $args "$@"
}

# Fails the current test, providing the error given.
function fail() {
	echo "$@" >&2
	exit 1
}

# Allows a test to specify what things it requires. If the environment can't
# support it, the test is skipped with a message.
function requires() {
	for var in "$@"; do
		case $var in
		criu)
			if [ ! -e "$CRIU" ]; then
				skip "test requires ${var}"
			fi
			;;
		root)
			if [ "$ROOTLESS" -ne 0 ]; then
				skip "test requires ${var}"
			fi
			;;
		cgroups_kmem)
			if [ ! -e "$KMEM" ]; then
				skip "Test requires ${var}."
			fi
			;;
		cgroups_rt)
			if [ ! -e "$RT_PERIOD" ]; then
				skip "Test requires ${var}."
			fi
			;;
		*)
			fail "BUG: Invalid requires ${var}."
			;;
		esac
	done
}

# Retry a command $1 times until it succeeds. Wait $2 seconds between retries.
function retry() {
	local attempts=$1
	shift
	local delay=$1
	shift
	local i

	for ((i = 0; i < attempts; i++)); do
		run "$@"
		if [[ "$status" -eq 0 ]]; then
			return 0
		fi
		sleep $delay
	done

	echo "Command \"$@\" failed $attempts times. Output: $output"
	false
}

# retry until the given container has state
function wait_for_container() {
	local attempts=$1
	local delay=$2
	local cid=$3
	local i

	for ((i = 0; i < attempts; i++)); do
		runc state $cid
		if [[ "$status" -eq 0 ]]; then
			return 0
		fi
		sleep $delay
	done

	echo "runc state failed to return state $statecheck $attempts times. Output: $output"
	false
}

# retry until the given container has state
function wait_for_container_inroot() {
	local attempts=$1
	local delay=$2
	local cid=$3
	local i

	for ((i = 0; i < attempts; i++)); do
		ROOT=$4 runc state $cid
		if [[ "$status" -eq 0 ]]; then
			return 0
		fi
		sleep $delay
	done

	echo "runc state failed to return state $statecheck $attempts times. Output: $output"
	false
}

function testcontainer() {
	# test state of container
	runc state $1
	[ "$status" -eq 0 ]
	[[ "${output}" == *"$2"* ]]
}

function setup_recvtty() {
	# We need to start recvtty in the background, so we double fork in the shell.
	("$RECVTTY" --pid-file "$BATS_TMPDIR/recvtty.pid" --mode null "$CONSOLE_SOCKET" &) &
}

function teardown_recvtty() {
	# When we kill recvtty, the container will also be killed.
	if [ -f "$BATS_TMPDIR/recvtty.pid" ]; then
		kill -9 $(cat "$BATS_TMPDIR/recvtty.pid")
	fi

	# Clean up the files that might be left over.
	rm -f "$BATS_TMPDIR/recvtty.pid"
	rm -f "$CONSOLE_SOCKET"
}

function setup_busybox() {
	setup_recvtty
	run mkdir "$BUSYBOX_BUNDLE"
	run mkdir "$BUSYBOX_BUNDLE"/rootfs
	if [ -e "/testdata/busybox.tar" ]; then
		BUSYBOX_IMAGE="/testdata/busybox.tar"
	fi
	if [ ! -e $BUSYBOX_IMAGE ]; then
		curl -o $BUSYBOX_IMAGE -sSL 'https://github.com/docker-library/busybox/raw/a0558a9006ce0dd6f6ec5d56cfd3f32ebeeb815f/glibc/busybox.tar.xz'
	fi
	tar --exclude './dev/*' -C "$BUSYBOX_BUNDLE"/rootfs -xf "$BUSYBOX_IMAGE"
	cd "$BUSYBOX_BUNDLE"
	runc_spec
}

function setup_hello() {
	setup_recvtty
	run mkdir "$HELLO_BUNDLE"
	run mkdir "$HELLO_BUNDLE"/rootfs
	tar --exclude './dev/*' -C "$HELLO_BUNDLE"/rootfs -xf "$HELLO_IMAGE"
	cd "$HELLO_BUNDLE"
	runc_spec
	sed -i 's;"sh";"/hello";' config.json
}

function teardown_running_container() {
	runc list
	# $1 should be a container name such as "test_busybox"
	# here we detect "test_busybox "(with one extra blank) to avoid conflict prefix
	# e.g. "test_busybox" and "test_busybox_update"
	if [[ "${output}" == *"$1 "* ]]; then
		runc kill $1 KILL
		retry 10 1 eval "__runc state '$1' | grep -q 'stopped'"
		runc delete $1
	fi
}

function teardown_running_container_inroot() {
	ROOT=$2 runc list
	# $1 should be a container name such as "test_busybox"
	# here we detect "test_busybox "(with one extra blank) to avoid conflict prefix
	# e.g. "test_busybox" and "test_busybox_update"
	if [[ "${output}" == *"$1 "* ]]; then
		ROOT=$2 runc kill $1 KILL
		retry 10 1 eval "ROOT='$2' __runc state '$1' | grep -q 'stopped'"
		ROOT=$2 runc delete $1
	fi
}

function teardown_busybox() {
	cd "$INTEGRATION_ROOT"
	teardown_recvtty
	teardown_running_container test_busybox
	run rm -f -r "$BUSYBOX_BUNDLE"
}

function teardown_hello() {
	cd "$INTEGRATION_ROOT"
	teardown_recvtty
	teardown_running_container test_hello
	run rm -f -r "$HELLO_BUNDLE"
}
