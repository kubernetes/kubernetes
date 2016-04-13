#!/usr/bin/env bash
set -e

# This script builds various binary artifacts from a checkout of the docker
# source code.
#
# Requirements:
# - The current directory should be a checkout of the docker source code
#   (https://github.com/docker/docker). Whatever version is checked out
#   will be built.
# - The VERSION file, at the root of the repository, should exist, and
#   will be used as Docker binary version and package version.
# - The hash of the git commit will also be included in the Docker binary,
#   with the suffix -dirty if the repository isn't clean.
# - The script is intented to be run inside the docker container specified
#   in the Dockerfile at the root of the source. In other words:
#   DO NOT CALL THIS SCRIPT DIRECTLY.
# - The right way to call this script is to invoke "make" from
#   your checkout of the Docker repository.
#   the Makefile will do a "docker build -t docker ." and then
#   "docker run hack/make.sh" in the resulting image.
#

set -o pipefail

export DOCKER_PKG='github.com/docker/docker'
export SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export MAKEDIR="$SCRIPTDIR/make"

# We're a nice, sexy, little shell script, and people might try to run us;
# but really, they shouldn't. We want to be in a container!
if [ "$PWD" != "/go/src/$DOCKER_PKG" ] || [ -z "$DOCKER_CROSSPLATFORMS" ]; then
	{
		echo "# WARNING! I don't seem to be running in the Docker container."
		echo "# The result of this command might be an incorrect build, and will not be"
		echo "#   officially supported."
		echo "#"
		echo "# Try this instead: make all"
		echo "#"
	} >&2
fi

echo

# List of bundles to create when no argument is passed
DEFAULT_BUNDLES=(
	validate-dco
	validate-gofmt
	validate-pkg
	validate-test
	validate-toml
	validate-vet

	binary

	test-unit
	test-integration-cli
	test-docker-py

	dynbinary

	cover
	cross
	tgz
	ubuntu
)

VERSION=$(< ./VERSION)
if command -v git &> /dev/null && git rev-parse &> /dev/null; then
	GITCOMMIT=$(git rev-parse --short HEAD)
	if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
		GITCOMMIT="$GITCOMMIT-dirty"
	fi
	BUILDTIME=$(date -u)
elif [ "$DOCKER_GITCOMMIT" ]; then
	GITCOMMIT="$DOCKER_GITCOMMIT"
else
	echo >&2 'error: .git directory missing and DOCKER_GITCOMMIT not specified'
	echo >&2 '  Please either build with the .git directory accessible, or specify the'
	echo >&2 '  exact (--short) commit hash you are building using DOCKER_GITCOMMIT for'
	echo >&2 '  future accountability in diagnosing build issues.  Thanks!'
	exit 1
fi

if [ "$AUTO_GOPATH" ]; then
	rm -rf .gopath
	mkdir -p .gopath/src/"$(dirname "${DOCKER_PKG}")"
	ln -sf ../../../.. .gopath/src/"${DOCKER_PKG}"
	export GOPATH="${PWD}/.gopath:${PWD}/vendor"
fi

if [ ! "$GOPATH" ]; then
	echo >&2 'error: missing GOPATH; please see https://golang.org/doc/code.html#GOPATH'
	echo >&2 '  alternatively, set AUTO_GOPATH=1'
	exit 1
fi

if [ "$DOCKER_EXPERIMENTAL" ]; then
	echo >&2 '# WARNING! DOCKER_EXPERIMENTAL is set: building experimental features'
	echo >&2
	DOCKER_BUILDTAGS+=" experimental"
fi

if [ -z "$DOCKER_CLIENTONLY" ]; then
	DOCKER_BUILDTAGS+=" daemon"
fi

if [ "$DOCKER_EXECDRIVER" = 'lxc' ]; then
	DOCKER_BUILDTAGS+=' test_no_exec'
fi

# test whether "btrfs/version.h" exists and apply btrfs_noversion appropriately
if \
	command -v gcc &> /dev/null \
	&& ! gcc -E - &> /dev/null <<<'#include <btrfs/version.h>' \
; then
	DOCKER_BUILDTAGS+=' btrfs_noversion'
fi

# test whether "libdevmapper.h" is new enough to support deferred remove
# functionality.
if \
	command -v gcc &> /dev/null \
	&& ! ( echo -e  '#include <libdevmapper.h>\nint main() { dm_task_deferred_remove(NULL); }'| gcc -ldevmapper -xc - &> /dev/null ) \
; then
       DOCKER_BUILDTAGS+=' libdm_no_deferred_remove'
fi

# Use these flags when compiling the tests and final binary

IAMSTATIC='true'
source "$SCRIPTDIR/make/.go-autogen"
if [ -z "$DOCKER_DEBUG" ]; then
	LDFLAGS='-w'
fi

LDFLAGS_STATIC='-linkmode external'
# Cgo -H windows is incompatible with -linkmode external.
if [ "$(go env GOOS)" == 'windows' ]; then
	LDFLAGS_STATIC=''
fi
EXTLDFLAGS_STATIC='-static'
# ORIG_BUILDFLAGS is necessary for the cross target which cannot always build
# with options like -race.
ORIG_BUILDFLAGS=( -a -tags "netgo static_build $DOCKER_BUILDTAGS" -installsuffix netgo )
# see https://github.com/golang/go/issues/9369#issuecomment-69864440 for why -installsuffix is necessary here
BUILDFLAGS=( $BUILDFLAGS "${ORIG_BUILDFLAGS[@]}" )
# Test timeout.
: ${TIMEOUT:=60m}
TESTFLAGS+=" -test.timeout=${TIMEOUT}"

# A few more flags that are specific just to building a completely-static binary (see hack/make/binary)
# PLEASE do not use these anywhere else.
EXTLDFLAGS_STATIC_DOCKER="$EXTLDFLAGS_STATIC -lpthread -Wl,--unresolved-symbols=ignore-in-object-files"
LDFLAGS_STATIC_DOCKER="
	$LDFLAGS_STATIC
	-extldflags \"$EXTLDFLAGS_STATIC_DOCKER\"
"

if [ "$(uname -s)" = 'FreeBSD' ]; then
	# Tell cgo the compiler is Clang, not GCC
	# https://code.google.com/p/go/source/browse/src/cmd/cgo/gcc.go?spec=svne77e74371f2340ee08622ce602e9f7b15f29d8d3&r=e6794866ebeba2bf8818b9261b54e2eef1c9e588#752
	export CC=clang

	# "-extld clang" is a workaround for
	# https://code.google.com/p/go/issues/detail?id=6845
	LDFLAGS="$LDFLAGS -extld clang"
fi

# If sqlite3.h doesn't exist under /usr/include,
# check /usr/local/include also just in case
# (e.g. FreeBSD Ports installs it under the directory)
if [ ! -e /usr/include/sqlite3.h ] && [ -e /usr/local/include/sqlite3.h ]; then
	export CGO_CFLAGS='-I/usr/local/include'
	export CGO_LDFLAGS='-L/usr/local/lib'
fi

HAVE_GO_TEST_COVER=
if \
	go help testflag | grep -- -cover > /dev/null \
	&& go tool -n cover > /dev/null 2>&1 \
; then
	HAVE_GO_TEST_COVER=1
fi

# If $TESTFLAGS is set in the environment, it is passed as extra arguments to 'go test'.
# You can use this to select certain tests to run, eg.
#
#     TESTFLAGS='-test.run ^TestBuild$' ./hack/make.sh test-unit
#
# For integration-cli test, we use [gocheck](https://labix.org/gocheck), if you want
# to run certain tests on your local host, you should run with command:
#
#     TESTFLAGS='-check.f DockerSuite.TestBuild*' ./hack/make.sh binary test-integration-cli
#
go_test_dir() {
	dir=$1
	coverpkg=$2
	testcover=()
	if [ "$HAVE_GO_TEST_COVER" ]; then
		# if our current go install has -cover, we want to use it :)
		mkdir -p "$DEST/coverprofiles"
		coverprofile="docker${dir#.}"
		coverprofile="$ABS_DEST/coverprofiles/${coverprofile//\//-}"
		testcover=( -cover -coverprofile "$coverprofile" $coverpkg )
	fi
	(
		echo '+ go test' $TESTFLAGS "${DOCKER_PKG}${dir#.}"
		cd "$dir"
		export DEST="$ABS_DEST" # we're in a subshell, so this is safe -- our integration-cli tests need DEST, and "cd" screws it up
		test_env go test ${testcover[@]} -ldflags "$LDFLAGS" "${BUILDFLAGS[@]}" $TESTFLAGS
	)
}
test_env() {
	# use "env -i" to tightly control the environment variables that bleed into the tests
	env -i \
		DEST="$DEST" \
		DOCKER_EXECDRIVER="$DOCKER_EXECDRIVER" \
		DOCKER_GRAPHDRIVER="$DOCKER_GRAPHDRIVER" \
		DOCKER_USERLANDPROXY="$DOCKER_USERLANDPROXY" \
		DOCKER_HOST="$DOCKER_HOST" \
		GOPATH="$GOPATH" \
		HOME="$ABS_DEST/fake-HOME" \
		PATH="$PATH" \
		TEMP="$TEMP" \
		TEST_DOCKERINIT_PATH="$TEST_DOCKERINIT_PATH" \
		"$@"
}

# a helper to provide ".exe" when it's appropriate
binary_extension() {
	if [ "$(go env GOOS)" = 'windows' ]; then
		echo -n '.exe'
	fi
}

# This helper function walks the current directory looking for directories
# holding certain files ($1 parameter), and prints their paths on standard
# output, one per line.
find_dirs() {
	find . -not \( \
		\( \
			-path './vendor/*' \
			-o -path './integration-cli/*' \
			-o -path './contrib/*' \
			-o -path './pkg/mflag/example/*' \
			-o -path './.git/*' \
			-o -path './bundles/*' \
			-o -path './docs/*' \
			-o -path './pkg/libcontainer/nsinit/*' \
		\) \
		-prune \
	\) -name "$1" -print0 | xargs -0n1 dirname | sort -u
}

hash_files() {
	while [ $# -gt 0 ]; do
		f="$1"
		shift
		dir="$(dirname "$f")"
		base="$(basename "$f")"
		for hashAlgo in md5 sha256; do
			if command -v "${hashAlgo}sum" &> /dev/null; then
				(
					# subshell and cd so that we get output files like:
					#   $HASH docker-$VERSION
					# instead of:
					#   $HASH /go/src/github.com/.../$VERSION/binary/docker-$VERSION
					cd "$dir"
					"${hashAlgo}sum" "$base" > "$base.$hashAlgo"
				)
			fi
		done
	done
}

bundle() {
	local bundle="$1"; shift
	echo "---> Making bundle: $(basename "$bundle") (in $DEST)"
	source "$SCRIPTDIR/make/$bundle" "$@"
}

main() {
	# We want this to fail if the bundles already exist and cannot be removed.
	# This is to avoid mixing bundles from different versions of the code.
	mkdir -p bundles
	if [ -e "bundles/$VERSION" ]; then
		echo "bundles/$VERSION already exists. Removing."
		rm -fr "bundles/$VERSION" && mkdir "bundles/$VERSION" || exit 1
		echo
	fi

	if [ "$(go env GOHOSTOS)" != 'windows' ]; then
		# Windows and symlinks don't get along well

		rm -f bundles/latest
		ln -s "$VERSION" bundles/latest
	fi

	if [ $# -lt 1 ]; then
		bundles=(${DEFAULT_BUNDLES[@]})
	else
		bundles=($@)
	fi
	for bundle in ${bundles[@]}; do
		export DEST="bundles/$VERSION/$(basename "$bundle")"
		# Cygdrive paths don't play well with go build -o.
		if [[ "$(uname -s)" == CYGWIN* ]]; then
			export DEST="$(cygpath -mw "$DEST")"
		fi
		mkdir -p "$DEST"
		ABS_DEST="$(cd "$DEST" && pwd -P)"
		bundle "$bundle"
		echo
	done
}

main "$@"
