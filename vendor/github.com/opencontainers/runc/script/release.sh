#!/bin/bash
# Copyright (C) 2017 SUSE LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

## --->
# Project-specific options and functions. In *theory* you shouldn't need to
# touch anything else in this script in order to use this elsewhere.
project="runc"
root="$(readlink -f "$(dirname "${BASH_SOURCE}")/..")"

# This function takes an output path as an argument, where the built
# (preferably static) binary should be placed.
function build_project() {
	builddir="$(dirname "$1")"

	# Build with all tags enabled.
	make -C "$root" COMMIT_NO= BUILDTAGS="seccomp selinux apparmor" static
	mv "$root/$project" "$1"
}

# End of the easy-to-configure portion.
## <---

# Print usage information.
function usage() {
	echo "usage: release.sh [-S <gpg-key-id>] [-c <commit-ish>] [-r <release-dir>] [-v <version>]" >&2
	exit 1
}

# Log something to stderr.
function log() {
	echo "[*] $*" >&2
}

# Log something to stderr and then exit with 0.
function bail() {
	log "$@"
	exit 0
}

# Conduct a sanity-check to make sure that GPG provided with the given
# arguments can sign something. Inability to sign things is not a fatal error.
function gpg_cansign() {
	gpg "$@" --clear-sign </dev/null >/dev/null
}

# When creating releases we need to build static binaries, an archive of the
# current commit, and generate detached signatures for both.
keyid=""
commit="HEAD"
version=""
releasedir=""
hashcmd=""
while getopts "S:c:r:v:h:" opt; do
	case "$opt" in
		S)
			keyid="$OPTARG"
			;;
		c)
			commit="$OPTARG"
			;;
		r)
			releasedir="$OPTARG"
			;;
		v)
			version="$OPTARG"
			;;
		h)
			hashcmd="$OPTARG"
			;;
		\:)
			echo "Missing argument: -$OPTARG" >&2
			usage
			;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			usage
			;;
	esac
done

version="${version:-$(<"$root/VERSION")}"
releasedir="${releasedir:-release/$version}"
hashcmd="${hashcmd:-sha256sum}"
goarch="$(go env GOARCH || echo "amd64")"

log "creating $project release in '$releasedir'"
log "  version: $version"
log "   commit: $commit"
log "      key: ${keyid:-DEFAULT}"
log "     hash: $hashcmd"

# Make explicit what we're doing.
set -x

# Make the release directory.
rm -rf "$releasedir" && mkdir -p "$releasedir"

# Build project.
build_project "$releasedir/$project.$goarch"

# Generate new archive.
git archive --format=tar --prefix="$project-$version/" "$commit" | xz > "$releasedir/$project.tar.xz"

# Generate sha256 checksums for both.
( cd "$releasedir" ; "$hashcmd" "$project".{"$goarch",tar.xz} > "$project.$hashcmd" ; )

# Set up the gpgflags.
[[ "$keyid" ]] && export gpgflags="--default-key $keyid"
gpg_cansign $gpgflags || bail "Could not find suitable GPG key, skipping signing step."

# Sign everything.
gpg $gpgflags --detach-sign --armor "$releasedir/$project.$goarch"
gpg $gpgflags --detach-sign --armor "$releasedir/$project.tar.xz"
gpg $gpgflags --clear-sign --armor \
	--output "$releasedir/$project.$hashcmd"{.tmp,} && \
	mv "$releasedir/$project.$hashcmd"{.tmp,}
