#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "$KUBE_ROOT/hack/lib/init.sh"

# Convert a path relative to $KUBE_ROOT to a real path
localpath() {
    realpath "$KUBE_ROOT/$1"
}

# Configuration for fetching this file, relative to this repository root
ENVFILE=openshift-hack/kubensenter.env

# The source of the file, relative to the remote repository root
SOURCE=utils/kubensenter/kubensenter

# The destination of the file, relative to this repository root
DESTINATION=openshift-hack/images/hyperkube/kubensenter

usage() {
    source_env
    echo "Usage:"
    echo "  $0 [--to-latest]"
    echo
    echo "Updates the local copy of $DESTINATION as configured in $ENVFILE:"
    echo "  REPO: $REPO"
    echo "  COMMIT: $COMMIT"
    echo
    echo "Options:"
    echo "  --to-latest (or env UPDATE_TO_LATEST=1)"
    echo "    Update $ENVFILE to the latest commit or tag in $REPO configured by the TARGET entry"
    echo "    (currently \"$TARGET\"), and synchronize to the updated commit."
    echo "    - If TARGET resolves to a branch, pin to the latest commit hash from that branch"
    echo "    - If TARGET resolves to a tag, pin to the latest tag that matches that pattern"
    echo "    - TARGET may be a glob-like expression such as \"v1.1.*\" that would match any of the following:"
    echo "        v1.1.0 v1.1.3 v1.1.22-rc1"
    exit 1
}

source_env() {
    source "$(localpath "$ENVFILE")"
    # Intentionally global scope:
    REPO=${REPO:-"github.com/containers/kubensmnt"}
    COMMIT=${COMMIT:-"main"}
    TARGET=${TARGET:-"main"}
}

edit_envfile() {
    local envfile=$1
    local refname=$2

    # Shell-quote refname in case it contains any shell-special characters
    local newcommit=$(printf 'COMMIT=%q' "$refname")
    if [[ $# -gt 2 ]]; then
        shift 2
        # Add the comment suffix
        newcommit="$newcommit # $*"
    fi

    local patch
    patch=$(printf "%q" "$newcommit")
    # Note: Using ':' since it is not a valid tag character according to git-check-ref-format(1)
    sed -i "s:^COMMIT=.*:$patch:" "$envfile"
}

update_env() {
    local repouri latest refhash reftype refname
    source_env
    repouri=https://$REPO.git
    echo "Updating to latest $TARGET from $repouri"

    latest=$(git \
                   -c "versionsort.suffix=-alpha" \
                   -c "versionsort.suffix=-beta" \
                   -c "versionsort.suffix=-rc" \
                 ls-remote \
                   --heads --tags \
                   --sort='-version:refname' \
                   "$repouri" "$TARGET" \
             | head -n 1)
    if [[ -z $latest ]]; then
        echo "ERROR: No matching ref found for $TARGET"
        return 1
    fi
    refhash=$(cut -f1 <<<"$latest")
    reftype=$(cut -d/ -f2 <<<"$latest")
    refname=$(cut -d/ -f3 <<<"$latest")

    if [[ $reftype == "tags" ]]; then
        echo "  Latest tag is $refname ($refhash)"
        edit_envfile "$ENVFILE" "$refname" "($refhash)"
    else
        echo "  Latest on branch $refname is $refhash"
        edit_envfile "$ENVFILE" "$refhash"
    fi
}

do_fetch() {
    source_env
    local repohost reponame uri
    repohost=$(cut -d/ -f1 <<<"$REPO")
    reponame=${REPO#$repohost/}
    case $repohost in
        github.com)
            uri=https://raw.githubusercontent.com/$reponame/$COMMIT/$SOURCE
            ;;
        *)
            echo "No support for repositories hosted on $repohost"
            return 2
            ;;
    esac

    echo "Fetching $DESTINATION from $uri"
    curl -fsLo "$(localpath "$DESTINATION")" "$uri"
}

main() {
    local to_latest=${UPDATE_TO_LATEST:-}
    if [[ $# -gt 0 ]]; then
        if [[ $1 == "--help" || $1 == "-h" ]]; then
            usage
        elif [[ $1 == "--to-latest" ]]; then
            to_latest=1
        fi
    fi

    if [[ $to_latest ]]; then
        update_env
    fi

    do_fetch
}

# bash modulino
[[ "${BASH_SOURCE[0]}" == "$0" ]] && main "$@"
