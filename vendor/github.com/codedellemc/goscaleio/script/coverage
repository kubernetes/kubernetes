#!/bin/sh
# Generate test coverage statistics for Go packages.
#
# Works around the fact that `go test -coverprofile` currently does not work
# with multiple packages, see https://code.google.com/p/go/issues/detail?id=6909
#
# Usage: script/coverage [--html|--coveralls]
#
#     --html      Additionally create HTML report and open it in browser
#     --coveralls Push coverage statistics to coveralls.io
#

set -e

workdir=.cover
profile="$workdir/cover.out"
mode=count

generate_cover_data() {
    rm -rf "$workdir"
    mkdir "$workdir"

    for pkg in "$@"; do
        f="$workdir/$(echo $pkg | tr / -).cover"
        go test -covermode="$mode" -coverprofile="$f" "$pkg"
    done

    echo "mode: $mode" >"$profile"
    grep -h -v "^mode:" "$workdir"/*.cover >>"$profile"
}

show_cover_report() {
    go tool cover -${1}="$profile"
}

push_to_coveralls() {
    echo "Pushing coverage statistics to coveralls.io"
    goveralls -coverprofile="$profile"
}

generate_cover_data $(go list ./...)
show_cover_report func
case "$1" in
"")
    ;;
--html)
    show_cover_report html ;;
--coveralls)
    push_to_coveralls ;;
*)
    echo >&2 "error: invalid option: $1"; exit 1 ;;
esac