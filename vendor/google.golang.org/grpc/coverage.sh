#!/usr/bin/env bash


set -e

workdir=.cover
profile="$workdir/cover.out"
mode=set
end2endtest="google.golang.org/grpc/test"

generate_cover_data() {
    rm -rf "$workdir"
    mkdir "$workdir"

    for pkg in "$@"; do
        if [ $pkg == "google.golang.org/grpc" -o $pkg == "google.golang.org/grpc/transport" -o $pkg == "google.golang.org/grpc/metadata" -o $pkg == "google.golang.org/grpc/credentials" ]
            then
                f="$workdir/$(echo $pkg | tr / -)"
                go test -covermode="$mode" -coverprofile="$f.cover" "$pkg"
                go test -covermode="$mode" -coverpkg "$pkg" -coverprofile="$f.e2e.cover" "$end2endtest"
        fi
    done

    echo "mode: $mode" >"$profile"
    grep -h -v "^mode:" "$workdir"/*.cover >>"$profile"
}

show_cover_report() {
    go tool cover -${1}="$profile"
}

push_to_coveralls() {
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
    echo >&2 "error: invalid option: $1" ;;
esac
rm -rf "$workdir"
