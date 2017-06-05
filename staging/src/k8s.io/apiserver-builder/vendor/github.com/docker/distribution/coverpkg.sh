#!/usr/bin/env bash
# Given a subpackage and the containing package, figures out which packages
# need to be passed to `go test -coverpkg`:  this includes all of the
# subpackage's dependencies within the containing package, as well as the
# subpackage itself.
DEPENDENCIES="$(go list -f $'{{range $f := .Deps}}{{$f}}\n{{end}}' ${1} | grep ${2} | grep -v github.com/docker/distribution/vendor)"
echo "${1} ${DEPENDENCIES}" | xargs echo -n | tr ' ' ','
