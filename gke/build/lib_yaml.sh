#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

function _yq()
{
  # If there is a local version of yq, use that instead of spawning a temporary
  # container.
  if 2>/dev/null >&2 command -v yq; then
    yq "$@"
  else
    # This is version 3.4.1 of yq.
    local yq_digest="40c7256194d63079e3f9efad931909d80026400dfa72ab42c3120acd5b840184"
    local yq_image="gcr.io/gke-release-staging/mikefarah/yq@sha256:${yq_digest}"
    # The -i flag makes it so that this docker image can take in standard input.
    # That is, we can do "foo | _yq ..." and the stdout from "foo" will pass
    # through into the Dockerized 'yq' environment.
    docker run --rm \
      -i \
      -v "${PWD}":"${PWD}" \
      -w "${PWD}" \
      "${yq_image}" \
      yq "$@"
  fi
}
