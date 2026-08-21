#!/bin/bash
set -euo pipefail

# stub for the `docker` executable. it is used as CDK_DOCKER when executing unit
# tests in `test.staging.ts` It outputs the command line to
# `/tmp/docker-stub.input` and accepts one of 3 commands that impact it's
# behavior.

echo "$@" >> /tmp/docker-stub.input.concat
echo "$@" > /tmp/docker-stub.input

if echo "$@" | grep "DOCKER_STUB_SUCCESS_NO_OUTPUT"; then
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_FAIL"; then
  echo "A HUGE FAILING DOCKER STUFF"
  exit 1
fi

if echo "$@" | grep "DOCKER_STUB_SUCCESS"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  touch ${outdir}/test.txt
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_MULTIPLE_FILES"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  touch ${outdir}/test1.txt
  touch ${outdir}/test2.txt
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_SINGLE_ARCHIVE"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  touch ${outdir}/test.zip
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_SYMLINK"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  target=$(mktemp) # a real file, so the output resolves to a regular file but is itself a symlink
  ln -s ${target} ${outdir}/test.zip
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_SINGLE_FILE_WITHOUT_EXT"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  touch ${outdir}/test # create a file witout extension
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_SINGLE_FILE"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  touch ${outdir}/test.txt
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_EXEC"; then
  while [[ "$1" != "DOCKER_STUB_EXEC" ]]; do
    shift
  done
  shift

  exec "$@" # Execute what's left
fi

echo "Docker mock only supports one of the following commands: DOCKER_STUB_SUCCESS_NO_OUTPUT,DOCKER_STUB_FAIL,DOCKER_STUB_SUCCESS,DOCKER_STUB_MULTIPLE_FILES,DOCKER_SINGLE_ARCHIVE,DOCKER_STUB_EXEC, got '$@'"
exit 1
