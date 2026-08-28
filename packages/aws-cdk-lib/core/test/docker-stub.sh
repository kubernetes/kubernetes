#!/bin/bash
set -euo pipefail

# stub for the `docker` executable. it is used as CDK_DOCKER when executing unit
# tests in `test.staging.ts` It outputs the command line to
# `/tmp/docker-stub.input` and accepts one of 3 commands that impact it's
# behavior.

echo "$@" >> /tmp/docker-stub.input.concat
echo "$@" > /tmp/docker-stub.input

# Emulate a bundler that builds its output from its input: copy everything that is mounted
# at /asset-input into the directory mounted at /asset-output. Stubs that must produce a
# *directory* output add their own entries on top of that, the way a real bundler would,
# instead of making the whole output consist of the entries they synthesize.
copy_input_to_output() {
  local indir outdir
  indir=$(echo "$@" | xargs -n1 | grep ":/asset-input" | head -n1 | cut -d":" -f1)
  outdir=$(echo "$@" | xargs -n1 | grep ":/asset-output" | head -n1 | cut -d":" -f1)
  cp -R "${indir}/." "${outdir}/"
}

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

# Produces an output that is *nothing but* a symlink, so it is auto-discovered as a single
# archive. Use one of the DOCKER_STUB_DIR_WITH_* commands below for a directory output that
# merely contains a symlink.
if echo "$@" | grep "DOCKER_STUB_SYMLINK"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  target=$(mktemp) # a real file, so the output resolves to a regular file but is itself a symlink
  ln -s ${target} ${outdir}/test.zip
  exit 0
fi

# The following DOCKER_STUB_DIR_WITH_* commands bundle the input into the output directory
# and add a symlink to it. The copied input makes the output a *directory* (more than one
# entry, so it is not auto-discovered as a single archive).

if echo "$@" | grep "DOCKER_STUB_DIR_WITH_LOCAL_SYMLINK"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  copy_input_to_output "$@"
  touch ${outdir}/test.txt
  ln -s test.txt ${outdir}/local-link.txt # resolves inside the output directory
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_DIR_WITH_EXTERNAL_SYMLINK"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  copy_input_to_output "$@"
  target=$(mktemp) # a real file, but outside of the output directory
  ln -s ${target} ${outdir}/payload.zip
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_DIR_WITH_NESTED_EXTERNAL_SYMLINK"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  copy_input_to_output "$@"
  mkdir -p ${outdir}/subdir/subsubdir
  target=$(mktemp) # a real file, but outside of the output directory
  ln -s ${target} ${outdir}/subdir/subsubdir/payload.zip
  exit 0
fi

if echo "$@" | grep "DOCKER_STUB_DIR_WITH_EXTERNAL_DIR_SYMLINK"; then
  outdir=$(echo "$@" | xargs -n1 | grep "/asset-output" | head -n1 | cut -d":" -f1)
  copy_input_to_output "$@"
  target=$(mktemp -d) # a real directory, but outside of the output directory
  ln -s ${target} ${outdir}/payload
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
