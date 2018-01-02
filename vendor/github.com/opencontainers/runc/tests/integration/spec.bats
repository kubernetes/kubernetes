#!/usr/bin/env bats

load helpers

function setup() {
  # initial cleanup in case a prior test exited and did not cleanup
  cd "$INTEGRATION_ROOT"
  run rm -f -r "$HELLO_BUNDLE"

  # setup hello-world for spec generation testing
  run mkdir "$HELLO_BUNDLE"
  run mkdir "$HELLO_BUNDLE"/rootfs
  run tar -C "$HELLO_BUNDLE"/rootfs -xf "$HELLO_IMAGE"
}

function teardown() {
  cd "$INTEGRATION_ROOT"
  run rm -f -r "$HELLO_BUNDLE"
}

@test "spec generation cwd" {
  cd "$HELLO_BUNDLE"
  # note this test runs from the bundle not the integration root

  # test that config.json does not exist after the above partial setup
  [ ! -e config.json ]

  # test generation of spec does not return an error
  runc_spec
  [ "$status" -eq 0 ]

  # test generation of spec created our config.json (spec)
  [ -e config.json ]

  # test existence of required args parameter in the generated config.json
  run bash -c "grep -A2 'args' config.json | grep 'sh'"
  [[ "${output}" == *"sh"* ]]

  # change the default args parameter from sh to hello
  sed -i 's;"sh";"/hello";' config.json

  # ensure the generated spec works by running hello-world
  runc run test_hello
  [ "$status" -eq 0 ]
}

@test "spec generation --bundle" {
  # note this test runs from the integration root not the bundle

  # test that config.json does not exist after the above partial setup
  [ ! -e "$HELLO_BUNDLE"/config.json ]

  # test generation of spec does not return an error
  runc_spec --bundle "$HELLO_BUNDLE"
  [ "$status" -eq 0 ]

  # test generation of spec created our config.json (spec)
  [ -e "$HELLO_BUNDLE"/config.json ]

  # change the default args parameter from sh to hello
  sed -i 's;"sh";"/hello";' "$HELLO_BUNDLE"/config.json

  # ensure the generated spec works by running hello-world
  runc run --bundle "$HELLO_BUNDLE" test_hello
  [ "$status" -eq 0 ]
}

@test "spec validator" {
  TESTDIR=$(pwd)
  cd "$HELLO_BUNDLE"

  run git clone https://github.com/opencontainers/runtime-spec.git src/runtime-spec
  [ "$status" -eq 0 ]

  SPEC_COMMIT=$(grep '^github.com/opencontainers/runtime-spec' ${TESTDIR}/../../vendor.conf | cut -d ' ' -f 2)
  run git -C src/runtime-spec reset --hard "${SPEC_COMMIT}"

  [ "$status" -eq 0 ]
  [ -e src/runtime-spec/schema/config-schema.json ]

  run bash -c "GOPATH='$GOPATH' go get github.com/xeipuuv/gojsonschema"
  [ "$status" -eq 0 ]

  GOPATH="$GOPATH" go build src/runtime-spec/schema/validate.go
  [ -e ./validate ]

  runc spec
  [ -e config.json ]

  run ./validate src/runtime-spec/schema/config-schema.json config.json
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == *"The document is valid"* ]]
}
