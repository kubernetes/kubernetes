#!/usr/bin/env bats

load helpers

function setup() {
  teardown_running_container_inroot test_dotbox $HELLO_BUNDLE
  teardown_busybox
  setup_busybox
}

function teardown() {
  teardown_running_container_inroot test_dotbox $HELLO_BUNDLE
  teardown_busybox
}

@test "global --root" {
  # run busybox detached using $HELLO_BUNDLE for state
  ROOT=$HELLO_BUNDLE runc run -d --console-socket $CONSOLE_SOCKET test_dotbox
  [ "$status" -eq 0 ]

  # run busybox detached in default root
  runc run -d --console-socket $CONSOLE_SOCKET test_busybox
  [ "$status" -eq 0 ]

  runc state test_busybox
  [ "$status" -eq 0 ]
  [[ "${output}" == *"running"* ]]

  ROOT=$HELLO_BUNDLE runc state test_dotbox
  [ "$status" -eq 0 ]
  [[ "${output}" == *"running"* ]]

  ROOT=$HELLO_BUNDLE runc state test_busybox
  [ "$status" -ne 0 ]

  runc state test_dotbox
  [ "$status" -ne 0 ]

  runc kill test_busybox KILL
  [ "$status" -eq 0 ]
  retry 10 1 eval "__runc state test_busybox | grep -q 'stopped'"
  runc delete test_busybox
  [ "$status" -eq 0 ]

  ROOT=$HELLO_BUNDLE runc kill test_dotbox KILL
  [ "$status" -eq 0 ]
  retry 10 1 eval "ROOT='$HELLO_BUNDLE' __runc state test_dotbox | grep -q 'stopped'"
  ROOT=$HELLO_BUNDLE runc delete test_dotbox
  [ "$status" -eq 0 ]
}
