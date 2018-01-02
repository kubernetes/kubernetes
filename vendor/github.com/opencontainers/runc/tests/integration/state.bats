#!/usr/bin/env bats

load helpers

function setup() {
  teardown_busybox
  setup_busybox
}

function teardown() {
  teardown_busybox
}

@test "state (kill + delete)" {
  runc state test_busybox
  [ "$status" -ne 0 ]

  # run busybox detached
  runc run -d --console-socket $CONSOLE_SOCKET test_busybox
  [ "$status" -eq 0 ]

  # check state
  testcontainer test_busybox running

  runc kill test_busybox KILL
  [ "$status" -eq 0 ]

  # wait for busybox to be in the destroyed state
  retry 10 1 eval "__runc state test_busybox | grep -q 'stopped'"

  # delete test_busybox
  runc delete test_busybox
  [ "$status" -eq 0 ]

  runc state test_busybox
  [ "$status" -ne 0 ]
}

@test "state (pause + resume)" {
  # XXX: pause and resume require cgroups.
  requires root

  runc state test_busybox
  [ "$status" -ne 0 ]

  # run busybox detached
  runc run -d --console-socket $CONSOLE_SOCKET test_busybox
  [ "$status" -eq 0 ]

  # check state
  testcontainer test_busybox running

  # pause busybox
  runc pause test_busybox
  [ "$status" -eq 0 ]

  # test state of busybox is paused
  testcontainer test_busybox paused

  # resume busybox
  runc resume test_busybox
  [ "$status" -eq 0 ]

  # test state of busybox is back to running
  testcontainer test_busybox running
}
