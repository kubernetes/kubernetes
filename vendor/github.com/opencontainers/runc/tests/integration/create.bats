#!/usr/bin/env bats

load helpers

function setup() {
  teardown_busybox
  setup_busybox
}

function teardown() {
  teardown_busybox
}

@test "runc create" {
  runc create --console-socket $CONSOLE_SOCKET test_busybox
  [ "$status" -eq 0 ]

  testcontainer test_busybox created

  # start the command
  runc start test_busybox
  [ "$status" -eq 0 ]

  testcontainer test_busybox running
}

@test "runc create exec" {
  runc create --console-socket $CONSOLE_SOCKET test_busybox
  [ "$status" -eq 0 ]

  testcontainer test_busybox created

  runc exec test_busybox true
  [ "$status" -eq 0 ]

  testcontainer test_busybox created

  # start the command
  runc start test_busybox
  [ "$status" -eq 0 ]

  testcontainer test_busybox running
}

@test "runc create --pid-file" {
  runc create --pid-file pid.txt --console-socket $CONSOLE_SOCKET test_busybox
  [ "$status" -eq 0 ]

  testcontainer test_busybox created

  # check pid.txt was generated
  [ -e pid.txt ]

  run cat pid.txt
  [ "$status" -eq 0 ]
  [[ ${lines[0]} == $(__runc state test_busybox | jq '.pid') ]]

  # start the command
  runc start test_busybox
  [ "$status" -eq 0 ]

  testcontainer test_busybox running
}

@test "runc create --pid-file with new CWD" {
  # create pid_file directory as the CWD
  run mkdir pid_file
  [ "$status" -eq 0 ]
  run cd pid_file
  [ "$status" -eq 0 ]

  runc create --pid-file pid.txt -b $BUSYBOX_BUNDLE --console-socket $CONSOLE_SOCKET  test_busybox
  [ "$status" -eq 0 ]

  testcontainer test_busybox created

  # check pid.txt was generated
  [ -e pid.txt ]

  run cat pid.txt
  [ "$status" -eq 0 ]
  [[ ${lines[0]} == $(__runc state test_busybox | jq '.pid') ]]

  # start the command
  runc start test_busybox
  [ "$status" -eq 0 ]

  testcontainer test_busybox running
}
