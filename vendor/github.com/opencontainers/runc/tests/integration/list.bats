#!/usr/bin/env bats

load helpers

function setup() {
  teardown_running_container_inroot test_box1 $HELLO_BUNDLE
  teardown_running_container_inroot test_box2 $HELLO_BUNDLE
  teardown_running_container_inroot test_box3 $HELLO_BUNDLE
  teardown_busybox
  setup_busybox
}

function teardown() {
  teardown_running_container_inroot test_box1 $HELLO_BUNDLE
  teardown_running_container_inroot test_box2 $HELLO_BUNDLE
  teardown_running_container_inroot test_box3 $HELLO_BUNDLE
  teardown_busybox
}

@test "list" {
  # run a few busyboxes detached
  ROOT=$HELLO_BUNDLE runc run -d --console-socket $CONSOLE_SOCKET test_box1
  [ "$status" -eq 0 ]

  ROOT=$HELLO_BUNDLE runc run -d --console-socket $CONSOLE_SOCKET test_box2
  [ "$status" -eq 0 ]

  ROOT=$HELLO_BUNDLE runc run -d --console-socket $CONSOLE_SOCKET test_box3
  [ "$status" -eq 0 ]

  ROOT=$HELLO_BUNDLE runc list
  [ "$status" -eq 0 ]
  [[ ${lines[0]} =~ ID\ +PID\ +STATUS\ +BUNDLE\ +CREATED+ ]]
  [[ "${lines[1]}" == *"test_box1"*[0-9]*"running"*$BUSYBOX_BUNDLE*[0-9]* ]]
  [[ "${lines[2]}" == *"test_box2"*[0-9]*"running"*$BUSYBOX_BUNDLE*[0-9]* ]]
  [[ "${lines[3]}" == *"test_box3"*[0-9]*"running"*$BUSYBOX_BUNDLE*[0-9]* ]]

  ROOT=$HELLO_BUNDLE runc list -q
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "test_box1" ]]
  [[ "${lines[1]}" == "test_box2" ]]
  [[ "${lines[2]}" == "test_box3" ]]

  ROOT=$HELLO_BUNDLE runc list --format table
  [ "$status" -eq 0 ]
  [[ ${lines[0]} =~ ID\ +PID\ +STATUS\ +BUNDLE\ +CREATED+ ]]
  [[ "${lines[1]}" == *"test_box1"*[0-9]*"running"*$BUSYBOX_BUNDLE*[0-9]* ]]
  [[ "${lines[2]}" == *"test_box2"*[0-9]*"running"*$BUSYBOX_BUNDLE*[0-9]* ]]
  [[ "${lines[3]}" == *"test_box3"*[0-9]*"running"*$BUSYBOX_BUNDLE*[0-9]* ]]

  ROOT=$HELLO_BUNDLE runc list --format json
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == [\[][\{]"\"ociVersion\""[:]"\""*[0-9][\.]*[0-9][\.]*[0-9]*"\""[,]"\"id\""[:]"\"test_box1\""[,]"\"pid\""[:]*[0-9][,]"\"status\""[:]*"\"running\""[,]"\"bundle\""[:]*$BUSYBOX_BUNDLE*[,]"\"rootfs\""[:]"\""*"\""[,]"\"created\""[:]*[0-9]*[\}]* ]]
  [[ "${lines[0]}" == *[,][\{]"\"ociVersion\""[:]"\""*[0-9][\.]*[0-9][\.]*[0-9]*"\""[,]"\"id\""[:]"\"test_box2\""[,]"\"pid\""[:]*[0-9][,]"\"status\""[:]*"\"running\""[,]"\"bundle\""[:]*$BUSYBOX_BUNDLE*[,]"\"rootfs\""[:]"\""*"\""[,]"\"created\""[:]*[0-9]*[\}]* ]]
  [[ "${lines[0]}" == *[,][\{]"\"ociVersion\""[:]"\""*[0-9][\.]*[0-9][\.]*[0-9]*"\""[,]"\"id\""[:]"\"test_box3\""[,]"\"pid\""[:]*[0-9][,]"\"status\""[:]*"\"running\""[,]"\"bundle\""[:]*$BUSYBOX_BUNDLE*[,]"\"rootfs\""[:]"\""*"\""[,]"\"created\""[:]*[0-9]*[\}][\]] ]]
}
