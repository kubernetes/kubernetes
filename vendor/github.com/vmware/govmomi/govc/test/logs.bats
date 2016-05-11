#!/usr/bin/env bats

load test_helper

@test "logs" {
  run govc logs
  assert_success
  nlogs=${#lines[@]}
  # there should be plenty more than 1 line of hostd logs
  [ $nlogs -ge 1 ]

  # test -n flag
  run govc logs -n $((nlogs - 1))
  assert_success
  [ ${#lines[@]} -le $nlogs ]

  run govc logs -log vmkernel
  assert_success
  nlogs=${#lines[@]}
  # there should be plenty more than 1 line of vmkernel logs
  [ $nlogs -ge 1 ]

  # -host ignored against ESX
  run govc logs -host enoent
  assert_success

  run govc logs -log enoent
  assert_failure
}

@test "logs.ls" {
  run govc logs.ls
  assert_success

  # -host ignored against ESX
  run govc logs.ls -host enoent
  assert_success
}
