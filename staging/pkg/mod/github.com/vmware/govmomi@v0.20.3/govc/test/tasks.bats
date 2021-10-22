#!/usr/bin/env bats

load test_helper

@test "tasks" {
  esx_env

  run govc tasks
  assert_success
}

@test "tasks host" {
  esx_env

  run govc tasks 'host/*'
  assert_success
}
