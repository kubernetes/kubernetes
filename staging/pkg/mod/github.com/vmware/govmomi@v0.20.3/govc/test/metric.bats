#!/usr/bin/env bats

load test_helper

@test "metric.ls" {
  esx_env

  run govc metric.ls
  assert_failure

  run govc metric.ls enoent
  assert_failure

  host=$(govc ls -t HostSystem ./... | head -n 1)
  pool=$(govc ls -t ResourcePool ./... | head -n 1)

  run govc metric.ls "$host"
  assert_success

  run govc metric.ls -json "$host"
  assert_success

  run govc metric.ls "$pool"
  assert_success
}

@test "metric.sample" {
  esx_env

  host=$(govc ls -t HostSystem ./... | head -n 1)
  metrics=($(govc metric.ls "$host"))

  run govc metric.sample "$host" enoent
  assert_failure

  run govc metric.sample "$host" "${metrics[@]}"
  assert_success

  run govc metric.sample -instance - "$host" "${metrics[@]}"
  assert_success

  run govc metric.sample -json "$host" "${metrics[@]}"
  assert_success

  vm=$(new_ttylinux_vm)

  run govc metric.ls "$vm"
  assert_output ""

  run govc vm.power -on "$vm"
  assert_success

  run govc vm.ip "$vm"
  assert_success

  metrics=($(govc metric.ls "$vm"))

  run govc metric.sample "$vm" "${metrics[@]}"
  assert_success

  run govc metric.sample -json "$vm" "${metrics[@]}"
  assert_success

  run govc metric.sample "govc-test-*" "${metrics[@]}"
  assert_success
}

@test "metric.info" {
  esx_env

  host=$(govc ls -t HostSystem ./... | head -n 1)
  metrics=($(govc metric.ls "$host"))

  run govc metric.info "$host" enoent
  assert_failure

  run govc metric.info "$host"
  assert_success

  run govc metric.info -json "$host"
  assert_success

  run govc metric.info -dump "$host"
  assert_success

  run govc metric.sample "$host" "${metrics[@]}"
  assert_success

  run govc metric.info "$host" "${metrics[@]}"
  assert_success

  run govc metric.info - "${metrics[@]}"
  assert_success
}

@test "metric manager" {
  vcsim_env

  moid=$(govc object.collect -s - content.perfManager)

  govc object.collect -json "$moid" | jq .
}
