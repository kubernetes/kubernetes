#!/usr/bin/env bats

load test_helper

@test "create and destroy datacenters" {
  vcsim_env
  dcs=(`uuidgen` `uuidgen`)
  run govc datacenter.create ${dcs[0]} ${dcs[1]}
  assert_success

  for dc in ${dcs[*]}; do
    run govc ls /$dc
    assert_success
    # /<datacenter>/{vm,network,host,datastore}
    [ ${#lines[@]} -eq 4 ]
  done

  run govc datacenter.destroy ${dcs[0]} ${dcs[1]}
  assert_success

  for dc in ${dcs[*]}; do
    run govc ls /$dc
    assert_success
    [ ${#lines[@]} -eq 0 ]
  done
}

@test "destroy datacenter using glob" {
  vcsim_env
  prefix=test-dc
  dcs=(${prefix}-`uuidgen` ${prefix}-`uuidgen`)
  run govc datacenter.create ${dcs[0]} ${dcs[1]}
  assert_success

  run govc datacenter.destroy ${prefix}-*
  assert_success

  for dc in ${dcs[*]}; do
    run govc ls /$dc
    assert_success
    [ ${#lines[@]} -eq 0 ]
  done
}

@test "destroy datacenter that doesn't exist" {
  vcsim_env
  dc=$(uuidgen)

  run govc datacenter.destroy $dc
  assert_success
}

@test "create datacenter that already exists" {
  vcsim_env
  dc=$(uuidgen)

  run govc datacenter.create $dc
  assert_success

  run govc datacenter.create $dc
  assert_success

  run govc datacenter.destroy $dc
  assert_success
}

@test "fails when datacenter name not specified" {
  run govc datacenter.create
  assert_failure

  run govc datacenter.destroy
  assert_failure
}

@test "fails when operation attempted on standalone ESX host" {
  run govc datacenter.create something
  assert_failure
  assert_output "govc: ServerFaultCode: The operation is not supported on the object."
}

@test "fails when attempting to destroy ha-datacenter" {
  run govc datacenter.destroy ha-datacenter
  assert_failure
  assert_output "govc: The operation is not supported on the object."
}
