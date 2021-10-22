#!/usr/bin/env bats

load test_helper

@test "permissions" {
  vcsim_env

  perm=$(govc permissions.ls /DC0)

  run govc permissions.ls -json
  assert_success

  run govc permissions.set -principal root -role Admin /DC0
  assert_success

  run govc permissions.ls /DC0
  refute_line "$perm"

  run govc permissions.remove -principal root /DC0
  assert_success

  run govc permissions.ls /DC0
  assert_success "$perm"
}

@test "role.ls" {
  vcsim_env

  run govc role.ls
  assert_success

  run govc role.ls -json
  assert_success

  run govc role.ls Admin
  assert_success

  run govc role.ls -json Admin
  assert_success

  run govc role.ls enoent
  assert_failure
}

@test "role.usage" {
  vcsim_env

  run govc role.usage
  assert_success

  run govc role.usage -json
  assert_success

  run govc role.usage Admin
  assert_success

  run govc role.usage -json Admin
  assert_success

  run govc role.usage enoent
  assert_failure
}

@test "role.create" {
  vcsim_env

  id=$(new_id)
  run govc role.create "$id"
  assert_success

  run govc role.ls "$id"
  assert_success

  priv=$(govc role.ls "$id" | wc -l)
  [ "$priv" -eq 3 ]

  vm_priv=($(govc role.ls Admin | grep VirtualMachine.))

  # Test set
  run govc role.update "$id" "${vm_priv[@]}"
  assert_success

  # invalid priv id
  run govc role.update "$id" enoent
  assert_failure

  npriv=$(govc role.ls "$id" | wc -l)
  [ "$npriv" -gt "$priv" ]
  priv=$npriv

  op_priv=($(govc role.ls "$id" | grep VirtualMachine.GuestOperations.))
  # Test remove
  run govc role.update -r "$id" "${op_priv[@]}"
  assert_success

  npriv=$(govc role.ls "$id" | wc -l)
  [ "$npriv" -lt "$priv" ]
  priv=$npriv

  # Test add
  run govc role.update -a "$id" "${op_priv[@]}"
  assert_success

  npriv=$(govc role.ls "$id" | wc -l)
  [ "$npriv" -gt "$priv" ]
  priv=$npriv

  # Test rename
  run govc role.update -name "${id}-N" "$id"
  assert_success

  id="${id}-N"
  # Test we didn't drop any privileges during rename
  [ "$priv" -eq "$(govc role.ls "$id" | wc -l)" ]

  run govc role.remove "${id}"
  assert_success

  run govc role.ls "$id"
  assert_failure
}
