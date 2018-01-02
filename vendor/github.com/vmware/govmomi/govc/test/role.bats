#!/usr/bin/env bats

load test_helper

@test "permissions.ls" {
  run govc permissions.ls
  assert_success

  run govc permissions.ls -json
  assert_success
}

@test "role.ls" {
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
  id=$(new_id)
  run govc role.create "$id"
  assert_success

  run govc role.ls "$id"
  assert_success

  priv=$(govc role.ls "$id" | wc -l)
  vm_priv=($(govc role.ls Admin | grep VirtualMachine.))

  # Test set
  run govc role.update "$id" "${vm_priv[@]}"
  assert_success

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
