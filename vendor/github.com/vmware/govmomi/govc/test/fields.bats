#!/usr/bin/env bats

load test_helper

@test "fields" {
  vcsim_env

  vm_id=$(new_id)
  run govc vm.create $vm_id
  assert_success

  field=$(new_id)

  result=$(govc fields.ls | grep $field | wc -l)
  [ $result -eq 0 ]

  key=$(govc fields.add $field)

  result=$(govc fields.ls | grep $field | wc -l)
  [ $result -eq 1 ]

  key=$(govc fields.ls | grep $field | awk '{print $1}')

  val="foo"
  run govc fields.set $field $val vm/$vm_id
  assert_success

  info=$(govc vm.info -json $vm_id | jq .VirtualMachines[0].CustomValue[0])

  ikey=$(jq -r .Key <<<"$info")
  assert_equal $key $ikey

  ival=$(jq -r .Value <<<"$info")
  assert_equal $val $ival

  old_field=$field
  field=$(new_id)
  run govc fields.rename $key $field
  assert_success
  result=$(govc fields.ls | grep $old_field | wc -l)
  [ $result -eq 0 ]

  run govc fields.rm $field
  assert_success

  result=$(govc fields.ls | grep $field | wc -l)
  [ $result -eq 0 ]
}
