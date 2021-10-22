#!/usr/bin/env bats

load test_helper

test_vm_snapshot() {
  vm=$1
  id=$(new_id)

  # No snapshots == no output
  run govc snapshot.tree -vm "$vm"
  assert_success ""

  run govc snapshot.remove -vm "$vm" '*'
  assert_success

  run govc snapshot.revert -vm "$vm"
  assert_failure

  run govc snapshot.create -vm "$vm" "$id"
  assert_success

  run govc snapshot.revert -vm "$vm" enoent
  assert_failure

  run govc snapshot.revert -vm "$vm"
  assert_success

  run govc snapshot.remove -vm "$vm" "$id"
  assert_success

  run govc snapshot.create -vm "$vm" root
  assert_success

  run govc snapshot.tree -C -vm "$vm"
  assert_success "root"

  run govc snapshot.create -vm "$vm" child
  assert_success

  run govc snapshot.tree -C -vm "$vm"
  assert_success "child"

  run govc snapshot.create -vm "$vm" grand
  assert_success

  run govc snapshot.create -vm "$vm" child
  assert_success

  result=$(govc snapshot.tree -vm "$vm" -f | grep -c root/child/grand/child)
  [ "$result" -eq 1 ]

  run govc snapshot.revert -vm "$vm" root
  assert_success

  run govc snapshot.create -vm "$vm" child
  assert_success

  # 3 snapshots named "child"
  result=$(govc snapshot.tree -vm "$vm" | grep -c child)
  [ "$result" -eq 3 ]

  run govc snapshot.remove -vm "$vm" child
  assert_failure

  # 2 snapshots with path "root/child"
  result=$(govc snapshot.tree -vm "$vm" -f | egrep -c 'root/child$')
  [ "$result" -eq 2 ]

  run govc snapshot.remove -vm "$vm" root/child
  assert_failure

  # path is unique
  run govc snapshot.remove -vm "$vm" root/child/grand/child
  assert_success

  # set current to grand
  run govc snapshot.revert -vm "$vm" grand

  # name is unique
  run govc snapshot.remove -vm "$vm" grand
  assert_success

  result=$(govc snapshot.tree -vm "$vm" -f | grep root/child/grand/child | wc -l)
  [ "$result" -eq 0 ]

  # current given to parent of previous current
  result=$(govc snapshot.tree -vm "$vm" -f | grep '\.' | wc -l)
  [ "$result" -eq 1 ]

  id=$(govc snapshot.tree -vm "$vm" -f -i | egrep 'root/child$' | head -n1 | awk '{print $1}' | tr -d '[]')
  # moid is unique
  run govc snapshot.remove -vm "$vm" "$id"
  assert_success

  # now root/child is unique
  run govc snapshot.remove -vm "$vm" root/child
  assert_success

  run govc snapshot.remove -vm "$vm" root
  assert_success

  # current is removed
  result=$(govc snapshot.tree -vm "$vm" -f | grep '\.' | wc -l)
  [ "$result" -eq 0 ]

  # new root
  run govc snapshot.create -vm "$vm" 2ndroot
  assert_success

  # new snapshot 2ndroot is current
  result=$(govc snapshot.tree -vm "$vm" -f | grep '\.' | wc -l)
  [ "$result" -eq 1 ]
}

@test "vm.snapshot vcsim" {
  vcsim_env

  vm=$(new_empty_vm)

  test_vm_snapshot $vm
}

@test "vm.snapshot" {
  esx_env

  vm=$(new_ttylinux_vm)

  test_vm_snapshot $vm
}
