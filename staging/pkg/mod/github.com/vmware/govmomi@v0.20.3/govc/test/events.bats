#!/usr/bin/env bats

load test_helper

@test "events dc" {
  esx_env

  run govc events
  assert_success
  nevents=${#lines[@]}
  # there should be plenty more than 1 event at the top (dc) level
  [ $nevents -ge 1 ]

  # test -n flag
  run govc events -n $((nevents - 1))
  assert_success
  [ ${#lines[@]} -le $nevents ]
}

@test "events host" {
  esx_env

  run govc events 'host/*'
  assert_success
  [ ${#lines[@]} -ge 1 ]
}

@test "events vm" {
  esx_env

  vm=$(new_id)

  run govc vm.create -on=false $vm
  assert_success

  run govc events vm/$vm
  assert_success
  nevents=${#lines[@]}
  [ $nevents -gt 1 ]

  # glob should have same # of events
  run govc events vm/${vm}*
  assert_success
  [ ${#lines[@]} -eq $nevents ]

  # create a new vm, glob should match more events
  run govc vm.create -on=false "${vm}-2"
  assert_success
  run govc events vm/${vm}*
  assert_success
  [ ${#lines[@]} -gt $nevents ]
  nevents=${#lines[@]}

  run govc events vm
  assert_success
  [ ${#lines[@]} -ge $nevents ]

  run govc events -type VmPoweredOffEvent -type VmPoweredOnEvent "vm/$vm"
  [ ${#lines[@]} -eq 0 ]

  run govc vm.power -on "$vm"
  assert_success

  run govc events -type VmPoweredOffEvent -type VmPoweredOnEvent "vm/$vm"
  [ ${#lines[@]} -eq 1 ]

  run govc vm.power -off "$vm"
  assert_success

  run govc events -type VmPoweredOffEvent -type VmPoweredOnEvent "vm/$vm"
  [ ${#lines[@]} -eq 2 ]
}

@test "events json" {
  esx_env

  # make sure we fmt.Printf properly
  govc events | grep -v '%!s(MISSING)'

  govc events -json | jq .

  # test multiple objects
  govc vm.create "$(new_id)"
  govc vm.create "$(new_id)"

  govc events 'vm/*'
  govc events -json 'vm/*' | jq .
}
