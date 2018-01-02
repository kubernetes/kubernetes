#!/usr/bin/env bats

load test_helper

@test "events dc" {
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
  run govc events 'host/*'
  assert_success
  [ ${#lines[@]} -ge 1 ]
}

@test "events vm" {
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
}
