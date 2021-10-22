#!/usr/bin/env bats

load test_helper

@test "network vm list" {
  esx_env

  # make sure there's at least 1 VM so we get a table header to count against
  vm=$(new_empty_vm)
  govc vm.power -on $vm

  nlines=$(govc host.esxcli network vm list | wc -l)

  vm=$(new_empty_vm)
  govc vm.power -on $vm

  xlines=$(govc host.esxcli network vm list | wc -l)

  # test that we see a new row
  [ $(($nlines + 1)) -eq $xlines ]

  run govc host.esxcli network vm list enoent
  assert_failure
}

@test "network ip connection list" {
  esx_env

  run govc host.esxcli -- network ip connection list -t tcp
  assert_success

  # test that we get the expected number of table columns
  nf=$(echo "${lines[3]}" | awk '{print NF}')
  [ $nf -eq 9 ]

  run govc host.esxcli -- network ip connection list -t enoent
  assert_failure
}

@test "system settings advanced list" {
  esx_env

  run govc host.esxcli -- system settings advanced list -o /Net/GuestIPHack
  assert_success
  assert_line "Path: /Net/GuestIPHack"

  run govc host.esxcli -- system settings advanced list -o /Net/ENOENT
  assert_failure
}
