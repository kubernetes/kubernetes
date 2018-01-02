#!/usr/bin/env bats

load test_helper

@test "object.destroy" {
    run govc object.destroy "/enoent"
    assert_failure

    run govc object.destroy
    assert_failure

    vm=$(new_id)
    run govc vm.create "$vm"
    assert_success

    # fails when powered on
    run govc object.destroy "vm/$vm"
    assert_failure

    run govc vm.power -off "$vm"
    assert_success

    run govc object.destroy "vm/$vm"
    assert_success
}

@test "object.rename" {
  run govc object.rename "/enoent" "nope"
  assert_failure

  vm=$(new_id)
  run govc vm.create -on=false "$vm"
  assert_success

  run govc object.rename "vm/$vm" "${vm}-renamed"
  assert_success

  run govc object.rename "vm/$vm" "${vm}-renamed"
  assert_failure

  run govc object.destroy "vm/${vm}-renamed"
  assert_success
}

@test "object.mv" {
  vcsim_env

  folder=$(new_id)

  run govc folder.create "vm/$folder"
  assert_success

  for _ in $(seq 1 3) ; do
    vm=$(new_id)
    run govc vm.create -folder "$folder" "$vm"
    assert_success
  done

  result=$(govc ls "vm/$folder" | wc -l)
  [ "$result" -eq "3" ]

  run govc folder.create "vm/${folder}-2"
  assert_success

  run govc object.mv "vm/$folder/*" "vm/${folder}-2"
  assert_success

  result=$(govc ls "vm/${folder}-2" | wc -l)
  [ "$result" -eq "3" ]

  result=$(govc ls "vm/$folder" | wc -l)
  [ "$result" -eq "0" ]
}

@test "object.collect" {
  run govc object.collect
  assert_success

  run govc object.collect -json
  assert_success

  run govc object.collect -
  assert_success

  run govc object.collect -json -
  assert_success

  run govc object.collect - content
  assert_success

  run govc object.collect -json - content
  assert_success

  root=$(govc object.collect - content | grep content.rootFolder | awk '{print $3}')

  dc=$(govc object.collect "$root" childEntity | awk '{print $3}' | cut -d, -f1)

  hostFolder=$(govc object.collect "$dc" hostFolder | awk '{print $3}')

  cr=$(govc object.collect "$hostFolder" childEntity | awk '{print $3}' | cut -d, -f1)

  host=$(govc object.collect "$cr" host | awk '{print $3}' | cut -d, -f1)

  run govc object.collect "$host"
  assert_success

  run govc object.collect "$host" hardware
  assert_success

  run govc object.collect "$host" hardware.systemInfo
  assert_success

  uuid=$(govc object.collect "$host" hardware.systemInfo.uuid | awk '{print $3}')
  uuid_s=$(govc object.collect -s "$host" hardware.systemInfo.uuid)
  assert_equal "$uuid" "$uuid_s"

  run govc object.collect "$(govc ls host | head -n1)"
  assert_success

  # test against slice of interface
  perfman=$(govc object.collect -s - content.perfManager)
  result=$(govc object.collect -s "$perfman" description.counterType)
  assert_equal "..." "$result"

  # test against an interface field
  run govc object.collect '/ha-datacenter/network/VM Network' summary
  assert_success
}

@test "object.find" {
  unset GOVC_DATACENTER

  run govc find "/enoent"
  assert_failure

  run govc find
  assert_success

  run govc find .
  assert_success

  run govc find /
  assert_success

  run govc find . -type HostSystem
  assert_success

  dc=$(govc find / -type Datacenter | head -1)

  run govc find "$dc" -maxdepth 0
  assert_output "$dc"

  run govc find "$dc/vm" -maxdepth 0
  assert_output "$dc/vm"

  run govc find "$dc" -maxdepth 1 -type Folder
  assert_success
  # /<datacenter>/{vm,network,host,datastore}
  [ ${#lines[@]} -eq 4 ]

  folder=$(govc find -type Folder -name vm)

  vm=$(new_empty_vm)

  run govc find . -name "$vm"
  assert_output "$folder/$vm"

  run govc find "$folder" -name "$vm"
  assert_output "$folder/$vm"

  # moref for VM Network
  net=$(govc find -i network -name "$GOVC_NETWORK")

  # $vm.network.contains($net) == true
  run govc find . -type m -name "$vm" -network "$net"
  assert_output "$folder/$vm"

  # remove network reference
  run govc device.remove -vm "$vm" ethernet-0
  assert_success

  # $vm.network.contains($net) == false
  run govc find . -type VirtualMachine -name "$vm" -network "$net"
  assert_output ""

  run govc find "$folder" -type VirtualMachine -name "govc-test-*" -runtime.powerState poweredOn
  assert_output ""

  run govc find "$folder" -type VirtualMachine -name "govc-test-*" -runtime.powerState poweredOff
  assert_output "$folder/$vm"

  run govc vm.power -on "$vm"
  assert_success

  run govc find "$folder" -type VirtualMachine -name "govc-test-*" -runtime.powerState poweredOff
  assert_output ""

  run govc find "$folder" -type VirtualMachine -name "govc-test-*" -runtime.powerState poweredOn
  assert_output "$folder/$vm"

  # output paths should be relative to "." in these cases
  export GOVC_DATACENTER=$dc

  folder="./vm"

  run govc find . -name "$vm"
  assert_output "$folder/$vm"

  run govc find "$folder" -name "$vm"
}

@test "object.method" {
  vcsim_env

  vm=$(govc find vm -type m | head -1)

  run govc object.method -enable=false -name NoSuchMethod "$vm"
  assert_failure

  run govc object.method -enable=false -name Destroy_Task enoent
  assert_failure

  run govc object.collect -s "$vm" disabledMethod
  ! assert_matches "Destroy_Task" "$output"

  run govc object.method -enable=false -name Destroy_Task "$vm"
  assert_success

  run govc object.collect -s "$vm" disabledMethod
  assert_matches "Destroy_Task" "$output"

  run govc object.method -enable -name Destroy_Task "$vm"
  assert_success

  run govc object.collect -s "$vm" disabledMethod
  ! assert_matches "Destroy_Task" "$output"
}
