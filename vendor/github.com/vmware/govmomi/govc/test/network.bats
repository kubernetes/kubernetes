#!/usr/bin/env bats

load test_helper

@test "network dvs backing" {
  vcsim_env

  # DVS backed network by default (from vcsim_env)
  vm=$(new_empty_vm)

  eth0=$(govc device.ls -vm $vm | grep ethernet- | awk '{print $1}')
  run govc device.info -vm $vm $eth0
  assert_success

  summary=$(govc device.info -vm $vm $eth0 | grep Summary: | awk '{print $2}')
  assert_equal "DVSwitch:" $summary

  run govc device.remove -vm $vm $eth0
  assert_success

  eth0=$(govc device.ls -vm $vm | grep ethernet- | awk '{print $1}')
  [ -z "$eth0" ]

  # Standard network backing
  run govc vm.network.add -vm $vm -net "VM Network"
  assert_success

  eth0=$(govc device.ls -vm $vm | grep ethernet- | awk '{print $1}')

  run govc device.info -vm $vm $eth0
  assert_success

  summary=$(govc device.info -vm $vm $eth0 | grep Summary: | awk -F: '{print $2}')
  assert_equal "VM Network" $(collapse_ws $summary)

  run govc device.remove -vm $vm $eth0
  assert_success

  run govc device.remove -vm $vm $eth0
  assert_failure "govc: device '$eth0' not found"
}

@test "network change backing" {
  vcsim_env

  vm=$(new_empty_vm)

  eth0=$(govc device.ls -vm $vm | grep ethernet- | awk '{print $1}')
  run govc vm.network.change -vm $vm $eth0 enoent
  assert_failure "govc: network 'enoent' not found"

  run govc vm.network.change -vm $vm enoent "VM Network"
  assert_failure "govc: device 'enoent' not found"

  run govc vm.network.change -vm $vm $eth0 "VM Network"
  assert_success

  run govc vm.network.change -vm $vm $eth0
  assert_success

  unset GOVC_NETWORK
  run govc vm.network.change -vm $vm $eth0
  assert_failure "govc: default network resolves to multiple instances, please specify"

  run govc vm.power -on $vm
  assert_success
  run govc vm.power -off $vm

  mac=$(vm_mac $vm)
  run govc vm.network.change -vm $vm -net "VM Network" $eth0
  assert_success

  # verify we didn't change the mac address
  run govc vm.power -on $vm
  assert_success
  assert_equal $mac $(vm_mac $vm)
}

@test "network standard backing" {
  vm=$(new_empty_vm)

  run govc device.info -vm $vm ethernet-0
  assert_success

  run govc device.remove -vm $vm ethernet-0
  assert_success

  run govc device.info -vm $vm ethernet-0
  assert_failure

  run govc vm.network.add -vm $vm enoent
  assert_failure "govc: network 'enoent' not found"

  run govc vm.network.add -vm $vm "VM Network"
  assert_success

  run govc device.info -vm $vm ethernet-0
  assert_success
}

@test "network adapter" {
  vm=$(new_id)
  run govc vm.create -on=false -net.adapter=enoent $vm
  assert_failure "govc: unknown ethernet card type 'enoent'"

  vm=$(new_id)
  run govc vm.create -on=false -net.adapter=vmxnet3 $vm
  assert_success

  eth0=$(govc device.ls -vm $vm | grep ethernet- | awk '{print $1}')
  type=$(govc device.info -vm $vm $eth0 | grep Type: | awk -F: '{print $2}')
  assert_equal "VirtualVmxnet3" $(collapse_ws $type)

  run govc vm.network.add -vm $vm -net.adapter e1000e "VM Network"
  assert_success

  eth1=$(govc device.ls -vm $vm | grep ethernet- | grep -v $eth0 | awk '{print $1}')
  type=$(govc device.info -vm $vm $eth1 | grep Type: | awk -F: '{print $2}')
  assert_equal "VirtualE1000e" $(collapse_ws $type)
}

@test "network flag required" {
  vcsim_env

  # -net flag is required when there are multiple networks
  unset GOVC_NETWORK
  run govc vm.create -on=false $(new_id)
  assert_failure "govc: default network resolves to multiple instances, please specify"
}

@test "network change hardware address" {
  mac="00:00:0f$(dd bs=1 count=3 if=/dev/random 2>/dev/null | hexdump -v -e '/1 ":%02x"')"
  vm=$(new_id)
  run govc vm.create -on=false $vm
  assert_success

  run govc vm.network.change -vm $vm -net.address $mac ethernet-0
  assert_success

  run govc vm.power -on $vm
  assert_success

  assert_equal $mac $(vm_mac $vm)
}

@test "dvs.portgroup" {
  vcsim_env
  id=$(new_id)

  run govc dvs.create "$id"
  assert_success

  run govc dvs.portgroup.add -dvs "$id" -type earlyBinding -nports 16 "${id}-ExternalNetwork"
  assert_success

  run govc dvs.portgroup.add -dvs "$id" -type ephemeral -vlan 3122 "${id}-InternalNetwork"
  assert_success

  info=$(govc dvs.portgroup.info "$id" | grep VlanId: | uniq | grep 3122)
  [ -n "$info" ]

  run govc dvs.portgroup.change -vlan 3123 "${id}-InternalNetwork"
  assert_success

  info=$(govc dvs.portgroup.info "$id" | grep VlanId: | uniq | grep 3123)
  [ -n "$info" ]

  run govc object.destroy "network/${id}-ExternalNetwork" "network/${id}-InternalNetwork" "network/${id}"
  assert_success
}
