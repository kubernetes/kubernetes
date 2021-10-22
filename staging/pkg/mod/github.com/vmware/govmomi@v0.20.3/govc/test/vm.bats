#!/usr/bin/env bats

load test_helper

@test "vm.ip" {
  esx_env

  id=$(new_ttylinux_vm)

  run govc vm.power -on $id
  assert_success

  run govc vm.ip $id
  assert_success

  run govc vm.ip -a -v4 $id
  assert_success

  run govc vm.ip -n $(vm_mac $id) $id
  assert_success

  run govc vm.ip -n ethernet-0 $id
  assert_success

  ip=$(govc vm.ip $id)

  # add a second nic
  run govc vm.network.add -vm $id "VM Network"
  assert_success

  res=$(govc vm.ip -n ethernet-0 $id)
  assert_equal $ip $res
}

@test "vm.ip -esxcli" {
  esx_env

  ok=$(govc host.esxcli system settings advanced list -o /Net/GuestIPHack | grep ^IntValue: | awk '{print $2}')
  if [ "$ok" != "1" ] ; then
    skip "/Net/GuestIPHack=0"
  fi
  id=$(new_ttylinux_vm)

  run govc vm.power -on $id
  assert_success

  run govc vm.ip -esxcli $id
  assert_success

  ip_esxcli=$output

  run govc vm.ip $id
  assert_success
  ip_tools=$output

  assert_equal $ip_esxcli $ip_tools
}

@test "vm.create" {
  esx_env

  id=$(new_ttylinux_vm)

  run govc vm.power -on $id
  assert_success

  result=$(govc device.ls -vm $vm | grep disk- | wc -l)
  [ $result -eq 0 ]

  result=$(govc device.ls -vm $vm | grep cdrom- | wc -l)
  [ $result -eq 0 ]
}

@test "vm.change" {
  esx_env

  id=$(new_ttylinux_vm)

  run govc vm.change -g ubuntu64Guest -m 1024 -c 2 -vm $id
  assert_success

  run govc vm.info $id
  assert_success
  assert_line "Guest name: Ubuntu Linux (64-bit)"
  assert_line "Memory: 1024MB"
  assert_line "CPU: 2 vCPU(s)"

  run govc vm.change -e "guestinfo.a=1" -e "guestinfo.b=2" -vm $id
  assert_success

  run govc vm.info -e $id
  assert_success
  assert_line "guestinfo.a: 1"
  assert_line "guestinfo.b: 2"

  run govc vm.change -sync-time-with-host=false -vm $id
  assert_success

  run govc vm.info -t $id
  assert_success
  assert_line "SyncTimeWithHost: false"

  run govc vm.change -sync-time-with-host=true -vm $id
  assert_success

  run govc vm.info -t $id
  assert_success
  assert_line "SyncTimeWithHost: true"

  run govc object.collect -s "vm/$id" config.memoryAllocation.reservation
  assert_success 0

  govc vm.change -vm "$id" -mem.reservation 1024

  run govc object.collect -s "vm/$id" config.memoryAllocation.reservation
  assert_success 1024

  run govc vm.change -annotation $$ -vm "$id"
  assert_success

  run govc object.collect -s "vm/$id" config.annotation
  assert_success $$

  nid=$(new_id)
  run govc vm.change -name $nid -vm $id
  assert_success

  run govc vm.info $id
  [ ${#lines[@]} -eq 0 ]

  run govc vm.info $nid
  [ ${#lines[@]} -gt 0 ]
}

@test "vm.power" {
  esx_env

  vm=$(new_ttylinux_vm)

  run vm_power_state $vm
  assert_success "poweredOff"

  run govc vm.power $vm
  assert_failure

  run govc vm.power -on -off $vm
  assert_failure

  run govc vm.power -on $vm
  assert_success
  run vm_power_state $vm
  assert_success "poweredOn"

  run govc vm.power -suspend $vm
  assert_success
  run vm_power_state $vm
  assert_success "suspended"

  run govc vm.power -on $vm
  assert_success
  run vm_power_state $vm
  assert_success "poweredOn"
}

@test "vm.power -on -M" {
  for esx in true false ; do
    vcsim_env -esx=$esx -autostart=false

    vms=($(govc find / -type m | sort))

    # All VMs are off with -autostart=false
    off=($(govc find / -type m -runtime.powerState poweredOff | sort))
    assert_equal "${vms[*]}" "${off[*]}"

    # Power on 1 VM to test that -M is idempotent
    run govc vm.power -on "${vms[0]}"
    assert_success

    run govc vm.power -on -M "${vms[@]}"
    assert_success

    # All VMs should be powered on now
    on=($(govc find / -type m -runtime.powerState poweredOn | sort))
    assert_equal "${vms[*]}" "${on[*]}"

    vcsim_stop
  done
}

@test "vm.power -force" {
  esx_env

  vm=$(new_id)
  govc vm.create $vm

  run govc vm.power -r $vm
  assert_failure

  run govc vm.power -r -force $vm
  assert_success

  run govc vm.power -s $vm
  assert_failure

  run govc vm.power -s -force $vm
  assert_success

  run govc vm.power -off $vm
  assert_failure

  run govc vm.power -off -force $vm
  assert_success

  run govc vm.destroy $vm
  assert_success

  run govc vm.power -off $vm
  assert_failure

  run govc vm.power -off -force $vm
  assert_failure
}

@test "vm.create pvscsi" {
  esx_env

  vm=$(new_id)
  govc vm.create -on=false -disk.controller pvscsi $vm

  result=$(govc device.ls -vm $vm | grep pvscsi- | wc -l)
  [ $result -eq 1 ]

  result=$(govc device.ls -vm $vm | grep lsilogic- | wc -l)
  [ $result -eq 0 ]

  vm=$(new_id)
  govc vm.create -on=false -disk.controller pvscsi -disk=1GB $vm
}

@test "vm.create in cluster" {
  vcsim_env

  # using GOVC_HOST and its resource pool
  run govc vm.create -on=false $(new_id)
  assert_success

  # using no -host and the default resource pool for DC0
  unset GOVC_HOST
  run govc vm.create -on=false $(new_id)
  assert_success
}

@test "vm.create -datastore-cluster" {
  vcsim_env -pod 1 -ds 3

  pod=/DC0/datastore/DC0_POD0
  id=$(new_id)

  run govc vm.create -disk 10M -datastore-cluster $pod "$id"
  assert_failure

  run govc object.mv /DC0/datastore/LocalDS_{1,2} $pod
  assert_success

  run govc vm.create -disk 10M -datastore-cluster $pod "$id"
  assert_success
}

@test "vm.info" {
  vcsim_env -esx

  local num=3

  local prefix=$(new_id)

  for x in $(seq $num)
  do
    local id="${prefix}-${x}"

    # If VM is not found: No output, exit code==0
    run govc vm.info $id
    assert_success
    [ ${#lines[@]} -eq 0 ]

    # If VM is not found (using -json flag): Valid json output, exit code==0
    run govc vm.info -json $id
    assert_success
    assert_line "{\"VirtualMachines\":null}"

    run govc vm.info -dump $id
    assert_success

    run govc vm.create -on=false $id
    assert_success

    local info=$(govc vm.info -r $id)
    local found=$(grep Name: <<<"$info" | wc -l)
    [ "$found" -eq 1 ]

    # test that mo names are printed
    found=$(grep Host: <<<"$info" | awk '{print $2}')
    [ -n "$found" ]
    found=$(grep Storage: <<<"$info" | awk '{print $2}')
    [ -n "$found" ]
    found=$(grep Network: <<<"$info" | awk '{print $2}')
    [ -n "$found" ]
  done

  # test find slice
  local slice=$(govc vm.info ${prefix}-*)
  local found=$(grep Name: <<<"$slice" | wc -l)
  [ "$found" -eq $num ]

  # test -r
  found=$(grep Storage: <<<"$slice" | wc -l)
  [ "$found" -eq 0 ]
  found=$(grep Network: <<<"$slice" | wc -l)
  [ "$found" -eq 0 ]
  slice=$(govc vm.info -r ${prefix}-*)
  found=$(grep Storage: <<<"$slice" | wc -l)
  [ "$found" -eq $num ]
  found=$(grep Network: <<<"$slice" | wc -l)
  [ "$found" -eq $num ]

  # test extraConfig
  run govc vm.change -e "guestinfo.a=2" -vm $id
  assert_success
  run govc vm.info -e $id
  assert_success
  assert_line "guestinfo.a: 2"
  run govc vm.change -e "guestinfo.a=" -vm $id
  assert_success
  refute_line "guestinfo.a: 2"

  # test optional bool Config
  run govc vm.change -nested-hv-enabled=true -vm "$id"
  assert_success

  hv=$(govc vm.info -json "$id" | jq '.[][0].Config.NestedHVEnabled')
  assert_equal "$hv" "true"
}

@test "vm.create linked ide disk" {
  esx_env

  import_ttylinux_vmdk

  vm=$(new_id)

  run govc vm.create -disk $GOVC_TEST_VMDK -disk.controller ide -on=false $vm
  assert_success

  run govc device.info -vm $vm disk-200-0
  assert_success
  assert_line "Controller: ide-200"
}

@test "vm.create linked scsi disk" {
  esx_env

  import_ttylinux_vmdk

  vm=$(new_id)

  run govc vm.create -disk enoent -on=false $vm
  assert_failure "govc: cannot stat '[${GOVC_DATASTORE##*/}] enoent': No such file"

  run govc vm.create -disk $GOVC_TEST_VMDK -on=false $vm
  assert_success

  run govc device.info -vm $vm disk-1000-0
  assert_success
  assert_line "Controller: lsilogic-1000"
  assert_line "Parent: [${GOVC_DATASTORE##*/}] $GOVC_TEST_VMDK"
  assert_line "File: [${GOVC_DATASTORE##*/}] $vm/${vm}.vmdk"
}

@test "vm.create scsi disk" {
  esx_env

  import_ttylinux_vmdk

  vm=$(new_id)

  run govc vm.create -disk enoent -on=false $vm
  assert_failure "govc: cannot stat '[${GOVC_DATASTORE##*/}] enoent': No such file"


  run govc vm.create -disk $GOVC_TEST_VMDK -on=false -link=false $vm
  assert_success

  run govc device.info -vm $vm disk-1000-0
  assert_success
  assert_line "Controller: lsilogic-1000"
  refute_line "Parent: [${GOVC_DATASTORE##*/}] $GOVC_TEST_VMDK"
  assert_line "File: [${GOVC_DATASTORE##*/}] $GOVC_TEST_VMDK"
}

@test "vm.create scsi disk with datastore argument" {
  esx_env

  import_ttylinux_vmdk

  vm=$(new_id)

  run govc vm.create -disk="${GOVC_TEST_VMDK}" -disk-datastore="${GOVC_DATASTORE}" -on=false -link=false $vm
  assert_success

  run govc device.info -vm $vm disk-1000-0
  assert_success
  assert_line "File: [${GOVC_DATASTORE##*/}] $GOVC_TEST_VMDK"
}

@test "vm.create iso" {
  esx_env

  upload_iso

  vm=$(new_id)

  run govc vm.create -iso enoent -on=false $vm
  assert_failure "govc: cannot stat '[${GOVC_DATASTORE##*/}] enoent': No such file"

  run govc vm.create -iso $GOVC_TEST_ISO -on=false $vm
  assert_success

  run govc device.info -vm $vm cdrom-3000
  assert_success
  assert_line "Controller: ide-200"
  assert_line "Summary: ISO [${GOVC_DATASTORE##*/}] $GOVC_TEST_ISO"
}

@test "vm.create iso with datastore argument" {
  esx_env

  upload_iso

  vm=$(new_id)

  run govc vm.create -iso="${GOVC_TEST_ISO}" -iso-datastore="${GOVC_DATASTORE}" -on=false $vm
  assert_success

  run govc device.info -vm $vm cdrom-3000
  assert_success
  assert_line "Summary: ISO [${GOVC_DATASTORE##*/}] $GOVC_TEST_ISO"
}

@test "vm.disk.create empty vm" {
  esx_env

  vm=$(new_empty_vm)

  local name=$(new_id)

  run govc vm.disk.create -vm "$vm" -name "$name" -size 1G
  assert_success
  result=$(govc device.ls -vm "$vm" | grep -c disk-)
  [ "$result" -eq 1 ]
  govc device.info -json -vm "$vm" disk-* | jq .Devices[].Backing.Sharing | grep -v sharingMultiWriter

  name=$(new_id)

  run govc vm.disk.create -vm "$vm" -name "$vm/$name" -controller lsilogic-1000 -size 2G
  assert_success

  result=$(govc device.ls -vm "$vm" | grep -c disk-)
  [ "$result" -eq 2 ]
}

@test "vm.disk.share" {
  esx_env

  vm=$(new_empty_vm)

  run govc vm.disk.create -vm "$vm" -name "$vm/shared.vmdk" -size 1G -eager -thick -sharing sharingMultiWriter
  assert_success
  govc device.info -json -vm "$vm" disk-* | jq .Devices[].Backing.Sharing | grep sharingMultiWriter

  run govc vm.power -on "$vm"
  assert_success

  vm2=$(new_empty_vm)

  run govc vm.disk.attach -vm "$vm2" -link=false -disk "$vm/shared.vmdk"
  assert_success

  run govc vm.power -on "$vm2"
  assert_failure # requires sharingMultiWriter

  run govc device.remove -vm "$vm2" -keep disk-1000-0
  assert_success

  run govc vm.disk.attach -vm "$vm2" -link=false -sharing sharingMultiWriter -disk "$vm/shared.vmdk"
  assert_success

  run govc vm.power -on "$vm2"
  assert_success

  run govc vm.power -off "$vm"
  assert_success

  run govc vm.disk.change -vm "$vm" -disk.filePath "[$GOVC_DATASTORE] $vm/shared.vmdk" -sharing sharingNone
  assert_success

  ! govc device.info -json -vm "$vm" disk-* | jq .Devices[].Backing.Sharing | grep sharingMultiWriter
}

@test "vm.disk.create" {
  esx_env

  import_ttylinux_vmdk

  vm=$(new_id)

  govc vm.create -disk $GOVC_TEST_VMDK -on=false $vm
  result=$(govc device.ls -vm $vm | grep disk- | wc -l)
  [ $result -eq 1 ]

  local name=$(new_id)

  run govc vm.disk.create -vm $vm -name $name -size 1G
  assert_success
  result=$(govc device.ls -vm $vm | grep disk- | wc -l)
  [ $result -eq 2 ]

  run govc vm.disk.create -vm $vm -name $name -size 1G
  assert_success # TODO: should fail?
  result=$(govc device.ls -vm $vm | grep disk- | wc -l)
  [ $result -eq 2 ]
}

@test "vm.disk.attach" {
  esx_env

  import_ttylinux_vmdk

  vm=$(new_id)

  govc vm.create -disk $GOVC_TEST_VMDK -on=false $vm
  result=$(govc device.ls -vm $vm | grep disk- | wc -l)
  [ $result -eq 1 ]

  id=$(new_id)
  run govc import.vmdk $GOVC_TEST_VMDK_SRC $id
  assert_success

  run govc vm.disk.attach -vm $vm -link=false -disk enoent.vmdk
  assert_failure "govc: File [${GOVC_DATASTORE##*/}] enoent.vmdk was not found"

  run govc vm.disk.attach -vm $vm -disk enoent.vmdk
  assert_failure "govc: Invalid configuration for device '0'."

  run govc vm.disk.attach -vm $vm -disk $id/$(basename $GOVC_TEST_VMDK) -controller lsilogic-1000
  assert_success
  result=$(govc device.ls -vm $vm | grep disk- | wc -l)
  [ $result -eq 2 ]
}

@test "vm.create new disk with datastore argument" {
  esx_env

  vm=$(new_id)

  run govc vm.create -disk="1GiB" -ds="${GOVC_DATASTORE}" -on=false -link=false $vm
  assert_success

  run govc device.info -vm $vm disk-1000-0
  assert_success
  assert_line "File: [${GOVC_DATASTORE##*/}] ${vm}/${vm}.vmdk"
}

@test "vm.create new disk with datastore cluster argument" {
  if [ -z "${GOVC_DATASTORE_CLUSTER}" ]; then
    skip "requires datastore cluster"
  fi

  vm=$(new_id)

  run govc vm.create -disk="1GiB" -datastore-cluster="${GOVC_DATASTORE_CLUSTER}" -on=false -link=false $vm
  assert_success

  run govc device.info -vm $vm disk-1000-0
  assert_success
}

@test "vm.register" {
  esx_env

  run govc vm.unregister enoent
  assert_failure

  vm=$(new_empty_vm)

  run govc vm.unregister "$vm"
  assert_success

  run govc vm.register "$vm/${vm}.vmx"
  assert_success
}

@test "vm.register vcsim" {
  vcsim_env -autostart=false

  host=$GOVC_HOST
  pool=$GOVC_RESOURCE_POOL

  unset GOVC_HOST GOVC_RESOURCE_POOL

  vm=DC0_H0_VM0

  run govc vm.unregister $vm
  assert_success

  run govc vm.register "$vm/${vm}.vmx"
  assert_failure # -pool is required

  run govc vm.register -pool "$pool" "$vm/${vm}.vmx"
  assert_success

  run govc vm.unregister $vm
  assert_success

  run govc vm.register -template -pool "$pool" "$vm/${vm}.vmx"
  assert_failure # -pool is not allowed w/ template

  run govc vm.register -template -host "$host" "$vm/${vm}.vmx"
  assert_success
}

@test "vm.clone" {
  vcsim_env

  vm=$(new_empty_vm)
  clone=$(new_id)

  run govc vm.clone -vm "$vm" -annotation $$ "$clone"
  assert_success

  run govc object.collect -s "/$GOVC_DATACENTER/vm/$clone" config.annotation
  assert_success $$

  clone=$(new_id)
  run govc vm.clone -vm "$vm" -snapshot X "$clone"
  assert_failure

  run govc snapshot.create -vm "$vm" X
  assert_success

  run govc vm.clone -vm "$vm" -snapshot X "$clone"
  assert_success
}

@test "vm.clone change resources" {
  vcsim_env

  vm=$(new_empty_vm)
  clone=$(new_id)

  run govc vm.info -r "$vm"
  assert_success
  assert_line "Network: $(basename "$GOVC_NETWORK")" # DVPG0

  run govc vm.clone -m 1024 -c 2 -net "VM Network" -vm "$vm" "$clone"
  assert_success

  run govc vm.info -r "$clone"
  assert_success
  assert_line "Memory: 1024MB"
  assert_line "CPU: 2 vCPU(s)"
  assert_line "Network: VM Network"

  # Remove all NICs from source vm
  run govc device.remove -vm "$vm" "$(govc device.ls -vm "$vm" | grep ethernet- | awk '{print $1}')"
  assert_success

  clone=$(new_id)

  mac=00:00:0f:a7:a0:f1
  run govc vm.clone -net "VM Network" -net.address $mac -vm "$vm" "$clone"
  assert_success

  run govc vm.info -r "$clone"
  assert_success
  assert_line "Network: VM Network"

  run govc device.info -vm "$clone"
  assert_success
  assert_line "MAC Address: $mac"
}

@test "vm.clone usage" {
  # validate we require -vm flag
  run govc vm.clone enoent
  assert_failure
}

@test "vm.migrate" {
  vcsim_env -cluster 2

  host0=/DC0/host/DC0_C0/DC0_C0_H0
  host1=/DC0/host/DC0_C0/DC0_C0_H1
  moid0=$(govc find -maxdepth 0 -i $host0)
  moid1=$(govc find -maxdepth 0 -i $host1)

  vm=$(new_id)
  run govc vm.create -on=false -host $host0 "$vm"
  assert_success

  # assert VM is on H0
  run govc object.collect "vm/$vm" -runtime.host "$moid0"
  assert_success

  # WaitForUpdates until the VM runtime.host changes to H1
  govc object.collect "vm/$vm" -runtime.host "$moid1" &
  pid=$!

  # migrate from H0 to H1
  run govc vm.migrate -host $host1 "$vm"
  assert_success

  wait $pid

  # (re-)assert VM is now on H1
  run govc object.collect "vm/$vm" -runtime.host "$moid1"
  assert_success

  # migrate from C0 to C1
  run govc vm.migrate -pool DC0_C1/Resources "$vm"
  assert_success
}

@test "object name with slash" {
  esx_env

  vm=$(new_empty_vm)

  name="$vm/with-slash"

  # rename VM to include a '/'
  run govc vm.change -vm "$vm" -name "$name"
  assert_success

  path=$(govc ls "vm/$name")

  run govc vm.info "$name"
  assert_success
  assert_line "Name: $name"
  assert_line "Path: $path"

  run govc vm.info "$path"
  assert_success
  assert_line "Name: $name"
  assert_line "Path: $path"

  run govc find vm -name "$name"
  assert_success "vm/$name"

  # create a portgroup where name includes a '/'
  net=$(new_id)/with-slash

  run govc host.portgroup.add -vswitch vSwitch0 "$net"
  assert_success

  run govc vm.network.change -vm "$name" -net "$net" ethernet-0
  assert_success

  # change VM eth0 to use network that includes a '/' in the name
  run govc device.info -vm "$name" ethernet-0
  assert_success
  assert_line "Summary: $net"

  run govc host.portgroup.remove "$net"
  assert_success
}

@test "vm.console" {
  esx_env

  vm=$(new_empty_vm)

  run govc vm.console "$vm"
  assert_failure

  run govc vm.power -on "$vm"
  assert_success

  run govc vm.console "$vm"
  assert_success

  run govc vm.console -capture - "$vm"
  assert_success
}

@test "vm.upgrade" {
  vcsim_env

  vm=$(new_id)

  run govc vm.create -on=false -version 0.5 "$vm"
  assert_failure

  run govc vm.create -on=false -version 5.5 "$vm"
  assert_success

  run govc object.collect -s "vm/$vm" config.version
  assert_success "vmx-10"

  run govc vm.upgrade -vm "$vm"
  assert_success

  version=$(govc object.collect -s "vm/$vm" config.version)
  [[ "$version" > "vmx-10" ]]

  run govc vm.upgrade -vm "$vm"
  assert_success

  run govc vm.create -on=false -version vmx-11 "$(new_id)"
  assert_success
}

@test "vm.markastemplate" {
  vcsim_env

  id=$(new_id)

  run govc vm.create -on=true "$id"
  assert_success

  run govc vm.markastemplate "$id"
  assert_failure

  run govc vm.power -off "$id"
  assert_success

  run govc vm.markastemplate "$id"
  assert_success

  run govc vm.power -on "$id"
  assert_failure
}

@test "vm.option.info" {
  vcsim_env

  run govc vm.option.info -host "$GOVC_HOST"
  assert_success

  run govc vm.option.info -cluster "$(dirname "$GOVC_HOST")"
  assert_success

  run govc vm.option.info -vm DC0_H0_VM0
  assert_success
}
