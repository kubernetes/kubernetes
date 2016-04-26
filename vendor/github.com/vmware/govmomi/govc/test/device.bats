#!/usr/bin/env bats

load test_helper

@test "device.ls" {
  vm=$(new_empty_vm)

  result=$(govc device.ls -vm $vm | grep ethernet-0 | wc -l)
  [ $result -eq 1 ]
}

@test "device.info" {
  vm=$(new_empty_vm)

  run govc device.info -vm $vm ide-200
  assert_success

  run govc device.info -vm $vm ide-20000
  assert_failure

  run govc device.info -vm $vm -net enoent
  assert_failure

  run govc device.info -vm $vm -net "VM Network" ide-200
  assert_failure

  result=$(govc device.info -vm $vm -net "VM Network" | grep "MAC Address" | wc -l)
  [ $result -eq 1 ]
}

@test "device.boot" {
  vm=$(new_ttylinux_vm)

  result=$(govc device.ls -vm $vm -boot | wc -l)
  [ $result -eq 0 ]

  run govc device.boot -vm $vm -order floppy,cdrom,ethernet,disk
  assert_success

  result=$(govc device.ls -vm $vm -boot | wc -l)
  [ $result -eq 2 ]

  run govc device.cdrom.add -vm $vm
  assert_success

  run govc device.floppy.add -vm $vm
  assert_success

  run govc device.boot -vm $vm -order floppy,cdrom,ethernet,disk
  assert_success

  result=$(govc device.ls -vm $vm -boot | wc -l)
  [ $result -eq 4 ]
}

@test "device.cdrom" {
  vm=$(new_empty_vm)

  result=$(govc device.ls -vm $vm | grep cdrom- | wc -l)
  [ $result -eq 0 ]

  run govc device.cdrom.add -vm $vm
  assert_success
  id=$output

  result=$(govc device.ls -vm $vm | grep $id | wc -l)
  [ $result -eq 1 ]

  run govc device.info -vm $vm $id
  assert_success

  run govc device.cdrom.insert -vm $vm -device $id x.iso
  assert_success

  run govc device.info -vm $vm $id
  assert_line "Summary: ISO [${GOVC_DATASTORE}] x.iso"

  run govc device.disconnect -vm $vm $id
  assert_success

  run govc device.connect -vm $vm $id
  assert_success

  run govc device.remove -vm $vm $id
  assert_success

  run govc device.disconnect -vm $vm $id
  assert_failure "govc: device '$id' not found"

  run govc device.cdrom.insert -vm $vm -device $id x.iso
  assert_failure "govc: device '$id' not found"

  run govc device.remove -vm $vm $id
  assert_failure "govc: device '$id' not found"
}

@test "device.floppy" {
  vm=$(new_empty_vm)

  result=$(govc device.ls -vm $vm | grep floppy- | wc -l)
  [ $result -eq 0 ]

  run govc device.floppy.add -vm $vm
  assert_success
  id=$output

  result=$(govc device.ls -vm $vm | grep $id | wc -l)
  [ $result -eq 1 ]

  run govc device.info -vm $vm $id
  assert_success

  run govc device.floppy.insert -vm $vm -device $id x.img
  assert_success

  run govc device.info -vm $vm $id
  assert_line "Summary: Image [${GOVC_DATASTORE}] x.img"

  run govc device.disconnect -vm $vm $id
  assert_success

  run govc device.connect -vm $vm $id
  assert_success

  run govc device.remove -vm $vm $id
  assert_success

  run govc device.disconnect -vm $vm $id
  assert_failure "govc: device '$id' not found"

  run govc device.floppy.insert -vm $vm -device $id x.img
  assert_failure "govc: device '$id' not found"

  run govc device.remove -vm $vm $id
  assert_failure "govc: device '$id' not found"
}

@test "device.serial" {
  vm=$(new_empty_vm)

  result=$(govc device.ls -vm $vm | grep serial- | wc -l)
  [ $result -eq 0 ]

  run govc device.serial.add -vm $vm
  assert_success
  id=$output

  result=$(govc device.ls -vm $vm | grep $id | wc -l)
  [ $result -eq 1 ]

  run govc device.info -vm $vm $id
  assert_success

  uri=telnet://:33233
  run govc device.serial.connect -vm $vm -device $id $uri
  assert_success

  run govc device.info -vm $vm $id
  assert_line "Summary: Remote $uri"

  run govc device.serial.disconnect -vm $vm -device $id
  assert_success

  run govc device.info -vm $vm $id
  assert_line "Summary: Remote localhost:0"

  run govc device.disconnect -vm $vm $id
  assert_success

  run govc device.connect -vm $vm $id
  assert_success

  run govc device.remove -vm $vm $id
  assert_success

  run govc device.disconnect -vm $vm $id
  assert_failure "govc: device '$id' not found"

  run govc device.serial.connect -vm $vm -device $id $uri
  assert_failure "govc: device '$id' not found"

  run govc device.remove -vm $vm $id
  assert_failure "govc: device '$id' not found"
}

@test "device.scsi" {
  vm=$(new_empty_vm)

  result=$(govc device.ls -vm $vm | grep lsilogic- | wc -l)
  [ $result -eq 1 ]

  run govc device.scsi.add -vm $vm
  assert_success
  id=$output

  result=$(govc device.ls -vm $vm | grep $id | wc -l)
  [ $result -eq 1 ]

  result=$(govc device.ls -vm $vm | grep lsilogic- | wc -l)
  [ $result -eq 2 ]

  run govc device.scsi.add -vm $vm -type pvscsi
  assert_success
  id=$output

  result=$(govc device.ls -vm $vm | grep $id | wc -l)
  [ $result -eq 1 ]
}
