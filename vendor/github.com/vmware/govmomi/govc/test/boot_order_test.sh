#!/bin/bash -e

# This test is not run via bats.
# A VNC session will be opened to observe the VM boot order:
# 1) from floppy  (followed by: eject floppy, reboot)
# 2) from cdrom   (followed by: eject cdrom, reboot)
# 3) from network (will timeout)
# 4) from disk

. $(dirname $0)/test_helper.bash

upload_img
upload_iso

id=$(new_ttylinux_vm)

function cleanup() {
  quit_vnc $vnc
  govc vm.destroy $id
  pkill -TERM -g $$ ^nc
}

trap cleanup EXIT

govc device.cdrom.add -vm $id > /dev/null
govc device.cdrom.insert -vm $id $GOVC_TEST_ISO

govc device.floppy.add -vm $id > /dev/null
govc device.floppy.insert -vm $id $GOVC_TEST_IMG

govc device.boot -vm $id -delay 1000 -order floppy,cdrom,ethernet,disk

vnc=$(govc vm.vnc -port 21122 -password govmomi -enable "${id}" | awk '{print $2}')

echo "booting from floppy..."
govc vm.power -on $id

open_vnc $vnc

sleep 10

govc vm.power -off $id

govc device.floppy.eject -vm $id

# this is ttylinux-live, notice the 'boot:' prompt vs 'login:' prompt when booted from disk
echo "booting from cdrom..."
govc vm.power -on $id

sleep 10

govc vm.power -off $id

govc device.cdrom.eject -vm $id

govc device.serial.add -vm $id > /dev/null
govc device.serial.connect -vm $id -

echo "booting from network, will timeout then boot from disk..."
govc vm.power -on $id

# serial console log
device=$(govc device.ls -vm "$id" | grep serialport- | awk '{print $1}')
govc datastore.tail -f "$id/$device.log" &

ip=$(govc vm.ip $id)

echo "VM booted from disk (ip=$ip)"

sleep 5

govc vm.power -s $id

sleep 5
