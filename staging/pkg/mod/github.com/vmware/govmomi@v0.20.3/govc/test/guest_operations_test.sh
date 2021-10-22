#!/bin/bash -e

# This test is not run via bats.
# 1) Test guest operations (govc guest.* commands)
# 2) Test vm disk persistence

. "$(dirname "$0")"/test_helper.bash

esx_env

import_ttylinux_vmdk

export GOVC_GUEST_LOGIN=root:password

for persist in true false ; do
  id=govc-test-persist-$persist
  govc ls vm/$id | xargs -r govc vm.destroy

  if [ "$persist" = "true" ] ; then
    grepf=-v
    mode=persistent
  else
    mode=independent_nonpersistent
  fi

  echo "Creating vm..."
  govc vm.create -m 32 -disk.controller ide -on=false $id

  # Save some noise by defaulting to '-vm $id'
  export GOVC_VM=$id

  echo "Attaching linked disk..."
  govc vm.disk.attach -controller ide -persist=$persist -link=true -disk "$GOVC_TEST_VMDK"

  echo "Creating data disk..."
  govc vm.disk.create -controller ide -mode=$mode -name "$id"/data -size "10M"

  echo "Powering on vm..."
  govc vm.power -on $id 1>/dev/null
  echo "Waiting for tools to initialize..."
  govc vm.ip $id 1>/dev/null

  echo "Formatting the data disk..."
  govc guest.mkdir /data
  script=$(govc guest.mktemp)

  govc guest.upload -f - "$script" <<'EOF'
#!/bin/sh -xe

opts=(n p 1 1 ' ' w)
printf "%s\n" "${opts[@]}" | fdisk /dev/hdb
mkfs.ext3 /dev/hdb1
mount /dev/hdb1 /data
df -h
cp /etc/motd /data
EOF

  govc guest.chown 65534 "$script"
  govc guest.chown 65534:65534 "$script"
  govc guest.ls "$script" | grep 65534
  govc guest.chmod 0755 "$script"
  pid=$(govc guest.start "$script" '>&' /tmp/disk.log)
  status=$(govc guest.ps -p "$pid" -json -X | jq .ProcessInfo[].ExitCode)
  govc guest.download /tmp/disk.log -
  if [ "$status" -ne "0" ] ; then
    exit 1
  fi

  echo "Writing some data to the disks..."
  for d in /etc /data ; do
    govc guest.touch "$d/motd.bak"
    govc guest.touch -d "$(date -d '1 day ago')" "$d/motd"
    govc guest.ls "$d/motd"
    govc guest.download $d/motd - | grep Chop
  done
  govc version | govc guest.upload -f - /etc/motd
  govc guest.download /etc/motd - | grep -v Chop

  pid=$(govc guest.start /bin/sync)
  status=$(govc guest.ps -p "$pid" -json -X | jq .ProcessInfo[].ExitCode)
  if [ "$status" -ne "0" ] ; then
    exit 1
  fi

  echo "Rebooting vm..."
  govc vm.power -off $id
  govc vm.power -on $id
  echo "Waiting for tools to initialize..."
  govc vm.ip $id 1>/dev/null

  echo "Verifying data persistence..."
  govc guest.download /etc/motd - | grep $grepf Chop
  pid=$(govc guest.start /bin/mount /dev/hdb1 /data)
  status=$(govc guest.ps -p "$pid" -json -X | jq .ProcessInfo[].ExitCode)

  if [ "$persist" = "true" ] ; then
    govc guest.ls /data
    govc guest.download /data/motd - | grep -v Chop
    govc guest.rm /data/motd

    govc guest.mkdir /data/foo/bar/baz 2>/dev/null && exit 1 # should fail
    govc guest.mkdir -p /data/foo/bar/baz

    govc guest.rmdir /data/foo 2>/dev/null && exit 1 # should fail
    govc guest.rmdir /data/foo/bar/baz
    dir=$(govc guest.mktemp -d -p /data/foo -s govc)
    file=$(govc guest.mktemp -p "$dir")
    govc guest.mv -n "$(govc guest.mktemp)" "$file" 2>/dev/null && exit 1 # should fail
    govc guest.mv "$file" "${file}-old"
    govc guest.mv "$dir" "${dir}-old"
    govc guest.rmdir -r /data/foo
    govc guest.ls /data | grep -v foo
  else
    if [ "$status" -eq "0" ] ; then
      echo "expected failure"
      exit 1
    fi
  fi
done
