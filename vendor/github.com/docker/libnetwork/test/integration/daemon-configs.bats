#!/usr/bin/env bats

load helpers

export DRIVER=virtualbox
export NAME="bats-$DRIVER-daemon-configs"
export MACHINE_STORAGE_PATH=/tmp/machine-bats-daemon-test-$DRIVER
# Default memsize is 1024MB and disksize is 20000MB
# These values are defined in drivers/virtualbox/virtualbox.go
export DEFAULT_MEMSIZE=1024
export DEFAULT_DISKSIZE=20000
export CUSTOM_MEMSIZE=1536
export CUSTOM_DISKSIZE=10000
export CUSTOM_CPUCOUNT=1
export BAD_URL="http://dev.null:9111/bad.iso"

function setup() {
  # add sleep because vbox; ugh
  sleep 1
}

findDiskSize() {
  # SATA-0-0 is usually the boot2disk.iso image
  # We assume that SATA 1-0 is root disk VMDK and grab this UUID
  # e.g. "SATA-ImageUUID-1-0"="fb5f33a7-e4e3-4cb9-877c-f9415ae2adea"
  # TODO(slashk): does this work on Windows ?
  run bash -c "VBoxManage showvminfo --machinereadable $NAME | grep SATA-ImageUUID-1-0 | cut -d'=' -f2"
  run bash -c "VBoxManage showhdinfo $output | grep "Capacity:" | awk -F' ' '{ print $2 }'"
}

findMemorySize() {
  run bash -c "VBoxManage showvminfo --machinereadable $NAME | grep memory= | cut -d'=' -f2"
}

findCPUCount() {
  run bash -c "VBoxManage showvminfo --machinereadable $NAME | grep cpus= | cut -d'=' -f2"
}

buildMachineWithOldIsoCheckUpgrade() {
  run wget https://github.com/boot2docker/boot2docker/releases/download/v1.4.1/boot2docker.iso -O $MACHINE_STORAGE_PATH/cache/boot2docker.iso
  run machine create -d virtualbox $NAME
  run machine upgrade $NAME
}

@test "$DRIVER: machine should not exist" {
  run machine active $NAME
  [ "$status" -eq 1  ]
}

@test "$DRIVER: VM should not exist" {
  run VBoxManage showvminfo $NAME
  [ "$status" -eq 1  ]
}

@test "$DRIVER: create" {
  run machine create -d $DRIVER $NAME
  [ "$status" -eq 0  ]
}

@test "$DRIVER: active" {
  run machine active $NAME
  [ "$status" -eq 0  ]
}

@test "$DRIVER: check default machine memory size" {
  findMemorySize
  [[ ${output} == "${DEFAULT_MEMSIZE}"  ]]
}

@test "$DRIVER: check default machine disksize" {
  findDiskSize
  [[ ${output} == *"$DEFAULT_DISKSIZE"* ]]
}

@test "$DRIVER: test bridge-ip" {
  run machine ssh $NAME sudo /etc/init.d/docker stop
  run machine ssh $NAME sudo ifconfig docker0 down
  run machine ssh $NAME sudo ip link delete docker0
  BIP='--bip=172.168.45.1/24'
  set_extra_config $BIP
  cat ${TMP_EXTRA_ARGS_FILE} | machine ssh $NAME sudo tee /var/lib/boot2docker/profile
  cat ${DAEMON_CFG_FILE} | machine ssh $NAME "sudo tee -a /var/lib/boot2docker/profile"
  run machine ssh $NAME sudo /etc/init.d/docker start
  run machine ssh $NAME ifconfig docker0
  [ "$status" -eq 0  ]
  [[ ${lines[1]} =~ "172.168.45.1"  ]]
}

@test "$DRIVER: run busybox container" {
  run machine ssh $NAME sudo cat /var/lib/boot2docker/profile
  run docker $(machine config $NAME) run busybox echo hello world
  [ "$status" -eq 0  ]
}

@test "$DRIVER: remove machine" {
  run machine rm -f $NAME
}

# Cleanup of machine store should always be the last 'test'
@test "$DRIVER: cleanup" {
  run rm -rf $MACHINE_STORAGE_PATH
  [ "$status" -eq 0  ]
}

