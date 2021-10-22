#!/usr/bin/env bats

load test_helper

@test "disk.ls" {
  vcsim_env

  run govc disk.ls
  assert_success

  run govc disk.ls enoent
  assert_failure
}

@test "disk.create" {
  vcsim_env

  name=$(new_id)

  run govc disk.create -size 10M "$name"
  assert_success
  id="${lines[1]}"

  run govc disk.ls "$id"
  assert_success

  govc disk.ls -json "$id" | jq .

  run govc disk.rm "$id"
  assert_success

  run govc disk.rm "$id"
  assert_failure
}

@test "disk.create -datastore-cluster" {
  vcsim_env -pod 1 -ds 3 -cluster 2

  pod=/DC0/datastore/DC0_POD0
  id=$(new_id)

  run govc disk.create -datastore-cluster $pod "$id"
  assert_failure

  run govc object.mv /DC0/datastore/LocalDS_{1,2} $pod
  assert_success

  run govc disk.create -datastore-cluster $pod -size 10M "$id"
  assert_success

  id=$(new_id)
  pool=$GOVC_RESOURCE_POOL
  unset GOVC_RESOURCE_POOL
  run govc disk.create -datastore-cluster $pod -size 10M "$id"
  assert_failure # -pool is required

  run govc disk.create -datastore-cluster $pod -size 10M -pool "$pool" "$id"
  assert_success
}

@test "disk.register" {
  vcsim_env

  id=$(new_id)
  vmdk="$id/$id.vmdk"

  run govc datastore.mkdir "$id"
  assert_success

  # create with VirtualDiskManager
  run govc datastore.disk.create -size 10M "$vmdk"
  assert_success

  run govc disk.register "$id" "$id"
  assert_failure # expect fail for directory

  run govc disk.register "" "$id"
  assert_failure # expect fail for empty path

  run govc disk.register "$vmdk" "$id"
  assert_success
  id="$output"

  run govc disk.ls "$id"
  assert_success

  run govc disk.register "$vmdk" "$id"
  assert_failure

  run govc disk.rm "$id"
  assert_success

  run govc disk.rm "$id"
  assert_failure
}

@test "disk.snapshot" {
  vcsim_env

  name=$(new_id)

  run govc disk.create -size 10M "$name"
  assert_success
  id="${lines[1]}"

  run govc disk.snapshot.ls "$id"
  assert_success

  run govc disk.snapshot.create "$id"
  assert_success
  sid="${lines[1]}"

  govc disk.snapshot.ls "$id" | grep "$sid"

  govc disk.snapshot.ls -json "$id" | jq .

  run govc disk.snapshot.rm "$id" "$sid"
  assert_success

  run govc disk.snapshot.rm "$id" "$sid"
  assert_failure

  run govc disk.rm "$id"
  assert_success
}

@test "disk.tags" {
  vcsim_env

  run govc tags.category.create region
  assert_success

  run govc tags.create -c region US-WEST
  assert_success

  name=$(new_id)

  run govc disk.create -size 10M "$name"
  assert_success
  id="${lines[1]}"

  run govc disk.ls "$id"
  assert_success

  run govc disk.ls -c region -t US-WEST
  assert_success ""

  govc disk.ls -T | grep -v US-WEST

  run govc disk.tags.attach -c region US-WEST "$id"
  assert_success

  run govc disk.ls -c region -t US-WEST
  assert_success
  assert_matches "$id"

  run govc disk.ls -T
  assert_success
  assert_matches US-WEST

  run govc disk.tags.detach -c region enoent "$id"
  assert_failure

  run govc disk.tags.detach -c region US-WEST "$id"
  assert_success

  govc disk.ls -T | grep -v US-WEST
}
