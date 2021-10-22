#!/usr/bin/env bats

load test_helper

@test "ls" {
  vcsim_env

  run govc ls
  assert_success
  # /dc/{vm,network,host,datastore}
  n=${#lines[@]}
  [ $n -ge 4 ]

  run govc ls -json
  assert_success

  # list entire inventory
  run govc ls '/**'
  assert_success
  [ ${#lines[@]} -ge $n ]

  run govc ls ./...
  assert_success
  [ ${#lines[@]} -ge $n ]

  run govc ls -t HostSystem '*'
  assert_success
  [ ${#lines[@]} -eq 0 ]

  run govc ls host
  assert_success
  [ ${#lines[@]} -ge 1 ]

  run govc ls enoent
  assert_success
  [ ${#lines[@]} -eq 0 ]
}

@test "ls -R" {
  vcsim_env -esx

  # search entire inventory
  run govc ls ./...
  assert_success
  # should have at least 1 dc + folders, 1 host, 1 network, 1 datastore
  [ ${#lines[@]} -ge 9 ]

  run govc ls -t HostSystem ./...
  assert_success
  [ ${#lines[@]} -eq 1 ]

  run govc ls -t Datacenter /...
  assert_success
  [ ${#lines[@]} -eq 1 ]

  run govc ls -t ResourcePool host/...
  assert_success
  [ ${#lines[@]} -ge 1 ]

  run govc ls -t ResourcePool vm/...
  assert_success
  [ ${#lines[@]} -eq 0 ]

  c=$(govc ls -t ComputeResource ./... | head -1)
  run govc ls -t ResourcePool "$c/..."
  assert_success
  [ ${#lines[@]} -ge 1 ]
}

@test "ls vm" {
  vcsim_env -esx

  vm=$(new_empty_vm)

  run govc ls vm
  assert_success
  [ ${#lines[@]} -ge 1 ]

  run govc ls vm/$vm
  assert_success
  [ ${#lines[@]} -eq 1 ]

  run govc ls /*/vm/$vm
  assert_success
  [ ${#lines[@]} -eq 1 ]
}

@test "ls network" {
  vcsim_env -esx

  run govc ls network
  assert_success
  [ ${#lines[@]} -ge 1 ]

  local path=${lines[0]}
  run govc ls "$path"
  assert_success
  [ ${#lines[@]} -eq 1 ]

  run govc ls "network/$(basename "$path")"
  assert_success
  [ ${#lines[@]} -eq 1 ]

  run govc ls "/*/network/$(basename "$path")"
  assert_success
  [ ${#lines[@]} -eq 1 ]
}

@test "ls multi ds" {
  vcsim_env -dc 2

  run govc ls
  assert_success
  # /DC0/{vm,network,host,datastore}
  [ ${#lines[@]} -eq 4 ]

  run govc ls /DC*
  assert_success
  # /DC[0,1]/{vm,network,host,datastore}
  [ ${#lines[@]} -eq 8 ]

  # here 'vm' is relative to /DC0
  run govc ls vm
  assert_success
  [ ${#lines[@]} -gt 0 ]

  unset GOVC_DATACENTER

  run govc ls
  assert_success
  # /DC[0,1]
  [ ${#lines[@]} -eq 2 ]

  run govc ls -dc enoent
  assert_failure
  [ ${#lines[@]} -gt 0 ]

  # here 'vm' is relative to '/' - so there are no matches
  run govc ls vm
  assert_success
  [ ${#lines[@]} -eq 0 ]

  # ls all vms in all datacenters
  run govc ls */vm
  assert_success
  [ ${#lines[@]} -gt 0 ]
}

@test "ls moref" {
  vcsim_env -esx

  # ensure the vm folder isn't empty
  run govc vm.create -on=false "$(new_id)"
  assert_success

  # list dc folder paths
  folders1=$(govc ls)
  # list dc folder refs | govc ls -L ; should output the same paths
  folders2=$(govc ls -i | xargs govc ls -L)

  assert_equal "$folders1" "$folders2"

  for folder in $folders1
  do
    # list paths in $folder
    items1=$(govc ls "$folder")
    # list refs in $folder | govc ls -L ; should output the same paths
    items2=$(govc ls -i "$folder" | xargs -d '\n' govc ls -L)

    assert_equal "$items1" "$items2"
  done

  ref=ViewManager:ViewManager
  path=$(govc ls -L $ref)
  assert_equal "$ref" "$path"

  path=$(govc ls -L Folder:ha-folder-root)
  assert_equal "/" "$path"
}

@test "ls substr" {
  # Test fix for issue #815, introduced by b35abbc

  vcsim_env

  id=$(new_id)

  run govc vm.create -on=false "${id}"
  assert_success

  run govc vm.create -on=false "bar${id}"
  assert_success

  assert [ "$(govc ls "vm/$id" | wc -l)" -eq 1 ]
}
