#!/usr/bin/env bats

load test_helper

@test "pool.create" {
  path="*/Resources/$(new_id)/$(new_id)"
  run govc pool.create $path
  assert_failure
  assert_line "govc: cannot create resource pool '$(basename ${path})': parent not found"

  id=$(new_id)
  path="*/Resources/$id"
  run govc pool.create -cpu.shares low -mem.reservation 500 $path
  assert_success

  run govc pool.info $path
  assert_success

  assert_line "Name: $id"
  assert_line "CPU Shares: low"
  assert_line "Mem Reservation: 500MB (expandable=true)"

  run govc pool.destroy $path
  assert_success
}

@test "pool.create multiple" {
  id=$(new_id)
  path="*/Resources/$id"
  govc pool.create $path

  # Create multiple parent pools with multiple arguments (without globbing)
  run govc pool.create $path/a $path/b
  assert_success
  result=$(govc ls "host/$path/*" | wc -l)
  [ $result -eq 2 ]

  # Create multiple child pools with one argument (with globbing)
  run govc pool.create $path/*/{a,b}
  assert_success
  result=$(govc ls "host/$path/*/*" | wc -l)
  [ $result -eq 4 ]

  # Clean up
  run govc pool.destroy $path/*/* $path/* $path
  assert_success
}

@test "pool.change" {
  id=$(new_id)
  path="*/Resources/$id"
  govc pool.create $path

  run govc pool.change -mem.shares high $path
  assert_success
  run govc pool.info $path
  assert_success
  assert_line "Mem Shares: high"
  assert_line "CPU Shares: normal"

  nid=$(new_id)
  run govc pool.change -name $nid $path
  assert_success
  path="*/Resources/$nid"

  run govc pool.info $path
  assert_success
  assert_line "Name: $nid"

  run govc pool.destroy $path
  assert_success
}

@test "pool.change multiple" {
  id=$(new_id)
  path="*/Resources/$id"
  govc pool.create $path

  # Create some nested pools so that we can test changing multiple in one call
  govc pool.create $path/{a,b} $path/{a,b}/test

  # Test precondition
  run govc pool.info $path/a/test
  assert_success
  assert_line "Name: test"
  run govc pool.info $path/b/test
  assert_success
  assert_line "Name: test"

  # Change name of both test pools
  run govc pool.change -name hello $path/*/test
  assert_success

  # Test postcondition
  run govc pool.info $path/a/hello
  assert_success
  assert_line "Name: hello"
  run govc pool.info $path/b/hello
  assert_success
  assert_line "Name: hello"

  # Clean up
  govc pool.destroy $path/a/hello
  govc pool.destroy $path/a
  govc pool.destroy $path/b/hello
  govc pool.destroy $path/b
  govc pool.destroy $path
}

@test "pool.destroy" {
  id=$(new_id)

  # parent pool
  path="*/Resources/$id"
  run govc pool.create $path
  assert_success

  result=$(govc ls "host/$path/*" | wc -l)
  [ $result -eq 0 ]

  # child pools
  id1=$(new_id)
  run govc pool.create $path/$id1
  assert_success

  id2=$(new_id)
  run govc pool.create $path/$id2
  assert_success

  # 2 child pools
  result=$(govc ls "host/$path/*" | wc -l)
  [ $result -eq 2 ]

  # 1 parent pool
  result=$(govc ls "host/$path" | wc -l)
  [ $result -eq 1 ]

  run govc pool.destroy $path
  assert_success

  # no more parent pool
  result=$(govc ls "host/$path" | wc -l)
  [ $result -eq 0 ]

  # the child pools are not present anymore
  # the only place they could pop into is the parent pool

  # first child pool
  result=$(govc ls "host/*/Resources/$id1" | wc -l)
  [ $result -eq 0 ]

  # second child pool
  result=$(govc ls "host/*/Resources/$id2" | wc -l)
  [ $result -eq 0 ]
}

@test "pool.destroy children" {
  id=$(new_id)

  # parent pool
  path="*/Resources/$id"
  run govc pool.create $path
  assert_success

  result=$(govc ls "host/$path/*" | wc -l)
  [ $result -eq 0 ]

  # child pools
  run govc pool.create $path/$(new_id)
  assert_success

  run govc pool.create $path/$(new_id)
  assert_success

  # 2 child pools
  result=$(govc ls "host/$path/*" | wc -l)
  [ $result -eq 2 ]

  # 1 parent pool
  result=$(govc ls "host/*/Resources/govc-test-*" | wc -l)
  [ $result -eq 1 ]

  # delete childs
  run govc pool.destroy -children $path
  assert_success

  # no more child pools
  result=$(govc ls "host/$path/*" | wc -l)
  [ $result -eq 0 ]

  # cleanup
  run govc pool.destroy $path
  assert_success

  # cleanup check
  result=$(govc ls "host/$path" | wc -l)
  [ $result -eq 0 ]
}

@test "pool.destroy multiple" {
  id=$(new_id)
  path="*/Resources/$id"
  govc pool.create $path

  # Create some nested pools so that we can test destroying multiple in one call
  govc pool.create $path/{a,b}

  # Test precondition
  result=$(govc ls "host/$path/*" | wc -l)
  [ $result -eq 2 ]

  # Destroy both pools
  run govc pool.destroy $path/{a,b}
  assert_success

  # Test postcondition
  result=$(govc ls "host/$path/*" | wc -l)
  [ $result -eq 0 ]

  # Clean up
  govc pool.destroy $path
}

@test "vm.create -pool" {
  # test with full inventory path to pools
  parent_path=$(govc ls 'host/*/Resources')
  parent_name=$(basename $parent_path)
  [ "$parent_name" = "Resources" ]

  child_name=$(new_id)
  child_path="$parent_path/$child_name"

  grand_child_name=$(new_id)
  grand_child_path="$child_path/$grand_child_name"

  run govc pool.create $parent_path/$child_name{,/$grand_child_name}
  assert_success

  for path in $parent_path $child_path $grand_child_path
  do
    run govc vm.create -pool $path $(new_id)
    assert_success
  done

  run govc pool.change -mem.limit 100 -mem.expandable=false $child_path
  assert_failure

  run govc pool.change -mem.limit 100 $child_path
  assert_success

  run govc pool.change -mem.limit 120 -mem.expandable $child_path
  assert_success

  # test with glob inventory path to pools
  parent_path="*/$parent_name"
  child_path="$parent_path/$child_name"
  grand_child_path="$child_path/$grand_child_name"

  for path in $grand_child_path $child_path
  do
    run govc pool.destroy $path
    assert_success
  done
}

@test "vm.create -pool host" {
  id=$(new_id)

  path=$(govc ls host)

  run govc vm.create -on=false -pool enoent $id
  assert_failure "govc: resource pool 'enoent' not found"

  run govc vm.create -on=false -pool $path $id
  assert_success
}

@test "vm.create -pool cluster" {
  vcsim_env

  id=$(new_id)

  path=$(dirname $GOVC_HOST)

  unset GOVC_HOST
  unset GOVC_RESOURCE_POOL

  run govc vm.create -on=false -pool enoent $id
  assert_failure "govc: resource pool 'enoent' not found"

  run govc vm.create -on=false -pool $path $id
  assert_success
}
