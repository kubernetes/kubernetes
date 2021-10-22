#!/usr/bin/env bats

load test_helper

@test "datacenter.info" {
  vcsim_env -esx

  dc=$(govc ls -t Datacenter / | head -n1)
  run govc datacenter.info "$dc"
  assert_success

  run govc datacenter.info -json "$dc"
  assert_success

  run govc datacenter.info /enoent
  assert_failure
}

@test "datacenter.create" {
  vcsim_env
  unset GOVC_DATACENTER

  # name not specified
  run govc datacenter.create
  assert_failure

  dcs=($(new_id) $(new_id))
  run govc datacenter.create "${dcs[@]}"
  assert_success

  for dc in ${dcs[*]}; do
    run govc ls "/$dc"
    assert_success
    # /<datacenter>/{vm,network,host,datastore}
    [ ${#lines[@]} -eq 4 ]

    run govc datacenter.info "/$dc"
    assert_success
  done

  run govc object.destroy "/$dc"
  assert_success
}

@test "datacenter commands fail against ESX" {
  vcsim_env -esx

  run govc datacenter.create something
  assert_failure

  run govc object.destroy /ha-datacenter
  assert_failure
}
