#!/usr/bin/env bats

load test_helper

@test "vcsim rbvmomi" {
  if ! ruby -e "require 'rbvmomi'" ; then
    skip "requires rbvmomi"
  fi

  vcsim_env

  ruby ./vcsim_test.rb "$(govc env -x GOVC_URL_PORT)"
}

@test "vcsim examples" {
  vcsim_env

  # compile + run examples against vcsim
  for main in ../../examples/*/main.go ; do
    run go run "$main" -insecure -url "$GOVC_URL"
    assert_success
  done
}

@test "vcsim about" {
  vcsim_env -dc 2 -cluster 3 -vm 0 -ds 0

  url="https://$(govc env GOVC_URL)"

  run curl -skf "$url/about"
  assert_matches "CurrentTime" # 1 param (without Context)
  assert_matches "TerminateSession" # 2 params (with Context)

  run curl -skf "$url/debug/vars"
  assert_success

  model=$(curl -sfk "$url/debug/vars" | jq .vcsim.Model)
  [ "$(jq .Datacenter <<<"$model")" == "2" ]
  [ "$(jq .Cluster <<<"$model")" == "6" ]
  [ "$(jq .Machine <<<"$model")" == "0" ]
  [ "$(jq .Datastore <<<"$model")" == "0" ]
}

@test "vcsim host placement" {
  vcsim_start -dc 0

  # https://github.com/vmware/govmomi/issues/1258
  id=$(new_id)
  govc datacenter.create DC0
  govc cluster.create comp
  govc cluster.add -cluster comp -hostname test.host.com -username user -password pass
  govc cluster.add -cluster comp -hostname test2.host.com -username user -password pass
  govc datastore.create -type local -name vol6 -path "$TMPDIR" test.host.com
  govc pool.create comp/Resources/testPool
  govc vm.create -c 1 -ds vol6 -g centos64Guest -pool testPool -m 4096 "$id"
  govc vm.destroy "$id"
}

@test "vcsim host config.port" {
  vcsim_start -dc 0
  url=$(govc env GOVC_URL)
  port=$(govc env -x GOVC_URL_PORT)
  vcsim_stop

  vcsim_start -httptest.serve="$url" # reuse free port selection from above

  run govc object.collect -s -type h host/DC0_H0 summary.config.port
  assert_success "$port"
  ports=$(govc object.collect -s -type h / summary.config.port | uniq -u | wc -l)
  [ "$ports" = "0" ] # all host ports should be the same value

  vcsim_stop

  VCSIM_HOST_PORT_UNIQUE=true vcsim_start -httptest.serve="$url"

  hosts=$(curl -sk "https://$url/debug/vars" | jq .vcsim.Model.Host)
  ports=$(govc object.collect -s -type h / summary.config.port | uniq -u | wc -l)
  [ "$ports" = "$hosts" ] # all host ports should be unique
  [[ "$ports" != *$port* ]] # host ports should not include vcsim port
}

@test "vcsim set vm properties" {
  vcsim_env

  vm=/DC0/vm/DC0_H0_VM0

  run govc object.collect $vm guest.ipAddress
  assert_success ""

  run govc vm.change -vm $vm -e SET.guest.ipAddress=10.0.0.1
  assert_success

  run govc object.collect -s $vm guest.ipAddress
  assert_success "10.0.0.1"

  run govc object.collect -s $vm summary.guest.ipAddress
  assert_success "10.0.0.1"

  netip=$(govc object.collect -json -s $vm guest.net | jq -r .[].Val.GuestNicInfo[].IpAddress[0])
  [ "$netip" = "10.0.0.1" ]

  run govc vm.info -vm.ip 10.0.0.1
  assert_success

  run govc object.collect -s $vm guest.hostName
  assert_success ""

  run govc vm.change -vm $vm -e SET.guest.hostName=localhost.localdomain
  assert_success

  run govc object.collect -s $vm guest.hostName
  assert_success "localhost.localdomain"

  run govc object.collect -s $vm summary.guest.hostName
  assert_success "localhost.localdomain"

  run govc vm.info -vm.dns localhost.localdomain
  assert_success

  uuid=$(uuidgen)
  run govc vm.change -vm $vm -e SET.config.uuid="$uuid"
  assert_success

  run govc object.collect -s $vm config.uuid
  assert_success "$uuid"
}

@test "vcsim vm.create" {
  vcsim_env

  run govc vm.create foo.yakity
  assert_success

  run govc vm.create bar.yakity
  assert_success
}

@test "vcsim issue #1251" {
  vcsim_env

  govc object.collect -type ComputeResource -n 1 / name &
  pid=$!

  run govc object.rename /DC0/host/DC0_C0 DC0_C0b
  assert_success

  wait $pid

  govc object.collect -type ClusterComputeResource -n 1 / name &
  pid=$!

  run govc object.rename /DC0/host/DC0_C0b DC0_C0
  assert_success

  wait $pid
}

@test "vcsim run container" {
  if ! docker version ; then
    skip "docker client not installed"
  fi

  vm=DC0_H0_VM0

  if docker inspect $vm ; then
    flunk "$vm container still exists"
  fi

  vcsim_env -autostart=false

  run govc vm.change -vm $vm -e RUN.container=nginx
  assert_success

  run govc vm.power -on $vm
  assert_success

  if ! docker inspect $vm ; then
    flunk "$vm container does not exist"
  fi

  ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $vm)
  run govc object.collect -s vm/$vm guest.ipAddress
  assert_success "$ip"

  run govc object.collect -s vm/$vm summary.guest.ipAddress
  assert_success "$ip"

  netip=$(govc object.collect -json -s vm/$vm guest.net | jq -r .[].Val.GuestNicInfo[].IpAddress[0])
  [ "$netip" = "$ip" ]

  run govc vm.power -s $vm
  assert_success

  run docker inspect -f '{{.State.Status}}' $vm
  assert_success "exited"

  run govc vm.power -on $vm
  assert_success

  run docker inspect -f '{{.State.Status}}' $vm
  assert_success "running"

  run govc vm.destroy $vm
  assert_success

  if docker inspect $vm ; then
    flunk "$vm container still exists"
  fi

  vm=DC0_H0_VM1

  # test json encoded args
  run govc vm.change -vm $vm -e RUN.container="[\"-v\", \"$PWD:/usr/share/nginx/html:ro\", \"nginx\"]"
  assert_success

  run govc vm.power -on $vm
  assert_success

  run docker inspect $vm
  assert_success

  ip=$(govc object.collect -s vm/$vm guest.ipAddress)
  run curl -f "http://$ip/vcsim.bats"
  assert_success

  # test suspend/resume
  run docker inspect -f '{{.State.Status}}' $vm
  assert_success "running"

  run govc vm.power -suspend $vm
  assert_success

  run docker inspect -f '{{.State.Status}}' $vm
  assert_success "paused"

  run govc vm.power -on $vm
  assert_success

  run docker inspect -f '{{.State.Status}}' $vm
  assert_success "running"

  run govc vm.destroy $vm
  assert_success
}

@test "vcsim listen" {
  vcsim_start -dc 0
  url=$(govc option.ls vcsim.server.url)
  [[ "$url" == *"https://127.0.0.1:"* ]]
  vcsim_stop

  vcsim_start -dc 0 -httptest.serve 0.0.0.0:0
  url=$(govc option.ls vcsim.server.url)
  [[ "$url" != *"https://127.0.0.1:"* ]]
  [[ "$url" != *"https://[::]:"* ]]
  vcsim_stop

  vcsim_start -dc 0 -l :0 -httptest.serve ""
  url=$(govc option.ls vcsim.server.url)
  [[ "$url" != *"https://127.0.0.1:"* ]]
  [[ "$url" != *"https://[::]:"* ]]
  vcsim_stop
}

@test "vcsim vapi auth" {
  vcsim_env

  url=$(govc env GOVC_URL)

  run curl -fsk "https://$url/rest/com/vmware/cis/tagging/tag"
  [ "$status" -ne 0 ] # not authenticated

  run curl -fsk -X POST "https://$url/rest/com/vmware/cis/session"
  [ "$status" -ne 0 ] # no basic auth header

  run curl -fsk -X POST --user user: "https://$url/rest/com/vmware/cis/session"
  [ "$status" -ne 0 ] # no password

  run curl -fsk -X POST --user "$USER:pass" "https://$url/rest/com/vmware/cis/session"
  assert_success # login with user:pass

  id=$(jq -r .value <<<"$output")

  run curl -fsk "https://$url/rest/com/vmware/cis/session"
  [ "$status" -ne 0 ] # no header or cookie

  run curl -fsk "https://$url/rest/com/vmware/cis/session" -H "vmware-api-session-id:$id"
  assert_success # valid session header

  user=$(jq -r .value.user <<<"$output")
  assert_equal "$USER" "$user"
}
