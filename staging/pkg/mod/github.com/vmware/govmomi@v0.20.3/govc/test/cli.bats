#!/usr/bin/env bats

load test_helper

@test "about" {
  vcsim_env -esx

  run govc about
  assert_success
  assert_line "Vendor: VMware, Inc."

  run env GOVC_TLS_HANDSHAKE_TIMEOUT=10s govc about
  assert_success
  assert_line "Vendor: VMware, Inc."

  run env GOVC_TLS_HANDSHAKE_TIMEOUT=NOT_A_DURATION govc about
  assert_failure

  run govc about -json
  assert_success

  run govc about -json -l
  assert_success

  run govc about -dump
  assert_success

  run govc about -dump -l
  assert_success
}

@test "about.cert" {
  vcsim_env -esx

  run govc about.cert
  assert_success

  run govc about.cert -json
  assert_success

  run govc about.cert -show
  assert_success

  # with -k=true we get thumbprint output and exit 0
  thumbprint=$(govc about.cert -k=true -thumbprint)

  # with -k=true we get thumbprint output and exit 60
  run govc about.cert -k=false -thumbprint
  if [ "$status" -ne 60 ]; then
    flunk $(printf "expected failed exit status=60, got status=%d" $status)
  fi
  assert_output "$thumbprint"

  run govc about -k=false
  assert_failure

  run govc about -k=false -tls-known-hosts <(echo "$thumbprint")
  assert_success

  run govc about -k=false -tls-known-hosts <(echo "nope nope")
  assert_failure
}

@test "version" {
  vcsim_env -esx

  run govc version
  assert_success

  v=$(govc version | awk '{print $NF}')
  run govc version -require "$v"
  assert_success

  run govc version -require "not-a-version-string"
  assert_failure

  run govc version -require 100.0.0
  assert_failure
}

@test "login attempt without credentials" {
  vcsim_env -esx

  host=$(govc env -x GOVC_URL_HOST)
  port=$(govc env -x GOVC_URL_PORT)
  run govc about -u "enoent@$host:$port"
  assert_failure "govc: ServerFaultCode: Login failure"
}

@test "login attempt with GOVC_URL, GOVC_USERNAME, and GOVC_PASSWORD" {
  vcsim_env -esx

  govc_url_to_vars
  run govc about
  assert_success
}

@test "API version check" {
  vcsim_env -esx

  run env GOVC_MIN_API_VERSION=24.4 govc about
  assert grep -q "^govc: Require API version \"24.4\"," <<<"${output}"

  run env GOVC_MIN_API_VERSION=no.no govc about
  assert_failure

  run env GOVC_MIN_API_VERSION=- govc about
  assert_success

  run env GOVC_MIN_API_VERSION=5.0 govc about
  assert_success

  run govc about -vim-namespace urn:vim25 -vim-version 6.0
  assert_success
}

@test "govc env" {
  output="$(govc env -x -u 'user:pass@enoent:99999?key=val#anchor')"
  assert grep -q GOVC_URL=enoent:99999 <<<${output}
  assert grep -q GOVC_USERNAME=user <<<${output}
  assert grep -q GOVC_PASSWORD=pass <<<${output}

  assert grep -q GOVC_URL_SCHEME=https <<<${output}
  assert grep -q GOVC_URL_HOST=enoent <<<${output}
  assert grep -q GOVC_URL_PORT=99999 <<<${output}
  assert grep -q GOVC_URL_PATH=/sdk <<<${output}
  assert grep -q GOVC_URL_QUERY=key=val <<<${output}
  assert grep -q GOVC_URL_FRAGMENT=anchor <<<${output}

  password="pa\$sword!ok"
  run govc env -u "user:${password}@enoent:99999" GOVC_PASSWORD
  assert_output "$password"
}

@test "govc help" {
  run govc
  assert_failure

  run govc -h
  assert_success

  run govc -enoent
  assert_failure

  run govc vm.create
  assert_failure

  run govc vm.create -h
  assert_success

  run govc vm.create -enoent
  assert_failure
}
