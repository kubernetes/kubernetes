#!/usr/bin/env bats

load test_helper

@test "about" {
  run govc about
  assert_success
  assert_line "Vendor: VMware, Inc."
}

@test "login attempt without credentials" {
  run govc about -u $(echo $GOVC_URL | awk -F@ '{print $2}')
  assert_failure "govc: ServerFaultCode: Cannot complete login due to an incorrect user name or password."
}

@test "login attempt with GOVC_URL, GOVC_USERNAME, and GOVC_PASSWORD" {
  govc_url_to_vars
  run govc about
  assert_success
}

@test "connect to an endpoint with a non-supported API version" {
  run env GOVC_MIN_API_VERSION=24.4 govc about
  assert grep -q "^govc: Require API version 24.4," <<<${output}
}

@test "connect to an endpoint with user provided Vim namespace and Vim version" {
  run govc about -vim-namespace urn:vim25 -vim-version 6.0
  assert_success
}
