#!/usr/bin/env bats

load test_helper

@test "session.ls" {
  esx_env

  run govc session.ls
  assert_success

  run govc session.ls -json
  assert_success

  # Test User-Agent
  govc session.ls | grep "$(govc version | tr ' ' /)"
}

@test "session.rm" {
  esx_env

  run govc session.rm enoent
  assert_failure
  assert_output "govc: ServerFaultCode: The object or item referred to could not be found."

  # Can't remove the current session
  id=$(govc session.ls -json | jq -r .CurrentSession.Key)
  run govc session.rm "$id"
  assert_failure

  thumbprint=$(govc about.cert -thumbprint)
  # persist session just to avoid the Logout() so we can session.rm below
  dir=$(mktemp -d govc-test-XXXXX)

  id=$(GOVMOMI_HOME="$dir" govc session.ls -json -k=false -persist-session -tls-known-hosts <(echo "$thumbprint") | jq -r .CurrentSession.Key)

  rm -rf "$dir"

  run govc session.rm "$id"
  assert_success
}

@test "session.login" {
    vcsim_env

    # Remove username/password
    host=$(govc env GOVC_URL)

    # Validate auth is not required for service content
    run govc about -u "$host"
    assert_success

    # Auth is required here
    run govc ls -u "$host"
    assert_failure

    cookie=$(govc session.login -l)
    ticket=$(govc session.login -cookie "$cookie" -clone)

    run govc session.login -u "$host" -ticket "$ticket"
    assert_success
}

@test "session.loginbytoken" {
  vcsim_env

  # Remove username/password
  host=$(govc env GOVC_URL)
  # Token template, vcsim just checks Assertion.Subject.NameID
  token="<Assertion><Subject><NameID>%s</NameID></Subject></Assertion>"

  # shellcheck disable=2059
  run govc session.login -l -token "$(printf $token "")"
  assert_failure # empty NameID is a InvalidLogin fault

  # shellcheck disable=2059
  run govc session.login -l -token "$(printf $token root@localos)"
  assert_success # non-empty NameID is enough to login

  id=$(new_id)
  run govc extension.setcert -cert-pem ++ "$id" # generate a cert for testing
  assert_success

  # Test with STS simulator issued token
  token="$(govc session.login -issue)"
  run govc session.login -cert "$id.crt" -key "$id.key" -l -token "$token"
  assert_success

  run govc session.login -cert "$id.crt" -key "$id.key" -l -renew
  assert_failure # missing -token

  run govc session.login -cert "$id.crt" -key "$id.key" -l -renew -lifetime 24h -token "$token"
  assert_success

  # remove generated cert and key
  rm "$id".{crt,key}
}

@test "session.loginextension" {
  vcsim_env -tunnel 0

  run govc session.login -extension com.vmware.vsan.health
  assert_failure # no certificate

  id=$(new_id)
  run govc extension.setcert -cert-pem ++ "$id" # generate a cert for testing
  assert_success

  # vcsim will login if any certificate is provided
  run govc session.login -extension com.vmware.vsan.health -cert "$id.crt" -key "$id.key"
  assert_success

  # remove generated cert and key
  rm "$id".{crt,key}
}
