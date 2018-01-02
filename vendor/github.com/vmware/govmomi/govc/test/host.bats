#!/usr/bin/env bats

load test_helper

@test "host info esx" {
  run govc host.info
  assert_success
  grep -q Manufacturer: <<<$output

  run govc host.info -host enoent
  assert_failure "govc: host 'enoent' not found"

  for opt in dns ip ipath uuid
  do
    run govc host.info "-host.$opt" enoent
    assert_failure "govc: no such host"
  done

  # avoid hardcoding the esxbox hostname
  local name=$(govc ls '/*/host/*' | grep -v Resources)

  run govc host.info -host $name
  assert_success
  grep -q Manufacturer: <<<$output

  run govc host.info -host ${name##*/}
  assert_success
  grep -q Manufacturer: <<<$output

  run govc host.info -host.ipath $name
  assert_success

  run govc host.info -host.dns $(basename $(dirname $name))
  assert_success

  uuid=$(govc host.info -json | jq -r .HostSystems[].Hardware.SystemInfo.Uuid)
  run govc host.info -host.uuid $uuid
  assert_success

  run govc host.info "*"
  assert_success
}

@test "host info vc" {
  vcsim_env

  run govc host.info
  assert_success
  grep -q Manufacturer: <<<$output

  run govc host.info -host enoent
  assert_failure "govc: host 'enoent' not found"

  for opt in dns ip ipath uuid
  do
    run govc host.info "-host.$opt" enoent
    assert_failure "govc: no such host"
  done

  local name=$GOVC_HOST

  unset GOVC_HOST
  run govc host.info
  assert_failure "govc: default host resolves to multiple instances, please specify"

  run govc host.info -host $name
  assert_success
  grep -q Manufacturer: <<<$output

  run govc host.info -host.ipath $name
  assert_success

  run govc host.info -host.dns $(basename $name)
  assert_success

  uuid=$(govc host.info -host $name -json | jq -r .HostSystems[].Hardware.SystemInfo.Uuid)
  run govc host.info -host.uuid $uuid
  assert_success
}

@test "host.vnic.info" {
  run govc host.vnic.info
  assert_success
}

@test "host.vswitch.info" {
  run govc host.vswitch.info
  assert_success

  run govc host.vswitch.info -json
  assert_success
}

@test "host.portgroup.info" {
  run govc host.portgroup.info
  assert_success

  run govc host.portgroup.info -json
  assert_success
}

@test "host.options" {
    run govc host.option.ls Config.HostAgent.plugins.solo.enableMob
    assert_success

    run govc host.option.ls Config.HostAgent.plugins.
    assert_success

    run govc host.option.ls -json Config.HostAgent.plugins.
    assert_success

    run govc host.option.ls Config.HostAgent.plugins.solo.ENOENT
    assert_failure
}

@test "host.service" {
    run govc host.service.ls
    assert_success

    run govc host.service.ls -json
    assert_success

    run govc host.service status TSM-SSH
    assert_success
}

@test "host.cert.info" {
  run govc host.cert.info
  assert_success

  run govc host.cert.info -json
  assert_success

  expires=$(govc host.cert.info -json | jq -r .NotAfter)
  about_expires=$(govc about.cert -json | jq -r .NotAfter)
  assert_equal "$expires" "$about_expires"
}

@test "host.cert.csr" {
  #   Requested Extensions:
  #       X509v3 Subject Alternative Name:
  #       IP Address:...
  result=$(govc host.cert.csr -ip | openssl req -text -noout)
  assert_matches "IP Address:" "$result"
  ! assert_matches "DNS:" "$result"

  #   Requested Extensions:
  #       X509v3 Subject Alternative Name:
  #       DNS:...
  result=$(govc host.cert.csr | openssl req -text -noout)
  ! assert_matches "IP Address:" "$result"
  assert_matches "DNS:" "$result"
}

@test "host.cert.import" {
  issuer=$(govc host.cert.info -json | jq -r .Issuer)
  expires=$(govc host.cert.info -json | jq -r .NotAfter)

  # only mess with the cert if its already been signed by our test CA
  if [[ "$issuer" != CN=govc-ca,* ]] ; then
    skip "host cert not signed by govc-ca"
  fi

  govc host.cert.csr -ip | ./host_cert_sign.sh | govc host.cert.import
  expires2=$(govc host.cert.info -json | jq -r .NotAfter)

  # cert expiration should have changed
  [ "$expires" != "$expires2" ]

  # verify hostd is using the new cert too
  expires=$(govc about.cert -json | jq -r .NotAfter)
  assert_equal "$expires" "$expires2"

  # our cert is not trusted against the system CA list
  status=$(govc about.cert | grep Status:)
  assert_matches ERROR "$status"

  # with our CA trusted, the cert should be too
  status=$(govc about.cert -tls-ca-certs ./govc_ca.pem | grep Status:)
  assert_matches good "$status"
}

@test "host.date.info" {
  run govc host.date.info
  assert_success

  run govc host.date.info -json
  assert_success
}
