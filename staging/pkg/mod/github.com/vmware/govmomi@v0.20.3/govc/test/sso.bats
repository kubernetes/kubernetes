#!/usr/bin/env bats

load test_helper

@test "sso.service.ls" {
  vcsim_env

  sts=$(govc option.ls config.vpxd.sso.sts.uri | awk '{print $2}')

  # Remove credentials from URL, lookup service allows anonymous access
  GOVC_URL="$(govc env GOVC_URL)"

  run govc sso.service.ls
  assert_success

  run govc sso.service.ls -l
  assert_success

  run govc sso.service.ls -json
  assert_success

  run govc sso.service.ls -dump
  assert_success

  [ -z "$(govc sso.service.ls -t enoent)" ]

  run govc sso.service.ls -t sso:sts -U
  assert_success "$sts"

  govc sso.service.ls -t sso:sts | grep com.vmware.cis | grep -v https:
  govc sso.service.ls -t sso:sts -l | grep https:
  govc sso.service.ls -p com.vmware.cis -t sso:sts -P wsTrust -T com.vmware.cis.cs.identity.sso -l | grep wsTrust
  govc sso.service.ls -P vmomi | grep vcenterserver | grep -v https:
  govc sso.service.ls -P vmomi -l | grep https:
}
