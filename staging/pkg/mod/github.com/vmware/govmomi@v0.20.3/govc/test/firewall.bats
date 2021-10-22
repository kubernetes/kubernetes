#!/usr/bin/env bats

load test_helper

@test "firewall.ruleset.find" {
  vcsim_env -esx

  # Assumes sshServer ruleset is enabled
  run govc firewall.ruleset.find -c=false -direction inbound -port 22
  assert_success

  run govc firewall.ruleset.find -c=false -direction outbound -port 22
  if [ "$status" -eq 1 ] ; then
    # If outbound port 22 is blocked, we should be able to list disabled rules via:
    run govc firewall.ruleset.find -c=false -direction outbound -port 22 -enabled=false
    assert_success

    # find disabled should include sshClient ruleset in output
    result=$(govc firewall.ruleset.find -c=false -direction outbound -port 22 -enabled=false | grep sshClient | wc -l)
    [ $result -eq 1 ]
  fi
}
