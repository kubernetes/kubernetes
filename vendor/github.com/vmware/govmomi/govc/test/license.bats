#!/usr/bin/env bats

load test_helper

# These tests should only run against a server running an evaluation license.
verify_evaluation() {
  if [ "$(govc license.ls -json | jq -r .[0].EditionKey)" != "eval" ]; then
    skip "requires evaluation license"
  fi
}

get_key() {
  jq ".[] | select(.LicenseKey == \"$1\")"
}

get_property() {
  jq -r ".Properties[] | select(.Key == \"$1\") | .Value"
}

@test "license.add" {
  verify_evaluation

  run govc license.add -json 00000-00000-00000-00000-00001 00000-00000-00000-00000-00002
  assert_success

  # Expect to see an entry for both the first and the second key
  assert_equal "License is not valid for this product" $(get_key 00000-00000-00000-00000-00001 <<<${output} | get_property diagnostic)
  assert_equal "License is not valid for this product" $(get_key 00000-00000-00000-00000-00002 <<<${output} | get_property diagnostic)
}

@test "license.remove" {
  verify_evaluation

  run govc license.remove -json 00000-00000-00000-00000-00001
  assert_success
}

@test "license.ls" {
  verify_evaluation

  run govc license.ls -json
  assert_success

  # Expect the test instance to run in evaluation mode
  assert_equal "Evaluation Mode" $(get_key 00000-00000-00000-00000-00000 <<<$output | jq -r ".Name")
}

@test "license.decode" {
  verify_evaluation

  key=00000-00000-00000-00000-00000
  assert_equal "eval" $(govc license.decode $key | grep $key | awk '{print $2}')
}
