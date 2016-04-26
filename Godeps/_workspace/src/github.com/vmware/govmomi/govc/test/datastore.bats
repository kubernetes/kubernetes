#!/usr/bin/env bats

load test_helper

upload_file() {
  file=$($mktemp --tmpdir govc-test-XXXXX)
  name=$(basename ${file})
  echo "Hello world!" > ${file}

  run govc datastore.upload "${file}" "${name}"
  assert_success

  rm -f "${file}"
  echo "${name}"
}

@test "datastore.ls" {
  name=$(upload_file)

  # Single argument
  run govc datastore.ls "${name}"
  assert_success
  [ ${#lines[@]} -eq 1 ]

  # Multiple arguments
  run govc datastore.ls "${name}" "${name}"
  assert_success
  [ ${#lines[@]} -eq 2 ]

  # Pattern argument
  run govc datastore.ls "./govc-test-*"
  assert_success
  [ ${#lines[@]} -ge 1 ]

  # Long listing
  run govc datastore.ls -l "./govc-test-*"
  assert_success
  assert_equal "13B" $(awk '{ print $1 }' <<<${output})
}

@test "datastore.rm" {
  name=$(upload_file)

  # Not found is a failure
  run govc datastore.rm "${name}.notfound"
  assert_failure
  assert_matches "govc: File .* was not found" "${output}"

  # Not found is NOT a failure with the force flag
  run govc datastore.rm -f "${name}.notfound"
  assert_success
  assert_empty "${output}"

  # Verify the file is present
  run govc datastore.ls "${name}"
  assert_success

  # Delete the file
  run govc datastore.rm "${name}"
  assert_success
  assert_empty "${output}"

  # Verify the file is gone
  run govc datastore.ls "${name}"
  assert_failure
}

@test "datastore.info" {
  run govc datastore.info enoent
  assert_failure

  run govc datastore.info
  assert_success
  [ ${#lines[@]} -gt 1 ]
}
