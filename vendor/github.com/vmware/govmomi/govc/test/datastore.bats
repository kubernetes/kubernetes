#!/usr/bin/env bats

load test_helper

upload_file() {
  name=$(new_id)

  echo "Hello world" | govc datastore.upload - "$name"
  assert_success

  echo "$name"
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
  assert_equal "12B" $(awk '{ print $1 }' <<<${output})
}

@test "datastore.ls-R" {
  dir=$(new_id)

  run govc datastore.mkdir "$dir"
  assert_success

  for name in one two three ; do
    echo "$name world" | govc datastore.upload - "$dir/file-$name"
    run govc datastore.mkdir -p "$dir/dir-$name/subdir-$name"
    run govc datastore.mkdir -p "$dir/dir-$name/.hidden"
    assert_success
    echo "$name world" | govc datastore.upload - "$dir/dir-$name/.hidden/other-$name"
    echo "$name world" | govc datastore.upload - "$dir/dir-$name/other-$name"
    echo "$name world" | govc datastore.upload - "$dir/dir-$name/subdir-$name/last-$name"
  done

  # without -R
  json=$(govc datastore.ls -json -l -p "$dir")
  result=$(jq -r .[].File[].Path <<<"$json" | wc -l)
  [ "$result" -eq 6 ]

  result=$(jq -r .[].FolderPath <<<"$json" | wc -l)
  [ "$result" -eq 1 ]

  # with -R
  json=$(govc datastore.ls -json -l -p -R "$dir")
  result=$(jq -r .[].File[].Path <<<"$json" | wc -l)
  [ "$result" -eq 15 ]

  result=$(jq -r .[].FolderPath <<<"$json" | wc -l)
  [ "$result" -eq 7 ]

  # with -R -a
  json=$(govc datastore.ls -json -l -p -R -a "$dir")
  result=$(jq -r .[].File[].Path <<<"$json" | wc -l)
  [ "$result" -eq 21 ]

  result=$(jq -r .[].FolderPath <<<"$json" | wc -l)
  [ "$result" -eq 10 ]
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


@test "datastore.mkdir" {
  name=$(new_id)

  # Not supported datastore type is a failure
  run govc datastore.mkdir -namespace "notfound"
  assert_failure
  assert_matches "govc: ServerFaultCode: .*" "${output}"

  run govc datastore.mkdir "${name}"
  assert_success
  assert_empty "${output}"

  # Verify the dir is present
  run govc datastore.ls "${name}"
  assert_success

  # Delete the dir on an unsupported datastore type is a failure
  run govc datastore.rm -namespace "${name}"
  assert_failure
  assert_matches "govc: ServerFaultCode: .*" "${output}"

  # Delete the dir
  run govc datastore.rm "${name}"
  assert_success
  assert_empty "${output}"

  # Verify the dir is gone
  run govc datastore.ls "${name}"
  assert_failure
}

@test "datastore.download" {
  name=$(upload_file)
  run govc datastore.download "$name" -
  assert_success
  assert_output "Hello world"

  run govc datastore.download "$name" "$TMPDIR/$name"
  assert_success
  run cat "$TMPDIR/$name"
  assert_output "Hello world"
  rm "$TMPDIR/$name"
}

@test "datastore.upload" {
  name=$(new_id)
  echo -n "Hello world" | govc datastore.upload - "$name"

  run govc datastore.download "$name" -
  assert_success
  assert_output "Hello world"
}

@test "datastore.tail" {
  run govc datastore.tail "enoent/enoent.log"
  assert_failure

  id=$(new_id)
  govc vm.create "$id"
  govc vm.power -off "$id"

  # test with .log (> bufSize) and .vmx (< bufSize)
  for file in "$id/vmware.log" "$id/$id.vmx" ; do
    log=$(govc datastore.download "$file" -)

    for n in 0 1 5 10 123 456 7890 ; do
      expect=$(tail -n $n <<<"$log")

      run govc datastore.tail -n $n "$file"
      assert_output "$expect"

      expect=$(tail -c $n <<<"$log")

      run govc datastore.tail -c $n "$file"
      assert_output "$expect"
    done
  done
}

@test "datastore.disk" {
  id=$(new_id)
  vmdk="$id/$id.vmdk"

  run govc datastore.mkdir "$id"
  assert_success

  run govc datastore.disk.create "$vmdk"
  assert_success

  run govc datastore.disk.info "$vmdk"
  assert_success

  run govc datastore.rm "$vmdk"
  assert_success

  run govc datastore.mkdir -p "$id"
  assert_success

  run govc datastore.disk.create "$vmdk"
  assert_success

  id=$(new_id)
  run govc vm.create -on=false -link -disk "$vmdk" "$id"
  assert_success

  run govc datastore.disk.info -d "$vmdk"
  assert_success

  run govc datastore.disk.info -p=false "$vmdk"
  assert_success

  run govc datastore.disk.info -c "$vmdk"
  assert_success

  run govc datastore.disk.info -json "$vmdk"
  assert_success

  # should fail due to: ddb.deletable=false
  run govc datastore.rm "$vmdk"
  assert_failure

  run govc datastore.rm -f "$vmdk"
  assert_success

  # one more time, but rm the directory w/o -f
  run govc datastore.mkdir -p "$id"
  assert_success

  run govc datastore.disk.create "$vmdk"
  assert_success

  id=$(new_id)
  run govc vm.create -on=false -link -disk "$vmdk" "$id"
  assert_success

  run govc datastore.rm "$(dirname "$vmdk")"
  assert_success
}

@test "datastore.disk.info" {
  import_ttylinux_vmdk

  run govc datastore.disk.info
  assert_failure

  run govc datastore.disk.info enoent
  assert_failure

  run govc datastore.disk.info "$GOVC_TEST_VMDK"
  assert_success

  run govc datastore.disk.info -d "$GOVC_TEST_VMDK"
  assert_success

  run govc datastore.disk.info -c "$GOVC_TEST_VMDK"
  assert_success
}
