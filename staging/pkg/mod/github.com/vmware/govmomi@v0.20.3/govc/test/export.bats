#!/usr/bin/env bats

load test_helper

# ovftool -tt=ovf --noSSLVerify --skipManifestCheck "vi://$GOVC_URL/$GOVC_VM" .
@test "export.ovf" {
  esx_env

  id=$(new_ttylinux_vm)
  dir=$BATS_TMPDIR/$id-export

  run govc export.ovf -vm "$id" "$dir"
  assert_success

  run ls "$dir/$id/$id-disk-0.vmdk" "$dir/$id/$id.ovf"
  assert_success

  if [ -e "$dir/$id/$id.mf" ] ; then
    flunk ".mf was created"
  fi

  run govc export.ovf -vm "$id" "$dir"
  assert_failure

  run govc export.ovf -i -f -sha 256 -vm "$id" "$dir"
  assert_success

  run ls "$dir/$id/$id.mf"
  assert_success

  # make it an ova
  (cd "$dir/$id" && tar -cf "../$id.ova" .)

  # ovftool --noSSLVerify --skipManifestCheck --name="$GOVC_VM-import" "$GOVC_VM/$GOVC_VM.ovf" "vi://$GOVC_URL"
  run govc import.ova -name "${id}-import" "$dir/$id.ova"
  assert_success

  rm -rf "$dir"
}
