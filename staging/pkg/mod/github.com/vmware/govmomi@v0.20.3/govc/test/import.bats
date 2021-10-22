#!/usr/bin/env bats

load test_helper

@test "import.ova" {
  esx_env

  run govc import.ova $GOVC_IMAGES/${TTYLINUX_NAME}.ova
  assert_success

  run govc vm.destroy ${TTYLINUX_NAME}
  assert_success
}

@test "import.ova with iso" {
  esx_env

  run govc import.ova $GOVC_IMAGES/${TTYLINUX_NAME}-live.ova
  assert_success

  run govc vm.destroy ${TTYLINUX_NAME}-live
  assert_success
}

@test "import.ovf" {
  esx_env

  run govc import.ovf $GOVC_IMAGES/${TTYLINUX_NAME}.ovf
  assert_success

  run govc vm.destroy ${TTYLINUX_NAME}
  assert_success

  # test w/ relative dir
  pushd $BATS_TEST_DIRNAME >/dev/null
  run govc import.ovf ./images/${TTYLINUX_NAME}.ovf
  assert_success
  popd >/dev/null

  run govc vm.destroy ${TTYLINUX_NAME}
  assert_success
}

@test "import.ovf -host.ipath" {
  esx_env # TODO: should be against vcsim

  run govc import.ovf -host.ipath="$(govc find / -type h | head -1)" "$GOVC_IMAGES/${TTYLINUX_NAME}.ovf"
  assert_success

  run govc vm.destroy "$TTYLINUX_NAME"
  assert_success
}

@test "import.ovf with name in options" {
  esx_env

  name=$(new_id)
  file=$($mktemp --tmpdir govc-test-XXXXX)
  echo "{ \"Name\": \"${name}\"}" > ${file}

  run govc import.ovf -options="${file}" $GOVC_IMAGES/${TTYLINUX_NAME}.ovf
  assert_success

  run govc vm.destroy "${name}"
  assert_success

  rm -f ${file}
}

@test "import.ovf with import.spec result" {
  esx_env

  file=$($mktemp --tmpdir govc-test-XXXXX)
  name=$(new_id)

  govc import.spec $GOVC_IMAGES/${TTYLINUX_NAME}.ovf > ${file}

  run govc import.ovf -name="${name}" -options="${file}" $GOVC_IMAGES/${TTYLINUX_NAME}.ovf
  assert_success

  run govc vm.destroy "${name}"
  assert_success
}

@test "import.ovf with name as argument" {
  esx_env

  name=$(new_id)

  run govc import.ova -name="${name}" $GOVC_IMAGES/${TTYLINUX_NAME}.ova
  assert_success

  run govc vm.destroy "${name}"
  assert_success
}

@test "import.vmdk" {
  esx_env

  name=$(new_id)

  run govc import.vmdk "$GOVC_TEST_VMDK_SRC" "$name"
  assert_success

  run govc import.vmdk "$GOVC_TEST_VMDK_SRC" "$name"
  assert_failure # exists

  run govc import.vmdk -force "$GOVC_TEST_VMDK_SRC" "$name"
  assert_success # exists, but -force was used
}
