#!/bin/bash
PKG=$@
PKG_OUT=${PKG//\//_}
set -o pipefail;go test "${goflags[@]:+${goflags[@]}}" \
  ${KUBE_RACE} \
  ${KUBE_TIMEOUT} \
  -cover -covermode="${KUBE_COVERMODE}" \
  -coverprofile="${cover_report_dir}/${PKG}/${cover_profile}" \
  "${KUBE_GO_PACKAGE}/${PKG}" \
  ${testargs[@]:+${testargs[@]}} \
| tee ${junit_filename_prefix:+"${junit_filename_prefix}-${PKG_OUT}.stdout"} \
| grep "${go_test_grep_pattern}"
