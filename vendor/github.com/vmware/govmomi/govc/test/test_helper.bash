# set the following variables only if they've not been set
GOVC_TEST_URL=${GOVC_TEST_URL-"https://root:vagrant@localhost:18443/sdk"}
export GOVC_URL=$GOVC_TEST_URL
export GOVC_DATASTORE=${GOVC_DATASTORE-datastore1}
export GOVC_NETWORK=${GOVC_NETWORK-"VM Network"}

export GOVC_INSECURE=true
export GOVC_PERSIST_SESSION=false
unset GOVC_DEBUG
unset GOVC_TLS_KNOWN_HOSTS
unset GOVC_DATACENTER
unset GOVC_HOST
unset GOVC_USERNAME
unset GOVC_PASSWORD

if [ -z "$BATS_TEST_DIRNAME" ]; then
  BATS_TEST_DIRNAME=$(dirname ${BASH_SOURCE})
fi

# gnu core utils
readlink=$(type -p greadlink readlink | head -1)
xargs=$(type -p gxargs xargs | head -1)
mktemp=$(type -p gmktemp mktemp | head -1)

BATS_TEST_DIRNAME=$($readlink -nf $BATS_TEST_DIRNAME)

GOVC_IMAGES=$BATS_TEST_DIRNAME/images
TTYLINUX_NAME=ttylinux-pc_i486-16.1

GOVC_TEST_VMDK_SRC=$GOVC_IMAGES/${TTYLINUX_NAME}-disk1.vmdk
GOVC_TEST_VMDK=govc-images/$(basename $GOVC_TEST_VMDK_SRC)

GOVC_TEST_ISO_SRC=$GOVC_IMAGES/${TTYLINUX_NAME}.iso
GOVC_TEST_ISO=govc-images/$(basename $GOVC_TEST_ISO_SRC)

GOVC_TEST_IMG_SRC=$GOVC_IMAGES/floppybird.img
GOVC_TEST_IMG=govc-images/$(basename $GOVC_TEST_IMG_SRC)

PATH="$(dirname $BATS_TEST_DIRNAME):$PATH"

teardown() {
  govc ls vm | grep govc-test- | $xargs -r govc vm.destroy
  govc datastore.ls | grep govc-test- | awk '{print ($NF)}' | $xargs -n1 -r govc datastore.rm
  govc ls "host/*/Resources/govc-test-*" | $xargs -r govc pool.destroy
}

new_id() {
  echo "govc-test-$(uuidgen)"
}

import_ttylinux_vmdk() {
  govc datastore.mkdir -p govc-images
  govc datastore.ls "$GOVC_TEST_VMDK" >/dev/null 2>&1 || \
    govc import.vmdk "$GOVC_TEST_VMDK_SRC" govc-images > /dev/null
}

datastore_upload() {
  src=$1
  dst=govc-images/$(basename $src)

  govc datastore.mkdir -p govc-images
  govc datastore.ls "$dst" >/dev/null 2>&1 || \
    govc datastore.upload "$src" "$dst" > /dev/null
}

upload_img() {
  datastore_upload $GOVC_TEST_IMG_SRC
}

upload_iso() {
  datastore_upload $GOVC_TEST_ISO_SRC
}

new_ttylinux_vm() {
  import_ttylinux_vmdk # TODO: make this part of vagrant provision
  id=$(new_id)
  govc vm.create -m 32 -disk $GOVC_TEST_VMDK -disk.controller ide -on=false $id
  echo $id
}

new_empty_vm() {
  id=$(new_id)
  govc vm.create -on=false $id
  echo $id
}

vm_power_state() {
  govc vm.info "$1" | grep "Power state:" | awk -F: '{print $2}' | collapse_ws
}

vm_mac() {
  govc device.info -vm "$1" ethernet-0 | grep "MAC Address" | awk '{print $NF}'
}

# exports an environment for using vcsim if running, otherwise skips the calling test.
vcsim_env() {
  if [ "$(uname)" == "Darwin" ]; then
    PATH="/Applications/VMware Fusion.app/Contents/Library:$PATH"
  fi

  if [ "$(vmrun list | grep $BATS_TEST_DIRNAME/vcsim | wc -l)" -eq 1 ]; then
    export GOVC_URL=https://root:vmware@localhost:16443/sdk \
           GOVC_DATACENTER=DC0 \
           GOVC_DATASTORE=GlobalDS_0 \
           GOVC_HOST=/DC0/host/DC0_C0/DC0_C0_H0 \
           GOVC_RESOURCE_POOL=/DC0/host/DC0_C0/Resources \
           GOVC_NETWORK=/DC0/network/DC0_DVPG0
  else
    skip "requires vcsim"
  fi
}

# remove username/password from $GOVC_URL and set $GOVC_{USERNAME,PASSWORD}
govc_url_to_vars() {
  GOVC_USERNAME="$(govc env GOVC_USERNAME)"
  GOVC_PASSWORD="$(govc env GOVC_PASSWORD)"
  GOVC_URL="$(govc env GOVC_URL)"
  export GOVC_URL GOVC_USERNAME GOVC_PASSWORD

  # double check that we removed user/pass
  grep -q -v @ <<<"$GOVC_URL"
}

quit_vnc() {
  if [ "$(uname)" = "Darwin" ]; then
    osascript <<EOF
tell application "Screen Sharing"
   quit
end tell
EOF
  fi
}

open_vnc() {
  url=$1
  echo "open $url"

  if [ "$(uname)" = "Darwin" ]; then
    open $url
  fi
}

# collapse spaces, for example testing against Go's tabwriter output
collapse_ws() {
  local line
  if [ $# -eq 0 ]; then line="$(cat -)"
  else line="$@"
  fi
  echo "$line" | tr -s ' ' | sed -e 's/^ //'
}

# the following helpers are borrowed from the test_helper.bash in https://github.com/sstephenson/rbenv

flunk() {
  { if [ "$#" -eq 0 ]; then cat -
    else echo "$@"
    fi
  } >&2
  return 1
}

assert_success() {
  if [ "$status" -ne 0 ]; then
    flunk "command failed with exit status $status: $output"
  elif [ "$#" -gt 0 ]; then
    assert_output "$1"
  fi
}

assert_failure() {
  if [ "$status" -ne 1 ]; then
    flunk $(printf "expected failed exit status=1, got status=%d" $status)
  elif [ "$#" -gt 0 ]; then
    assert_output "$1"
  fi
}

assert_equal() {
  if [ "$1" != "$2" ]; then
    { echo "expected: $1"
      echo "actual:   $2"
    } | flunk
  fi
}

assert_output() {
  local expected
  if [ $# -eq 0 ]; then expected="$(cat -)"
  else expected="$1"
  fi
  assert_equal "$expected" "$output"
}

assert_matches() {
  local pattern="${1}"
  local actual="${2}"

  if [ $# -eq 1 ]; then
    actual="$(cat -)"
  fi

  if ! grep -q "${pattern}" <<<"${actual}"; then
    { echo "pattern: ${pattern}"
      echo "actual:  ${actual}"
    } | flunk
  fi
}

assert_empty() {
  local actual="${1}"

  if [ $# -eq 0 ]; then
    actual="$(cat -)"
  fi

  if [ -n "${actual}" ]; then
    { echo "actual: ${actual}"
    } | flunk
  fi
}

assert_line() {
  if [ "$1" -ge 0 ] 2>/dev/null; then
    assert_equal "$2" "$(collapse_ws ${lines[$1]})"
  else
    local line
    for line in "${lines[@]}"; do
      if [ "$(collapse_ws $line)" = "$1" ]; then return 0; fi
    done
    flunk "expected line \`$1'"
  fi
}

refute_line() {
  if [ "$1" -ge 0 ] 2>/dev/null; then
    local num_lines="${#lines[@]}"
    if [ "$1" -lt "$num_lines" ]; then
      flunk "output has $num_lines lines"
    fi
  else
    local line
    for line in "${lines[@]}"; do
      if [ "$line" = "$1" ]; then
        flunk "expected to not find line \`$line'"
      fi
    done
  fi
}

assert() {
  if ! "$@"; then
    flunk "failed: $@"
  fi
}
