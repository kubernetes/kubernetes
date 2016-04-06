#!/bin/bash
#
# Cleanup any artifacts created by govc
#

. $(dirname $0)/test_helper.bash

teardown

datastore_rm() {
  name=$1
  govc datastore.rm $name 2> /dev/null
}

datastore_rm $GOVC_TEST_IMG
datastore_rm $GOVC_TEST_ISO
datastore_rm $GOVC_TEST_VMDK
datastore_rm $(echo $GOVC_TEST_VMDK | sed 's/.vmdk/-flat.vmdk/')

# Recursively destroy all resource pools created by the test suite
govc ls host/*/Resources/govc-test-* | \
  xargs -rt govc pool.destroy -r

govc datastore.ls
