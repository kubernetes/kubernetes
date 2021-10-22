#!/bin/bash -e

# This test is not run via bats

# See also: datastore.bats@test "datastore.disk"

export GOVC_TEST_URL=$GOVC_URL

. "$(dirname "$0")"/test_helper.bash

echo -n "checking datastore type..."
type=$(govc object.collect -s "datastore/$GOVC_DATASTORE" summary.type)
echo "$type"

if [ "$type" = "vsan" ] ; then
  echo -n "checking for orphan objects..."
  objs=($(govc datastore.vsan.dom.ls -o))
  echo "${#objs[@]}"

  if [ "${#objs[@]}" -ne "0" ] ; then
    govc datastore.vsan.dom.rm "${objs[@]}"
  fi
fi

dir=govc-test-dfm

echo "uploading plain file..."
cal | govc datastore.upload - $dir/cal.txt
echo "removing plain file..."
govc datastore.rm $dir/cal.txt

scratch=$dir/govc-test-scratch/govc-test-scratch.vmdk

govc datastore.mkdir -p "$(dirname $scratch)"

echo "creating disk $scratch..."
govc datastore.disk.create -size 1M $scratch

id=$(new_id)

echo "creating $id VM with disk linked to $scratch..."
govc vm.create -on=false -link -disk $scratch "$id"
info=$(govc device.info -vm "$id" disk-*)
echo "$info"

disk="$(grep Name: <<<"$info" | awk '{print $2}')"
vmdk="$id/$id.vmdk"

echo "removing $disk device but keeping the .vmdk backing file..."
govc device.remove -vm "$id" -keep "$disk"

echo -n "checking delta disk ddb.deletable..."
govc datastore.download "$vmdk" - | grep -q -v ddb.deletable
echo "yes"

echo -n "checking scratch disk ddb.deletable..."
govc datastore.download "$scratch" - | grep ddb.deletable | grep -q false
echo "no"

echo "removing $vmdk"
govc datastore.rm "$vmdk"

echo -n "checking that rm $scratch fails..."
govc datastore.rm "$scratch" 2>/dev/null || echo "yes"

echo -n "checking that rm -f $scratch deletes..."
govc datastore.rm -f "$scratch" && echo "yes"

echo "removing disk Directory via FileManager..."
govc datastore.mkdir -p "$(dirname $scratch)"
govc datastore.disk.create -size 1M $scratch
govc datastore.rm "$(dirname $scratch)"

echo -n "checking for remaining files..."
govc datastore.ls -p -R $dir

teardown

status=0

if [ "$type" = "vsan" ] ; then
  echo -n "checking for leaked objects..."
  objs=($(govc datastore.vsan.dom.ls -l -o | awk '{print $3}'))
  echo "${#objs[@]}"

  if [ "${#objs[@]}" -ne "0" ] ; then
    printf "%s\n" "${objs[@]}"
    status=1
  else
    # this is expected to leak on vSAN currently
    echo -n "checking if FileManager.Delete still leaks..."
    govc datastore.mkdir -p "$(dirname $scratch)"
    govc datastore.disk.create -size 1M $scratch
    # '-t=false' forces use of FileManager instead of VirtualDiskManager
    govc datastore.rm -t=false $scratch
    govc datastore.rm $dir

    govc datastore.vsan.dom.ls -o | xargs -r govc datastore.vsan.dom.rm -v
  fi
fi

exit $status
