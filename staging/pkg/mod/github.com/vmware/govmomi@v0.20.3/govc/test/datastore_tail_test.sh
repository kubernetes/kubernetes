#!/bin/bash -xe

# This test is not run via bats as the bats pipeline hangs when we background a process

. "$(dirname "$0")"/test_helper.bash

name=$(new_id)
n=16
tmp=$(mktemp --tmpdir "${name}-XXXXX")

echo -n | govc datastore.upload - "$name"
govc datastore.tail -f "$name" > "$tmp" &
pid=$!

sleep 1
yes | dd bs=${n}K count=1 2>/dev/null | govc datastore.upload - "$name"
sleep 2

# stops following when the file has gone away
govc datastore.mv "$name" "${name}.old"
wait $pid

govc datastore.download "${name}.old" - | cmp "$tmp" -

rm "$tmp"
teardown
