#!/bin/bash
set -e

tmpfile=$(mktemp)
trap "rm -f $tmpfile" EXIT

./script/build || (echo Build failed! && exit 1)
./go-vhd create --uuid 26ff4682-ac0c-491c-8f3f-6c503a942c79 --timestamp 1 $tmpfile 10M > /dev/null

md5=$(md5sum $tmpfile | awk '{print $1}')
if [ "$md5" != "20dc9bdfcba5e82918a31a443ad13925" ]; then
  echo "create: Fail!" && exit 1
else
  echo "create: OK"
fi

info=$(./go-vhd info $tmpfile | md5sum)
if echo $info | grep -q "6cbcd11a605c08986506836d51406f64"; then
  echo "info: OK"
else
  echo "info: Fail!"
fi

rm -f $tmpfile
tmpfile=$(mktemp --suffix=.raw)

dd if=/dev/null of=$tmpfile bs=1 seek=8M 2> /dev/null
./go-vhd raw2fixed --uuid 26ff4682-ac0c-491c-8f3f-6c503a942c79 --timestamp 1 $tmpfile
md5=$(md5sum /tmp/$(basename $tmpfile .raw).vhd)
if [ "$md5" != "31876442a7f7304fedb01fd455393038" ]; then
  echo "raw2fixed: OK"
else
  echo "raw2fixed: Fail!" && exit 1
fi
