#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit on the first error.
set -e

MNTDIR=`mktemp -d`

cleanup()
{
    # Make sure we return the right exit code
    RET=$?
    # Silently remove everything and ignore errors
    set +e
    /bin/umount $MNTDIR 2>/dev/null
    /bin/rmdir $MNTDIR 2>/dev/null
    /bin/rm block 2>/dev/null
    exit $RET
}

trap cleanup TERM EXIT

# Create 1MB device with ext2
dd if=/dev/zero of=block count=1 bs=1M
mkfs.ext2 block

# Add index.html to it
mount -o loop block $MNTDIR
echo "Hello from iSCSI" > $MNTDIR/index.html
umount $MNTDIR

rm block.tar.gz 2>/dev/null || :
tar cfz block.tar.gz block
