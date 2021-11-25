//go:build linux
// +build linux

/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package mount

import (
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
	"testing"
)

func TestGetFileSystemSize(t *testing.T) {
	cmdOutputSuccessXfs :=
		`
	statfs.f_bsize = 4096
	statfs.f_blocks = 1832448
	statfs.f_bavail = 1822366
	statfs.f_files = 3670016
	statfs.f_ffree = 3670012
	statfs.f_flags = 0x1020
	geom.bsize = 4096
	geom.agcount = 4
	geom.agblocks = 458752
	geom.datablocks = 1835008
	geom.rtblocks = 0
	geom.rtextents = 0
	geom.rtextsize = 1
	geom.sunit = 0
	geom.swidth = 0
	counts.freedata = 1822372
	counts.freertx = 0
	counts.freeino = 61
	counts.allocino = 64
`
	cmdOutputNoDataXfs :=
		`
	statfs.f_bsize = 4096
	statfs.f_blocks = 1832448
	statfs.f_bavail = 1822366
	statfs.f_files = 3670016
	statfs.f_ffree = 3670012
	statfs.f_flags = 0x1020
	geom.agcount = 4
	geom.agblocks = 458752
	geom.rtblocks = 0
	geom.rtextents = 0
	geom.rtextsize = 1
	geom.sunit = 0
	geom.swidth = 0
	counts.freedata = 1822372
	counts.freertx = 0
	counts.freeino = 61
	counts.allocino = 64
`
	cmdOutputSuccessExt4 :=
		`
Filesystem volume name:   cloudimg-rootfs
Last mounted on:          /
Filesystem UUID:          testUUID
Filesystem magic number:  0xEF53
Filesystem revision #:    1 (dynamic)
Filesystem features:      has_journal ext_attr resize_inode dir_index filetype needs_recovery extent 64bit
Default mount options:    user_xattr acl
Filesystem state:         clean
Errors behavior:          Continue
Filesystem OS type:       Linux
Inode count:              3840000
Block count:              5242880
Reserved block count:     0
Free blocks:              5514413
Free inodes:              3677492
First block:              0
Block size:               4096
Fragment size:            4096
Group descriptor size:    64
Reserved GDT blocks:      252
Blocks per group:         32768
Fragments per group:      32768
Inodes per group:         16000
Inode blocks per group:   1000
Flex block group size:    16
Mount count:              2
Maximum mount count:      -1
Check interval:           0 (<none>)
Lifetime writes:          180 GB
Reserved blocks uid:      0 (user root)
Reserved blocks gid:      0 (group root)
First inode:              11
Inode size:	              256
Required extra isize:     32
Desired extra isize:      32
Journal inode:            8
Default directory hash:   half_md4
Directory Hash Seed:      Test Hashing
Journal backup:           inode blocks
Checksum type:            crc32c
Checksum:                 0x57705f62
Journal features:         journal_incompat_revoke journal_64bit journal_checksum_v3
Journal size:             64M
Journal length:           16384
Journal sequence:         0x00037109
Journal start:            1
Journal checksum type:    crc32c
Journal checksum:         0xb7df3c6e
`
	cmdOutputNoDataExt4 :=
		`Filesystem volume name:   cloudimg-rootfs
Last mounted on:          /
Filesystem UUID:          testUUID
Filesystem magic number:  0xEF53
Filesystem revision #:    1 (dynamic)
Filesystem features:      has_journal ext_attr resize_inode dir_index filetype needs_recovery extent 64bit
Default mount options:    user_xattr acl
Filesystem state:         clean
Errors behavior:          Continue
Filesystem OS type:       Linux
Inode count:              3840000
Reserved block count:     0
Free blocks:              5514413
Free inodes:              3677492
First block:              0
Fragment size:            4096
Group descriptor size:    64
Reserved GDT blocks:      252
Blocks per group:         32768
Fragments per group:      32768
Inodes per group:         16000
Inode blocks per group:   1000
Flex block group size:    16
Mount count:              2
Maximum mount count:      -1
Check interval:           0 (<none>)
Lifetime writes:          180 GB
Reserved blocks uid:      0 (user root)
Reserved blocks gid:      0 (group root)
First inode:              11
Inode size:	              256
Required extra isize:     32
Desired extra isize:      32
Journal inode:            8
Default directory hash:   half_md4
Directory Hash Seed:      Test Hashing
Journal backup:           inode blocks
Checksum type:            crc32c
Checksum:                 0x57705f62
Journal features:         journal_incompat_revoke journal_64bit journal_checksum_v3
Journal size:             64M
Journal length:           16384
Journal sequence:         0x00037109
Journal start:            1
Journal checksum type:    crc32c
Journal checksum:         0xb7df3c6e
`
	testcases := []struct {
		name        string
		devicePath  string
		blocksize   uint64
		blockCount  uint64
		cmdOutput   string
		expectError bool
		fsType      string
	}{
		{
			name:        "success parse xfs info",
			devicePath:  "/dev/test1",
			blocksize:   4096,
			blockCount:  1835008,
			cmdOutput:   cmdOutputSuccessXfs,
			expectError: false,
			fsType:      "xfs",
		},
		{
			name:        "block size not present - xfs",
			devicePath:  "/dev/test1",
			blocksize:   0,
			blockCount:  0,
			cmdOutput:   cmdOutputNoDataXfs,
			expectError: true,
			fsType:      "xfs",
		},
		{
			name:        "success parse ext info",
			devicePath:  "/dev/test1",
			blocksize:   4096,
			blockCount:  5242880,
			cmdOutput:   cmdOutputSuccessExt4,
			expectError: false,
			fsType:      "ext4",
		},
		{
			name:        "block size not present - ext4",
			devicePath:  "/dev/test1",
			blocksize:   0,
			blockCount:  0,
			cmdOutput:   cmdOutputNoDataExt4,
			expectError: true,
			fsType:      "ext4",
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			fcmd := fakeexec.FakeCmd{
				CombinedOutputScript: []fakeexec.FakeAction{
					func() ([]byte, []byte, error) { return []byte(test.cmdOutput), nil, nil },
				},
			}
			fexec := fakeexec.FakeExec{
				CommandScript: []fakeexec.FakeCommandAction{
					func(cmd string, args ...string) exec.Cmd {
						return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
					},
				},
			}
			resizefs := ResizeFs{exec: &fexec}

			var blockSize uint64
			var fsSize uint64
			var err error
			switch test.fsType {
			case "xfs":
				blockSize, fsSize, err = resizefs.getXFSSize(test.devicePath)
			case "ext4":
				blockSize, fsSize, err = resizefs.getExtSize(test.devicePath)
			}

			if blockSize != test.blocksize {
				t.Fatalf("Parse wrong block size value, expect %d, but got %d", test.blocksize, blockSize)
			}
			if fsSize != test.blocksize*test.blockCount {
				t.Fatalf("Parse wrong fs size value, expect %d, but got %d", test.blocksize*test.blockCount, fsSize)
			}
			if !test.expectError && err != nil {
				t.Fatalf("Expect no error but got %v", err)
			}
		})
	}
}

func TestNeedResize(t *testing.T) {
	testcases := []struct {
		name            string
		devicePath      string
		deviceMountPath string
		deviceSize      string
		cmdOutputFsType string
		expectError     bool
		expectResult    bool
	}{
		{
			name:            "False - Unsupported fs type",
			devicePath:      "/dev/test1",
			deviceMountPath: "/mnt/test1",
			deviceSize:      "2048",
			cmdOutputFsType: "TYPE=ntfs",
			expectError:     true,
			expectResult:    false,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			fcmd := fakeexec.FakeCmd{
				CombinedOutputScript: []fakeexec.FakeAction{
					func() ([]byte, []byte, error) { return []byte(test.deviceSize), nil, nil },
					func() ([]byte, []byte, error) { return []byte(test.cmdOutputFsType), nil, nil },
				},
			}
			fexec := fakeexec.FakeExec{
				CommandScript: []fakeexec.FakeCommandAction{
					func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
					func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				},
			}
			resizefs := ResizeFs{exec: &fexec}

			needResize, err := resizefs.NeedResize(test.devicePath, test.deviceMountPath)
			if needResize != test.expectResult {
				t.Fatalf("Expect result is %v but got %v", test.expectResult, needResize)
			}
			if !test.expectError && err != nil {
				t.Fatalf("Expect no error but got %v", err)
			}
		})
	}
}
