// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x
// +build zos,s390x

package unix_test

import (
	"os"
	"testing"
	"unsafe"

	"golang.org/x/sys/unix"
)

func TestFstatfs(t *testing.T) {

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(wd)
	if err != nil {
		t.Fatal(err)
	}

	//Query Statfs_t and Statvfs_t from wd, check content
	var stat unix.Statfs_t
	err = unix.Fstatfs(int(file.Fd()), &stat)
	if err != nil {
		t.Fatal(err)
	}
	var stat_v unix.Statvfs_t
	err = unix.Fstatvfs(int(file.Fd()), &stat_v)
	if stat.Bsize != stat_v.Bsize ||
		stat.Blocks != stat_v.Blocks ||
		stat.Bfree != stat_v.Bfree ||
		stat.Bavail != stat_v.Bavail ||
		stat.Files != stat_v.Files ||
		stat.Ffree != stat_v.Ffree ||
		stat.Fsid != stat_v.Fsid ||
		stat.Namelen != stat_v.Namemax ||
		stat.Frsize != stat_v.Frsize ||
		stat.Flags != stat_v.Flag {
		t.Errorf("Mismatching fields in Statfs_t and Statvfs_t.\nStatfs_t = %+v\nStatvfs_t = %+v", stat, stat_v)
	}

	//Initialize W_Mntent, find corresponding device and check filesystem type
	var mnt_ent_buffer struct {
		header       unix.W_Mnth
		filesys_info [128]unix.W_Mntent
	}
	var buffer_size int = int(unsafe.Sizeof(mnt_ent_buffer))
	var fs_count int = -1
	for fs_count < 0 {
		fs_count, err = unix.W_Getmntent((*byte)(unsafe.Pointer(&mnt_ent_buffer)), buffer_size)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < fs_count; i++ {
			if stat.Fsid == uint64(mnt_ent_buffer.filesys_info[i].Dev) {
				correct_type := uint32(mnt_ent_buffer.filesys_info[i].Fstname[0])
				if stat.Type != correct_type {
					t.Errorf("File system type is 0x%x. Should be 0x%x instead", stat.Type, correct_type)
				}
				return
			}
		}
	}

}
