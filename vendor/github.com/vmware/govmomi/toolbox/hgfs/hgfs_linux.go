/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package hgfs

import (
	"os"
	"syscall"
)

const attrMask = AttrValidAllocationSize |
	AttrValidAccessTime | AttrValidWriteTime | AttrValidCreateTime | AttrValidChangeTime |
	AttrValidSpecialPerms | AttrValidOwnerPerms | AttrValidGroupPerms | AttrValidOtherPerms | AttrValidEffectivePerms |
	AttrValidUserID | AttrValidGroupID | AttrValidFileID | AttrValidVolID

func (a *AttrV2) sysStat(info os.FileInfo) {
	sys, ok := info.Sys().(*syscall.Stat_t)

	if !ok {
		return
	}

	a.AllocationSize = uint64(sys.Blocks * 512)

	nt := func(t syscall.Timespec) uint64 {
		return uint64(t.Nano()) // TODO: this is supposed to be Windows NT system time, not needed atm
	}

	a.AccessTime = nt(sys.Atim)
	a.WriteTime = nt(sys.Mtim)
	a.CreationTime = a.WriteTime // see HgfsGetCreationTime
	a.AttrChangeTime = nt(sys.Ctim)

	a.SpecialPerms = uint8((sys.Mode & (syscall.S_ISUID | syscall.S_ISGID | syscall.S_ISVTX)) >> 9)
	a.OwnerPerms = uint8((sys.Mode & syscall.S_IRWXU) >> 6)
	a.GroupPerms = uint8((sys.Mode & syscall.S_IRWXG) >> 3)
	a.OtherPerms = uint8(sys.Mode & syscall.S_IRWXO)

	a.UserID = sys.Uid
	a.GroupID = sys.Gid
	a.HostFileID = sys.Ino
	a.VolumeID = uint32(sys.Dev)

	a.Mask |= attrMask
}
