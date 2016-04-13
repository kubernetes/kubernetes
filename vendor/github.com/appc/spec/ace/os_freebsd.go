// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build freebsd

package main

import (
	"syscall"
)

func isSameFilesystem(a, b *syscall.Statfs_t) bool {
	if a.Fsid != (syscall.Fsid{}) || b.Fsid != (syscall.Fsid{}) {
		// If Fsid is not empty, we can just compare the IDs
		return a.Fsid == b.Fsid
	}
	// Fsids are zero, this happens in jails, but we can compare the rest
	return a.Fstypename == b.Fstypename &&
		a.Mntfromname == b.Mntfromname &&
		a.Mntonname == b.Mntonname
}
