// Copyright 2016 The rkt Authors
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

//+build linux

package fs

import (
	"strings"
	"syscall"
)

// String returns a human readable representation of mountFlags based on which bits are set.
// E.g. for a value of syscall.MS_RDONLY|syscall.MS_BIND it will print "MS_RDONLY|MS_BIND"
func (f mountFlags) String() string {
	var s []string

	maybeAppendFlag := func(ff uintptr, desc string) {
		if uintptr(f)&ff != 0 {
			s = append(s, desc)
		}
	}

	maybeAppendFlag(syscall.MS_DIRSYNC, "MS_DIRSYNC")
	maybeAppendFlag(syscall.MS_MANDLOCK, "MS_MANDLOCK")
	maybeAppendFlag(syscall.MS_NOATIME, "MS_NOATIME")
	maybeAppendFlag(syscall.MS_NODEV, "MS_NODEV")
	maybeAppendFlag(syscall.MS_NODIRATIME, "MS_NODIRATIME")
	maybeAppendFlag(syscall.MS_NOEXEC, "MS_NOEXEC")
	maybeAppendFlag(syscall.MS_NOSUID, "MS_NOSUID")
	maybeAppendFlag(syscall.MS_RDONLY, "MS_RDONLY")
	maybeAppendFlag(syscall.MS_REC, "MS_REC")
	maybeAppendFlag(syscall.MS_RELATIME, "MS_RELATIME")
	maybeAppendFlag(syscall.MS_SILENT, "MS_SILENT")
	maybeAppendFlag(syscall.MS_STRICTATIME, "MS_STRICTATIME")
	maybeAppendFlag(syscall.MS_SYNCHRONOUS, "MS_SYNCHRONOUS")
	maybeAppendFlag(syscall.MS_REMOUNT, "MS_REMOUNT")
	maybeAppendFlag(syscall.MS_BIND, "MS_BIND")
	maybeAppendFlag(syscall.MS_SHARED, "MS_SHARED")
	maybeAppendFlag(syscall.MS_PRIVATE, "MS_PRIVATE")
	maybeAppendFlag(syscall.MS_SLAVE, "MS_SLAVE")
	maybeAppendFlag(syscall.MS_UNBINDABLE, "MS_UNBINDABLE")
	maybeAppendFlag(syscall.MS_MOVE, "MS_MOVE")

	return strings.Join(s, "|")
}
