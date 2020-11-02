// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import (
	"sync"
	"syscall"
	"unsafe"
)

// See version list in https://github.com/DragonFlyBSD/DragonFlyBSD/blob/master/sys/sys/param.h
var (
	osreldateOnce sync.Once
	osreldate     uint32
)

// First __DragonFly_version after September 2019 ABI changes
// http://lists.dragonflybsd.org/pipermail/users/2019-September/358280.html
const _dragonflyABIChangeVersion = 500705

func probeProtocolStack() int {
	osreldateOnce.Do(func() { osreldate, _ = syscall.SysctlUint32("kern.osreldate") })
	var p uintptr
	if int(unsafe.Sizeof(p)) == 8 && osreldate >= _dragonflyABIChangeVersion {
		return int(unsafe.Sizeof(p))
	}
	// 64-bit Dragonfly before the September 2019 ABI changes still requires
	// 32-bit aligned access to network subsystem.
	return 4
}
