// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris
// +build !386,!arm,!mips,!mipsle

package procfs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/sys/unix"
)

func TestProcMaps(t *testing.T) {
	tsts64 := []*ProcMap{
		{
			StartAddr: 0x55680ae1e000,
			EndAddr:   0x55680ae20000,
			Perms:     &ProcMapPermissions{true, false, false, false, true},
			Offset:    0,
			Dev:       unix.Mkdev(0xfd, 0x01),
			Inode:     47316994,
			Pathname:  "/bin/cat",
		},
		{
			StartAddr: 0x55680ae29000,
			EndAddr:   0x55680ae2a000,
			Perms:     &ProcMapPermissions{true, true, true, true, false},
			Offset:    40960,
			Dev:       unix.Mkdev(0xfd, 0x01),
			Inode:     47316994,
			Pathname:  "/bin/cat",
		},
		{
			StartAddr: 0x55680bed6000,
			EndAddr:   0x55680bef7000,
			Perms:     &ProcMapPermissions{true, true, false, false, true},
			Offset:    0,
			Dev:       unix.Mkdev(0, 0),
			Inode:     0,
			Pathname:  "[heap]",
		},
		{
			StartAddr: 0x7fdf964fc000,
			EndAddr:   0x7fdf973f2000,
			Perms:     &ProcMapPermissions{true, false, false, false, true},
			Offset:    0,
			Dev:       unix.Mkdev(0xfd, 0x01),
			Inode:     17432624,
			Pathname:  "/usr/lib/locale/locale-archive",
		},
		{
			StartAddr: 0x7fdf973f2000,
			EndAddr:   0x7fdf97417000,
			Perms:     &ProcMapPermissions{true, false, false, false, true},
			Offset:    0,
			Dev:       unix.Mkdev(0xfd, 0x01),
			Inode:     60571062,
			Pathname:  "/lib/x86_64-linux-gnu/libc-2.29.so",
		},
		{
			StartAddr: 0x7ffe9215c000,
			EndAddr:   0x7ffe9217f000,
			Perms:     &ProcMapPermissions{true, true, false, false, true},
			Offset:    0,
			Dev:       0,
			Inode:     0,
			Pathname:  "[stack]",
		},
		{
			StartAddr: 0x7ffe921da000,
			EndAddr:   0x7ffe921dd000,
			Perms:     &ProcMapPermissions{true, false, false, false, true},
			Offset:    0,
			Dev:       0,
			Inode:     0,
			Pathname:  "[vvar]",
		},
		{
			StartAddr: 0x7ffe921dd000,
			EndAddr:   0x7ffe921de000,
			Perms:     &ProcMapPermissions{true, false, true, false, true},
			Offset:    0,
			Dev:       0,
			Inode:     0,
			Pathname:  "[vdso]",
		},
		{
			StartAddr: 0xffffffffff600000,
			EndAddr:   0xffffffffff601000,
			Perms:     &ProcMapPermissions{false, false, true, false, true},
			Offset:    0,
			Dev:       0,
			Inode:     0,
			Pathname:  "[vsyscall]",
		},
	}

	// 64-bit test pid and fixtures
	tpid := 26232
	tsts := tsts64

	p, err := getProcFixtures(t).Proc(tpid)
	if err != nil {
		t.Fatal(err)
	}

	maps, err := p.ProcMaps()
	if err != nil {
		t.Fatal(err)
	}

	if want, have := len(maps), len(tsts); want > have {
		t.Errorf("want at least %d parsed proc/map entries, have %d", want, have)
	}

	for idx, tst := range tsts {
		want, got := tst, maps[idx]
		if diff := cmp.Diff(want, got); diff != "" {
			t.Fatalf("unexpected proc/map entry (-want +got):\n%s", diff)
		}
	}

}
