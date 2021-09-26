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
// +build 386 arm mips mipsle

package procfs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/sys/unix"
)

func TestProcMaps(t *testing.T) {
	tsts32 := []*ProcMap{
		{
			StartAddr: 0x08048000,
			EndAddr:   0x08089000,
			Perms:     &ProcMapPermissions{true, false, true, false, true},
			Offset:    0,
			Dev:       unix.Mkdev(0x03, 0x01),
			Inode:     104219,
			Pathname:  "/bin/tcsh",
		},
		{
			StartAddr: 0x08089000,
			EndAddr:   0x0808c000,
			Perms:     &ProcMapPermissions{true, true, false, false, true},
			Offset:    266240,
			Dev:       unix.Mkdev(0x03, 0x01),
			Inode:     104219,
			Pathname:  "/bin/tcsh",
		},
		{
			StartAddr: 0x0808c000,
			EndAddr:   0x08146000,
			Perms:     &ProcMapPermissions{true, true, true, false, true},
			Offset:    0,
			Dev:       unix.Mkdev(0x00, 0x00),
			Inode:     0,
			Pathname:  "",
		},
		{
			StartAddr: 0x40000000,
			EndAddr:   0x40015000,
			Perms:     &ProcMapPermissions{true, false, true, false, true},
			Offset:    0,
			Dev:       unix.Mkdev(0x03, 0x01),
			Inode:     61874,
			Pathname:  "/lib/ld-2.3.2.so",
		},
	}

	// 32-bit test pid and fixtures
	tpid := 26234
	tsts := tsts32

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
