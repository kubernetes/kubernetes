// Copyright 2018 The Prometheus Authors
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

package procfs

import (
	"reflect"
	"sort"
	"testing"
)

func TestSelf(t *testing.T) {
	fs := getProcFixtures(t)

	p1, err := fs.Proc(26231)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := fs.Self()
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(p1, p2) {
		t.Errorf("want process %v, have %v", p1, p2)
	}
}

func TestAllProcs(t *testing.T) {
	procs, err := getProcFixtures(t).AllProcs()
	if err != nil {
		t.Fatal(err)
	}
	sort.Sort(procs)
	for i, p := range []*Proc{{PID: 584}, {PID: 26231}} {
		if want, have := p.PID, procs[i].PID; want != have {
			t.Errorf("want processes %d, have %d", want, have)
		}
	}
}

func TestCmdLine(t *testing.T) {
	for _, tt := range []struct {
		process int
		want    []string
	}{
		{process: 26231, want: []string{"vim", "test.go", "+10"}},
		{process: 26232, want: []string{}},
		{process: 26233, want: []string{"com.github.uiautomator"}},
	} {
		p1, err := getProcFixtures(t).Proc(tt.process)
		if err != nil {
			t.Fatal(err)
		}
		c1, err := p1.CmdLine()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tt.want, c1) {
			t.Errorf("want cmdline %v, have %v", tt.want, c1)
		}
	}
}

func TestWchan(t *testing.T) {
	for _, tt := range []struct {
		process int
		want    string
	}{
		{process: 26231, want: "poll_schedule_timeout"},
		{process: 26232, want: ""},
	} {
		p1, err := getProcFixtures(t).Proc(tt.process)
		if err != nil {
			t.Fatal(err)
		}
		c1, err := p1.Wchan()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tt.want, c1) {
			t.Errorf("want wchan %v, have %v", tt.want, c1)
		}
	}
}

func TestComm(t *testing.T) {
	for _, tt := range []struct {
		process int
		want    string
	}{
		{process: 26231, want: "vim"},
		{process: 26232, want: "ata_sff"},
	} {
		p1, err := getProcFixtures(t).Proc(tt.process)
		if err != nil {
			t.Fatal(err)
		}
		c1, err := p1.Comm()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tt.want, c1) {
			t.Errorf("want comm %v, have %v", tt.want, c1)
		}
	}
}

func TestExecutable(t *testing.T) {
	for _, tt := range []struct {
		process int
		want    string
	}{
		{process: 26231, want: "/usr/bin/vim"},
		{process: 26232, want: ""},
	} {
		p, err := getProcFixtures(t).Proc(tt.process)
		if err != nil {
			t.Fatal(err)
		}
		exe, err := p.Executable()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tt.want, exe) {
			t.Errorf("want absolute path to exe %v, have %v", tt.want, exe)
		}
	}
}

func TestCwd(t *testing.T) {
	for _, tt := range []struct {
		process    int
		want       string
		brokenLink bool
	}{
		{process: 26231, want: "/usr/bin"},
		{process: 26232, want: "/does/not/exist", brokenLink: true},
		{process: 26233, want: ""},
	} {
		p, err := getProcFixtures(t).Proc(tt.process)
		if err != nil {
			t.Fatal(err)
		}
		wd, err := p.Cwd()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tt.want, wd) {
			if wd == "" && tt.brokenLink {
				// Allow the result to be empty when can't os.Readlink broken links
				continue
			}
			t.Errorf("want absolute path to cwd %v, have %v", tt.want, wd)
		}
	}
}

func TestRoot(t *testing.T) {
	for _, tt := range []struct {
		process    int
		want       string
		brokenLink bool
	}{
		{process: 26231, want: "/"},
		{process: 26232, want: "/does/not/exist", brokenLink: true},
		{process: 26233, want: ""},
	} {
		p, err := getProcFixtures(t).Proc(tt.process)
		if err != nil {
			t.Fatal(err)
		}
		rdir, err := p.RootDir()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tt.want, rdir) {
			if rdir == "" && tt.brokenLink {
				// Allow the result to be empty when can't os.Readlink broken links
				continue
			}
			t.Errorf("want absolute path to rootdir %v, have %v", tt.want, rdir)
		}
	}
}

func TestFileDescriptors(t *testing.T) {
	p1, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}
	fds, err := p1.FileDescriptors()
	if err != nil {
		t.Fatal(err)
	}
	sort.Sort(byUintptr(fds))
	if want := []uintptr{0, 1, 2, 3, 10}; !reflect.DeepEqual(want, fds) {
		t.Errorf("want fds %v, have %v", want, fds)
	}
}

func TestFileDescriptorTargets(t *testing.T) {
	p1, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}
	fds, err := p1.FileDescriptorTargets()
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(fds)
	var want = []string{
		"../../symlinktargets/abc",
		"../../symlinktargets/def",
		"../../symlinktargets/ghi",
		"../../symlinktargets/uvw",
		"../../symlinktargets/xyz",
	}
	if !reflect.DeepEqual(want, fds) {
		t.Errorf("want fds %v, have %v", want, fds)
	}
}

func TestFileDescriptorsLen(t *testing.T) {
	p1, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}
	l, err := p1.FileDescriptorsLen()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := 5, l; want != have {
		t.Errorf("want fds %d, have %d", want, have)
	}
}

func TestFileDescriptorsInfo(t *testing.T) {
	p1, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}
	fdinfos, err := p1.FileDescriptorsInfo()
	if err != nil {
		t.Fatal(err)
	}
	sort.Sort(fdinfos)
	var want = ProcFDInfos{
		ProcFDInfo{FD: "0", Pos: "0", Flags: "02004000", MntID: "13", InotifyInfos: []InotifyInfo{
			InotifyInfo{WD: "3", Ino: "1", Sdev: "34", Mask: "fce"},
			InotifyInfo{WD: "2", Ino: "1300016", Sdev: "fd00002", Mask: "fce"},
			InotifyInfo{WD: "1", Ino: "2e0001", Sdev: "fd00000", Mask: "fce"},
		}},
		ProcFDInfo{FD: "1", Pos: "0", Flags: "02004002", MntID: "13", InotifyInfos: nil},
		ProcFDInfo{FD: "10", Pos: "0", Flags: "02004002", MntID: "9", InotifyInfos: nil},
		ProcFDInfo{FD: "2", Pos: "0", Flags: "02004002", MntID: "9", InotifyInfos: nil},
		ProcFDInfo{FD: "3", Pos: "0", Flags: "02004002", MntID: "9", InotifyInfos: nil},
	}
	if !reflect.DeepEqual(want, fdinfos) {
		t.Errorf("want fdinfos %+v, have %+v", want, fdinfos)
	}
}

type byUintptr []uintptr

func (a byUintptr) Len() int           { return len(a) }
func (a byUintptr) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byUintptr) Less(i, j int) bool { return a[i] < a[j] }
