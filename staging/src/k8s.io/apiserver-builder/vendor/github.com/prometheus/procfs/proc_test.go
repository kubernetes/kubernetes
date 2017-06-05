package procfs

import (
	"reflect"
	"sort"
	"testing"
)

func TestSelf(t *testing.T) {
	fs := FS("fixtures")

	p1, err := fs.NewProc(26231)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := fs.Self()
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(p1, p2) {
		t.Errorf("want process %v to equal %v", p1, p2)
	}
}

func TestAllProcs(t *testing.T) {
	procs, err := FS("fixtures").AllProcs()
	if err != nil {
		t.Fatal(err)
	}
	sort.Sort(procs)
	for i, p := range []*Proc{{PID: 584}, {PID: 26231}} {
		if want, got := p.PID, procs[i].PID; want != got {
			t.Errorf("want processes %d, got %d", want, got)
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
	} {
		p1, err := testProcess(tt.process)
		if err != nil {
			t.Fatal(err)
		}
		c1, err := p1.CmdLine()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tt.want, c1) {
			t.Errorf("want cmdline %v, got %v", tt.want, c1)
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
		p, err := testProcess(tt.process)
		if err != nil {
			t.Fatal(err)
		}
		exe, err := p.Executable()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tt.want, exe) {
			t.Errorf("want absolute path to cmdline %v, got %v", tt.want, exe)
		}
	}
}

func TestFileDescriptors(t *testing.T) {
	p1, err := testProcess(26231)
	if err != nil {
		t.Fatal(err)
	}
	fds, err := p1.FileDescriptors()
	if err != nil {
		t.Fatal(err)
	}
	sort.Sort(byUintptr(fds))
	if want := []uintptr{0, 1, 2, 3, 10}; !reflect.DeepEqual(want, fds) {
		t.Errorf("want fds %v, got %v", want, fds)
	}
}

func TestFileDescriptorTargets(t *testing.T) {
	p1, err := testProcess(26231)
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
		t.Errorf("want fds %v, got %v", want, fds)
	}
}

func TestFileDescriptorsLen(t *testing.T) {
	p1, err := testProcess(26231)
	if err != nil {
		t.Fatal(err)
	}
	l, err := p1.FileDescriptorsLen()
	if err != nil {
		t.Fatal(err)
	}
	if want, got := 5, l; want != got {
		t.Errorf("want fds %d, got %d", want, got)
	}
}

func testProcess(pid int) (Proc, error) {
	fs, err := NewFS("fixtures")
	if err != nil {
		return Proc{}, err
	}
	return fs.NewProc(pid)
}

type byUintptr []uintptr

func (a byUintptr) Len() int           { return len(a) }
func (a byUintptr) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byUintptr) Less(i, j int) bool { return a[i] < a[j] }
