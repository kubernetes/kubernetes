package procfs

import (
	"os"
	"reflect"
	"sort"
	"testing"
)

func TestSelf(t *testing.T) {
	p1, err := NewProc(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	p2, err := Self()
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(p1, p2) {
		t.Errorf("want process %v to equal %v", p1, p2)
	}
}

func TestAllProcs(t *testing.T) {
	fs, err := NewFS("fixtures")
	if err != nil {
		t.Fatal(err)
	}
	procs, err := fs.AllProcs()
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
	p1, err := testProcess(26231)
	if err != nil {
		t.Fatal(err)
	}
	c, err := p1.CmdLine()
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"vim", "test.go", "+10"}; !reflect.DeepEqual(want, c) {
		t.Errorf("want cmdline %v, got %v", want, c)
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
	if want := []uintptr{0, 1, 2, 3, 4}; !reflect.DeepEqual(want, fds) {
		t.Errorf("want fds %v, got %v", want, fds)
	}

	p2, err := Self()
	if err != nil {
		t.Fatal(err)
	}

	fdsBefore, err := p2.FileDescriptors()
	if err != nil {
		t.Fatal(err)
	}

	s, err := os.Open("fixtures")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	fdsAfter, err := p2.FileDescriptors()
	if err != nil {
		t.Fatal(err)
	}

	if len(fdsBefore)+1 != len(fdsAfter) {
		t.Errorf("want fds %v+1 to equal %v", fdsBefore, fdsAfter)
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
