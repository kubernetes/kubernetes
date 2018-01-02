package procfs

import (
	"os"
	"testing"
)

func TestProcStat(t *testing.T) {
	p, err := FS("fixtures").NewProc(26231)
	if err != nil {
		t.Fatal(err)
	}

	s, err := p.NewStat()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name string
		want int
		have int
	}{
		{name: "pid", want: 26231, have: s.PID},
		{name: "user time", want: 1677, have: int(s.UTime)},
		{name: "system time", want: 44, have: int(s.STime)},
		{name: "start time", want: 82375, have: int(s.Starttime)},
		{name: "virtual memory size", want: 56274944, have: s.VSize},
		{name: "resident set size", want: 1981, have: s.RSS},
	} {
		if test.want != test.have {
			t.Errorf("want %s %d, have %d", test.name, test.want, test.have)
		}
	}
}

func TestProcStatComm(t *testing.T) {
	s1, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "vim", s1.Comm; want != have {
		t.Errorf("want comm %s, have %s", want, have)
	}

	s2, err := testProcStat(584)
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "(a b ) ( c d) ", s2.Comm; want != have {
		t.Errorf("want comm %s, have %s", want, have)
	}
}

func TestProcStatVirtualMemory(t *testing.T) {
	s, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}

	if want, have := 56274944, s.VirtualMemory(); want != have {
		t.Errorf("want virtual memory %d, have %d", want, have)
	}
}

func TestProcStatResidentMemory(t *testing.T) {
	s, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}

	if want, have := 1981*os.Getpagesize(), s.ResidentMemory(); want != have {
		t.Errorf("want resident memory %d, have %d", want, have)
	}
}

func TestProcStatStartTime(t *testing.T) {
	s, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}

	time, err := s.StartTime()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := 1418184099.75, time; want != have {
		t.Errorf("want start time %f, have %f", want, have)
	}
}

func TestProcStatCPUTime(t *testing.T) {
	s, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}

	if want, have := 17.21, s.CPUTime(); want != have {
		t.Errorf("want cpu time %f, have %f", want, have)
	}
}

func testProcStat(pid int) (ProcStat, error) {
	p, err := FS("fixtures").NewProc(pid)
	if err != nil {
		return ProcStat{}, err
	}

	return p.NewStat()
}
