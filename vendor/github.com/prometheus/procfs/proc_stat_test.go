package procfs

import "testing"

func TestProcStat(t *testing.T) {
	fs, err := NewFS("fixtures")
	if err != nil {
		t.Fatal(err)
	}

	p, err := fs.NewProc(26231)
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
		got  int
	}{
		{name: "pid", want: 26231, got: s.PID},
		{name: "user time", want: 1677, got: int(s.UTime)},
		{name: "system time", want: 44, got: int(s.STime)},
		{name: "start time", want: 82375, got: int(s.Starttime)},
		{name: "virtual memory size", want: 56274944, got: s.VSize},
		{name: "resident set size", want: 1981, got: s.RSS},
	} {
		if test.want != test.got {
			t.Errorf("want %s %d, got %d", test.name, test.want, test.got)
		}
	}
}

func TestProcStatComm(t *testing.T) {
	s1, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}
	if want, got := "vim", s1.Comm; want != got {
		t.Errorf("want comm %s, got %s", want, got)
	}

	s2, err := testProcStat(584)
	if err != nil {
		t.Fatal(err)
	}
	if want, got := "(a b ) ( c d) ", s2.Comm; want != got {
		t.Errorf("want comm %s, got %s", want, got)
	}
}

func TestProcStatVirtualMemory(t *testing.T) {
	s, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}

	if want, got := 56274944, s.VirtualMemory(); want != got {
		t.Errorf("want virtual memory %d, got %d", want, got)
	}
}

func TestProcStatResidentMemory(t *testing.T) {
	s, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}

	if want, got := 1981*4096, s.ResidentMemory(); want != got {
		t.Errorf("want resident memory %d, got %d", want, got)
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
	if want, got := 1418184099.75, time; want != got {
		t.Errorf("want start time %f, got %f", want, got)
	}
}

func TestProcStatCPUTime(t *testing.T) {
	s, err := testProcStat(26231)
	if err != nil {
		t.Fatal(err)
	}

	if want, got := 17.21, s.CPUTime(); want != got {
		t.Errorf("want cpu time %f, got %f", want, got)
	}
}

func testProcStat(pid int) (ProcStat, error) {
	p, err := testProcess(pid)
	if err != nil {
		return ProcStat{}, err
	}

	return p.NewStat()
}
