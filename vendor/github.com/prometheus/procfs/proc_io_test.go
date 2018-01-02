package procfs

import "testing"

func TestProcIO(t *testing.T) {
	p, err := FS("fixtures").NewProc(26231)
	if err != nil {
		t.Fatal(err)
	}

	s, err := p.NewIO()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name string
		want int64
		have int64
	}{
		{name: "RChar", want: 750339, have: int64(s.RChar)},
		{name: "WChar", want: 818609, have: int64(s.WChar)},
		{name: "SyscR", want: 7405, have: int64(s.SyscR)},
		{name: "SyscW", want: 5245, have: int64(s.SyscW)},
		{name: "ReadBytes", want: 1024, have: int64(s.ReadBytes)},
		{name: "WriteBytes", want: 2048, have: int64(s.WriteBytes)},
		{name: "CancelledWriteBytes", want: -1024, have: s.CancelledWriteBytes},
	} {
		if test.want != test.have {
			t.Errorf("want %s %d, have %d", test.name, test.want, test.have)
		}
	}
}
