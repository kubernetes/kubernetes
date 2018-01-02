package procfs

import "testing"

func TestNewLimits(t *testing.T) {
	p, err := FS("fixtures").NewProc(26231)
	if err != nil {
		t.Fatal(err)
	}

	l, err := p.NewLimits()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name string
		want int
		have int
	}{
		{name: "cpu time", want: -1, have: l.CPUTime},
		{name: "open files", want: 2048, have: l.OpenFiles},
		{name: "msgqueue size", want: 819200, have: l.MsqqueueSize},
		{name: "nice priority", want: 0, have: l.NicePriority},
		{name: "address space", want: -1, have: l.AddressSpace},
	} {
		if test.want != test.have {
			t.Errorf("want %s %d, have %d", test.name, test.want, test.have)
		}
	}
}
