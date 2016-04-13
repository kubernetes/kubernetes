package procfs

import "testing"

func TestNewLimits(t *testing.T) {
	fs, err := NewFS("fixtures")
	if err != nil {
		t.Fatal(err)
	}

	p, err := fs.NewProc(26231)
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
		got  int
	}{
		{name: "cpu time", want: -1, got: l.CPUTime},
		{name: "open files", want: 2048, got: l.OpenFiles},
		{name: "msgqueue size", want: 819200, got: l.MsqqueueSize},
		{name: "nice priority", want: 0, got: l.NicePriority},
		{name: "address space", want: -1, got: l.AddressSpace},
	} {
		if test.want != test.got {
			t.Errorf("want %s %d, got %d", test.name, test.want, test.got)
		}
	}
}
