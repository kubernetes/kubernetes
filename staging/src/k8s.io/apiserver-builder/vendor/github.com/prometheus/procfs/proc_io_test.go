package procfs

import "testing"

func TestProcIO(t *testing.T) {
	fs, err := NewFS("fixtures")
	if err != nil {
		t.Fatal(err)
	}

	p, err := fs.NewProc(26231)
	if err != nil {
		t.Fatal(err)
	}

	s, err := p.NewIO()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name string
		want uint64
		got  uint64
	}{
		{name: "RChar", want: 750339, got: s.RChar},
		{name: "WChar", want: 818609, got: s.WChar},
		{name: "SyscR", want: 7405, got: s.SyscR},
		{name: "SyscW", want: 5245, got: s.SyscW},
		{name: "ReadBytes", want: 1024, got: s.ReadBytes},
		{name: "WriteBytes", want: 2048, got: s.WriteBytes},
	} {
		if test.want != test.got {
			t.Errorf("want %s %d, got %d", test.name, test.want, test.got)
		}
	}

	for _, test := range []struct {
		name string
		want int64
		got  int64
	}{
		{name: "CancelledWriteBytes", want: -1024, got: s.CancelledWriteBytes},
	} {
		if test.want != test.got {
			t.Errorf("want %s %d, got %d", test.name, test.want, test.got)
		}
	}
}
