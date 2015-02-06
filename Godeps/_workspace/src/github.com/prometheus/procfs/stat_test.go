package procfs

import "testing"

func TestStat(t *testing.T) {
	fs, err := NewFS("fixtures")
	if err != nil {
		t.Fatal(err)
	}

	s, err := fs.NewStat()
	if err != nil {
		t.Fatal(err)
	}

	if want, got := int64(1418183276), s.BootTime; want != got {
		t.Errorf("want boot time %d, got %d", want, got)
	}
}
