package procfs

import "testing"

func TestStat(t *testing.T) {
	s, err := FS("fixtures").NewStat()
	if err != nil {
		t.Fatal(err)
	}

	if want, have := int64(1418183276), s.BootTime; want != have {
		t.Errorf("want boot time %d, have %d", want, have)
	}
}
