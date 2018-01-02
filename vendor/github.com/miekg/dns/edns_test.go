package dns

import "testing"

func TestOPTTtl(t *testing.T) {
	e := &OPT{}
	e.Hdr.Name = "."
	e.Hdr.Rrtype = TypeOPT

	if e.Do() {
		t.Errorf("DO bit should be zero")
	}

	e.SetDo()
	if !e.Do() {
		t.Errorf("DO bit should be non-zero")
	}

	if e.Version() != 0 {
		t.Errorf("version should be non-zero")
	}

	e.SetVersion(42)
	if e.Version() != 42 {
		t.Errorf("set 42, expected %d, got %d", 42, e.Version())
	}

	e.SetExtendedRcode(42)
	if e.ExtendedRcode() != 42 {
		t.Errorf("set 42, expected %d, got %d", 42-15, e.ExtendedRcode())
	}
}
