package dns

import "testing"

func TestOPTTtl(t *testing.T) {
	e := &OPT{}
	e.Hdr.Name = "."
	e.Hdr.Rrtype = TypeOPT

	if e.Do() {
		t.Fail()
	}

	e.SetDo()
	if !e.Do() {
		t.Fail()
	}

	oldTtl := e.Hdr.Ttl

	if e.Version() != 0 {
		t.Fail()
	}

	e.SetVersion(42)
	if e.Version() != 42 {
		t.Fail()
	}

	e.SetVersion(0)
	if e.Hdr.Ttl != oldTtl {
		t.Fail()
	}

	if e.ExtendedRcode() != 0 {
		t.Fail()
	}

	e.SetExtendedRcode(42)
	if e.ExtendedRcode() != 42 {
		t.Fail()
	}

	e.SetExtendedRcode(0)
	if e.Hdr.Ttl != oldTtl {
		t.Fail()
	}
}
