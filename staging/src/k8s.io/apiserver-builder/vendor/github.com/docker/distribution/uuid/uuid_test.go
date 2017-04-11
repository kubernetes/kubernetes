package uuid

import (
	"testing"
)

const iterations = 1000

func TestUUID4Generation(t *testing.T) {
	for i := 0; i < iterations; i++ {
		u := Generate()

		if u[6]&0xf0 != 0x40 {
			t.Fatalf("version byte not correctly set: %v, %08b %08b", u, u[6], u[6]&0xf0)
		}

		if u[8]&0xc0 != 0x80 {
			t.Fatalf("top order 8th byte not correctly set: %v, %b", u, u[8])
		}
	}
}

func TestParseAndEquality(t *testing.T) {
	for i := 0; i < iterations; i++ {
		u := Generate()

		parsed, err := Parse(u.String())
		if err != nil {
			t.Fatalf("error parsing uuid %v: %v", u, err)
		}

		if parsed != u {
			t.Fatalf("parsing round trip failed: %v != %v", parsed, u)
		}
	}

	for _, c := range []string{
		"bad",
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",    // correct length, incorrect format
		"  20cc7775-2671-43c7-8742-51d1cfa23258",  // leading space
		"20cc7775-2671-43c7-8742-51d1cfa23258   ", // trailing space
		"00000000-0000-0000-0000-x00000000000",    // out of range character
	} {
		if _, err := Parse(c); err == nil {
			t.Fatalf("parsing %q should have failed", c)
		}
	}
}
