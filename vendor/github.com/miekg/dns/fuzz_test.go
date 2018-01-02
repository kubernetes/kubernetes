package dns

import "testing"

func TestFuzzString(t *testing.T) {
	testcases := []string{"", " MINFO ", "	RP ", "	NSEC 0 0", "	\" NSEC 0 0\"", "  \" MINFO \"",
		";a ", ";a����������",
		"	NSAP O ", "  NSAP N ",
		" TYPE4 TYPE6a789a3bc0045c8a5fb42c7d1bd998f5444 IN 9579b47d46817afbd17273e6",
		" TYPE45 3 3 4147994 TYPE\\(\\)\\)\\(\\)\\(\\(\\)\\(\\)\\)\\)\\(\\)\\(\\)\\(\\(\\R 948\"\")\\(\\)\\)\\)\\(\\ ",
		"$GENERATE 0-3 ${441189,5039418474430,o}",
		"$INCLUDE 00 TYPE00000000000n ",
		"$INCLUDE PE4 TYPE061463623/727071511 \\(\\)\\$GENERATE 6-462/0",
	}
	for i, tc := range testcases {
		rr, err := NewRR(tc)
		if err == nil {
			// rr can be nil because we can (for instance) just parse a comment
			if rr == nil {
				continue
			}
			t.Fatalf("parsed mailformed RR %d: %s", i, rr.String())
		}
	}
}
