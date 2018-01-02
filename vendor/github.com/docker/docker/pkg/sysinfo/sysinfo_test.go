package sysinfo

import "testing"

func TestIsCpusetListAvailable(t *testing.T) {
	cases := []struct {
		provided  string
		available string
		res       bool
		err       bool
	}{
		{"1", "0-4", true, false},
		{"01,3", "0-4", true, false},
		{"", "0-7", true, false},
		{"1--42", "0-7", false, true},
		{"1-42", "00-1,8,,9", false, true},
		{"1,41-42", "43,45", false, false},
		{"0-3", "", false, false},
	}
	for _, c := range cases {
		r, err := isCpusetListAvailable(c.provided, c.available)
		if (c.err && err == nil) && r != c.res {
			t.Fatalf("Expected pair: %v, %v for %s, %s. Got %v, %v instead", c.res, c.err, c.provided, c.available, (c.err && err == nil), r)
		}
	}
}
