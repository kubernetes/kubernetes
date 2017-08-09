package sanity

import "testing"

func Test(t *testing.T) {
	var tests = []struct {
		s    string
		want bool
	}{
		{"# Correct", true},
		{"# incorrect", false},
		{"", false}, // test defensive programming
	}
	for _, test := range tests {
		if got := thirdCharacterIsCapital(test.s); got != test.want {
			t.Errorf("thirdCharacterIsCapital(%s) = %q", got)
		}
	}
}
