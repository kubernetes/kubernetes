package humanize

import (
	"testing"
)

type testList []struct {
	name, got, exp string
}

func (tl testList) validate(t *testing.T) {
	for _, test := range tl {
		if test.got != test.exp {
			t.Errorf("On %v, expected '%v', but got '%v'",
				test.name, test.exp, test.got)
		}
	}
}
