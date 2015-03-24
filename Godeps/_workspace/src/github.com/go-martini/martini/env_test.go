package martini

import (
	"testing"
)

func Test_SetENV(t *testing.T) {
	tests := []struct {
		in  string
		out string
	}{
		{"", "development"},
		{"not_development", "not_development"},
	}

	for _, test := range tests {
		setENV(test.in)
		if Env != test.out {
			expect(t, Env, test.out)
		}
	}
}

func Test_Root(t *testing.T) {
	if len(Root) == 0 {
		t.Errorf("Expected root path will be set")
	}
}
