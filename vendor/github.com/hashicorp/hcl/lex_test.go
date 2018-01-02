package hcl

import (
	"testing"
)

func TestLexMode(t *testing.T) {
	cases := []struct {
		Input string
		Mode  lexModeValue
	}{
		{
			"",
			lexModeHcl,
		},
		{
			"foo",
			lexModeHcl,
		},
		{
			"{}",
			lexModeJson,
		},
		{
			"  {}",
			lexModeJson,
		},
	}

	for i, tc := range cases {
		actual := lexMode([]byte(tc.Input))

		if actual != tc.Mode {
			t.Fatalf("%d: %#v", i, actual)
		}
	}
}
