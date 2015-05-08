package types

import (
	"testing"
)

func TestEnvironmentAssertValid(t *testing.T) {
	tests := []struct {
		env  Environment
		werr bool
	}{
		// duplicate names should fail
		{
			Environment{
				EnvironmentVariable{"DEBUG", "true"},
				EnvironmentVariable{"DEBUG", "true"},
			},
			true,
		},
		// empty name should fail
		{
			Environment{
				EnvironmentVariable{"", "value"},
			},
			true,
		},
		// name beginning with digit should fail
		{
			Environment{
				EnvironmentVariable{"0DEBUG", "true"},
			},
			true,
		},
		// name with non [A-Za-z0-9_] should fail
		{
			Environment{
				EnvironmentVariable{"VERBOSE-DEBUG", "true"},
			},
			true,
		},
		// accepted environment variable forms
		{
			Environment{
				EnvironmentVariable{"DEBUG", "true"},
			},
			false,
		},
		{
			Environment{
				EnvironmentVariable{"_0_DEBUG_0_", "true"},
			},
			false,
		},
	}
	for i, test := range tests {
		env := Environment(test.env)
		err := env.assertValid()
		if gerr := (err != nil); gerr != test.werr {
			t.Errorf("#%d: gerr=%t, want %t (err=%v)", i, gerr, test.werr, err)
		}
	}
}
