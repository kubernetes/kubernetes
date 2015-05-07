package types

import "testing"

func TestExecValid(t *testing.T) {
	tests := []Exec{
		Exec{"/bin/httpd"},
		Exec{"/app"},
		Exec{"/app", "arg1", "arg2"},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err != nil {
			t.Errorf("#%d: err == %v, want nil", i, err)
		}
	}
}

func TestExecInvalid(t *testing.T) {
	tests := []Exec{
		Exec{},
		Exec{"app"},
	}
	for i, tt := range tests {
		if err := tt.assertValid(); err == nil {
			t.Errorf("#%d: err == nil, want non-nil", i)
		}
	}
}
