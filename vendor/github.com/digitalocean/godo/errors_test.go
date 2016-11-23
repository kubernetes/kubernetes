package godo

import "testing"

func TestArgError(t *testing.T) {
	expected := "foo is invalid because bar"
	err := NewArgError("foo", "bar")
	if got := err.Error(); got != expected {
		t.Errorf("ArgError().Error() = %q; expected %q", got, expected)
	}
}
