// +build go1.7

package expression

import (
	"testing"
)

func TestInvalidParameterError(t *testing.T) {
	cases := []struct {
		name     string
		input    InvalidParameterError
		expected string
	}{
		{
			name:     "invalid error",
			input:    newInvalidParameterError("func", "param"),
			expected: "func error: invalid parameter: param",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual := c.input.Error()
			if e, a := c.expected, actual; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestUnsetParameterError(t *testing.T) {
	cases := []struct {
		name     string
		input    UnsetParameterError
		expected string
	}{
		{
			name:     "unset error",
			input:    newUnsetParameterError("func", "param"),
			expected: "func error: unset parameter: param",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual := c.input.Error()
			if e, a := c.expected, actual; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}
