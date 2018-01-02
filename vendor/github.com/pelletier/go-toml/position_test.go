// Testing support for go-toml

package toml

import (
	"testing"
)

func TestPositionString(t *testing.T) {
	p := Position{123, 456}
	expected := "(123, 456)"
	value := p.String()

	if value != expected {
		t.Errorf("Expected %v, got %v instead", expected, value)
	}
}

func TestInvalid(t *testing.T) {
	for i, v := range []Position{
		Position{0, 1234},
		Position{1234, 0},
		Position{0, 0},
	} {
		if !v.Invalid() {
			t.Errorf("Position at %v is valid: %v", i, v)
		}
	}
}
