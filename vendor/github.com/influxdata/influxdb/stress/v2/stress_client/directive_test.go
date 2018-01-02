package stressClient

import (
	"testing"
)

func TestNewDirective(t *testing.T) {
	tr := NewTracer(map[string]string{})
	prop := "foo_prop"
	val := "foo_value"
	dir := NewDirective(prop, val, tr)
	got := dir.Property
	if prop != got {
		t.Errorf("expected: %v\ngot: %v\n", prop, got)
	}
	got = dir.Value
	if val != got {
		t.Errorf("expected: %v\ngot: %v\n", val, got)
	}
}
