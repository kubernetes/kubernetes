package dbus

import (
	"testing"
)

// TestBasicSetActions asserts that Add & Remove behavior is correct
func TestBasicSetActions(t *testing.T) {
	s := newSet()

	if s.Contains("foo") {
		t.Fatal("set should not contain 'foo'")
	}

	s.Add("foo")

	if !s.Contains("foo") {
		t.Fatal("set should contain 'foo'")
	}

	v := s.Values()
	if len(v) != 1 {
		t.Fatal("set.Values did not report correct number of values")
	}
	if v[0] != "foo" {
		t.Fatal("set.Values did not report value")
	}

	s.Remove("foo")

	if s.Contains("foo") {
		t.Fatal("set should not contain 'foo'")
	}

	v = s.Values()
	if len(v) != 0 {
		t.Fatal("set.Values did not report correct number of values")
	}
}
