package xunit_tests

import (
	"testing"
)

func TestAlwaysTrue(t *testing.T) {
	if AlwaysTrue() != true {
		t.Errorf("Expected true, got false")
	}
}
