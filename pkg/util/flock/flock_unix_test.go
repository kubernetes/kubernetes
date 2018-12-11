package flock

import (
	"testing"
)

func TestAquire(t *testing.T) {
	path := ""
	if r := Acquire(path); r == nil {
		t.Errorf("Unexpected result for zero path, got: %v, want: %v", r, nil)
	}
}
