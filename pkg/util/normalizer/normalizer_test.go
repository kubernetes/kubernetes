package normalizer

import (
	"testing"
)

func TestLongDesc(t *testing.T) {
	s := ""
	if r := LongDesc(s); r != s {
		t.Errorf("Unexpected result for zero length string, got: %v, want: %v", r, s)
	}
}
