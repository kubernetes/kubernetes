package normalizer

import (
	"testing"
)

func TestTrim(t *testing.T) {
	norm := normalizer{"  foobar"}
	normTrim := normalizer{"foobar"}
	if r := norm.Trim(); r != normTrim {
		t.Errorf("Unexpected result for normalizer trim, got: %v, want: %v", r, normTrim)
	}
}
