package s3crypto

import (
	"testing"
)

func TestGenerateBytes(t *testing.T) {
	b, _ := generateBytes(5)
	if e, a := 5, len(b); e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}
	b, _ = generateBytes(0)
	if e, a := 0, len(b); e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}
	b, _ = generateBytes(1024)
	if e, a := 1024, len(b); e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}
}
