package labels

import (
	"strings"
	"testing"

	"github.com/containerd/containerd/errdefs"
)

func TestValidLabels(t *testing.T) {
	shortStr := "s"
	longStr := strings.Repeat("s", maxSize-1)

	for key, value := range map[string]string{
		"some":   "value",
		shortStr: longStr,
	} {
		if err := Validate(key, value); err != nil {
			t.Fatalf("unexpected error: %v != nil", err)
		}
	}
}

func TestInvalidLabels(t *testing.T) {
	addOneStr := "s"
	maxSizeStr := strings.Repeat("s", maxSize)

	for key, value := range map[string]string{
		maxSizeStr: addOneStr,
	} {
		if err := Validate(key, value); err == nil {
			t.Fatal("expected invalid error")
		} else if !errdefs.IsInvalidArgument(err) {
			t.Fatal("error should be an invalid label error")
		}
	}
}
