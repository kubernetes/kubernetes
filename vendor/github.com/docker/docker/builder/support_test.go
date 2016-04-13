package builder

import (
	"fmt"
	"testing"
)

func TestSelectAcceptableMIME(t *testing.T) {
	validMimeStrings := []string{
		"application/x-bzip2",
		"application/bzip2",
		"application/gzip",
		"application/x-gzip",
		"application/x-xz",
		"application/xz",
		"application/tar",
		"application/x-tar",
		"application/octet-stream",
		"text/plain",
	}

	invalidMimeStrings := []string{
		"",
		"application/octet",
		"application/json",
	}

	for _, m := range invalidMimeStrings {
		if len(selectAcceptableMIME(m)) > 0 {
			err := fmt.Errorf("Should not have accepted %q", m)
			t.Fatal(err)
		}
	}

	for _, m := range validMimeStrings {
		if str := selectAcceptableMIME(m); str == "" {
			err := fmt.Errorf("Should have accepted %q", m)
			t.Fatal(err)
		}
	}
}
