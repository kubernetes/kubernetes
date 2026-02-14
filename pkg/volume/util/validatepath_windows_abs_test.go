//go:build windows

package util

import "testing"

func TestValidatePathRejectsWindowsAbsoluteAndUNC(t *testing.T) {
	cases := []string{
		`C:\Windows\Temp\pwn.txt`,
		`\\server\share\pwn.txt`,
		`C:relative-but-has-volume`,
	}
	for _, tc := range cases {
		if err := validatePath(tc); err == nil {
			t.Fatalf("expected validatePath to reject %q, but got nil error", tc)
		}
	}
}
