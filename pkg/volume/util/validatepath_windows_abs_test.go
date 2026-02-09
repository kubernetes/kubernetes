//go:build windows
package util

import "testing"

func TestValidatePathRejectsWindowsAbsolutePaths(t *testing.T) {
	cases := []string{
		`C:\Windows\Temp\pwn.txt`,
		`C:/Windows/Temp/pwn.txt`,
		`\\server\share\pwn.txt`,
	}
	for _, tc := range cases {
		if err := validatePath(tc); err == nil {
			t.Fatalf("expected validatePath to reject %q, but got nil error", tc)
		}
	}
}
