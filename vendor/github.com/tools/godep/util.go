package main

import (
	"path/filepath"
	"runtime"
	"strings"
)

// driveLetterToUpper converts Windows path's drive letters to uppercase. This
// is needed when comparing 2 paths with different drive letter case.
func driveLetterToUpper(path string) string {
	if runtime.GOOS != "windows" || path == "" {
		return path
	}

	p := path

	// If path's drive letter is lowercase, change it to uppercase.
	if len(p) >= 2 && p[1] == ':' && 'a' <= p[0] && p[0] <= 'z' {
		p = string(p[0]+'A'-'a') + p[1:]
	}

	return p
}

// clean the path and ensure that a drive letter is upper case (if it exists).
func cleanPath(path string) string {
	return driveLetterToUpper(filepath.Clean(path))
}

// deal with case insensitive filesystems and other weirdness
func pathEqual(a, b string) bool {
	a = cleanPath(a)
	b = cleanPath(b)
	return strings.EqualFold(a, b)
}
