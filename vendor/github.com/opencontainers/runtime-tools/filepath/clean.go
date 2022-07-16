package filepath

import (
	"fmt"
	"strings"
)

// Clean is an explicit-OS version of path/filepath's Clean.
func Clean(os, path string) string {
	abs := IsAbs(os, path)
	sep := Separator(os)
	elements := strings.Split(path, string(sep))

	// Replace multiple Separator elements with a single one.
	for i := 0; i < len(elements); i++ {
		if len(elements[i]) == 0 {
			elements = append(elements[:i], elements[i+1:]...)
			i--
		}
	}

	// Eliminate each . path name element (the current directory).
	for i := 0; i < len(elements); i++ {
		if elements[i] == "." && len(elements) > 1 {
			elements = append(elements[:i], elements[i+1:]...)
			i--
		}
	}

	// Eliminate each inner .. path name element (the parent directory)
	// along with the non-.. element that precedes it.
	for i := 1; i < len(elements); i++ {
		if i == 1 && abs && sep == '\\' {
			continue
		}
		if i > 0 && elements[i] == ".." {
			elements = append(elements[:i-1], elements[i+1:]...)
			i -= 2
		}
	}

	// Eliminate .. elements that begin a rooted path:
	// that is, replace "/.." by "/" at the beginning of a path,
	// assuming Separator is '/'.
	offset := 0
	if sep == '\\' {
		offset = 1
	}
	if abs {
		for len(elements) > offset && elements[offset] == ".." {
			elements = append(elements[:offset], elements[offset+1:]...)
		}
	}

	cleaned := strings.Join(elements, string(sep))
	if abs {
		if sep == '/' {
			cleaned = fmt.Sprintf("%c%s", sep, cleaned)
		} else if len(elements) == 1 {
			cleaned = fmt.Sprintf("%s%c", cleaned, sep)
		}
	}

	// If the result of this process is an empty string, Clean returns
	// the string ".".
	if len(cleaned) == 0 {
		cleaned = "."
	}

	if cleaned == path {
		return path
	}
	return Clean(os, cleaned)
}
