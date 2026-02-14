//go:build unix && !darwin

package testscript

import "os"

// cloneFile makes a clone of a file via a hard link.
func cloneFile(from, to string) error {
	return os.Link(from, to)
}
