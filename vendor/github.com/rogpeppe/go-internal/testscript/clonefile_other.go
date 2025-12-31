//go:build !unix

package testscript

import "fmt"

// cloneFile does not attempt anything on Windows, as hard links on it have
// led to "access denied" errors when deleting files at the end of a test.
// We haven't tested platforms like plan9 or wasm/wasi.
func cloneFile(from, to string) error {
	return fmt.Errorf("unavailable")
}
