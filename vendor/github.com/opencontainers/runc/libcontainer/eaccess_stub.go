//go:build go1.20

package libcontainer

func eaccess(path string) error {
	// Not needed in Go 1.20+ as the functionality is already in there
	// (added by https://go.dev/cl/416115, https://go.dev/cl/414824,
	// and fixed in Go 1.20.2 by https://go.dev/cl/469956).
	return nil
}
