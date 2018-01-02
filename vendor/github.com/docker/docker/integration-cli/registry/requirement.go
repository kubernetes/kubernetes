package registry

import "os/exec"

// Hosting returns wether the host can host a registry (v2) or not
func Hosting() bool {
	// for now registry binary is built only if we're running inside
	// container through `make test`. Figure that out by testing if
	// registry binary is in PATH.
	_, err := exec.LookPath(v2binary)
	return err == nil
}
