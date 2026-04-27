/*
Package selinux provides a high-level interface for interacting with selinux.

Usage:

	import "github.com/opencontainers/selinux/go-selinux"

	// Ensure that selinux is enforcing mode.
	if selinux.EnforceMode() != selinux.Enforcing {
		selinux.SetEnforceMode(selinux.Enforcing)
	}
*/
package selinux
