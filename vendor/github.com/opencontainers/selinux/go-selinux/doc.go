/*
Package selinux provides a high-level interface for interacting with selinux.

This package uses a selinux build tag to enable the selinux functionality. This
allows non-linux and linux users who do not have selinux support to still use
tools that rely on this library.

To compile with full selinux support use the -tags=selinux option in your build
and test commands.

Usage:

	import "github.com/opencontainers/selinux/go-selinux"

	// Ensure that selinux is enforcing mode.
	if selinux.EnforceMode() != selinux.Enforcing {
		selinux.SetEnforceMode(selinux.Enforcing)
	}

*/
package selinux
