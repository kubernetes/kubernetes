// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build !aix && !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris && !windows && !zos
// +build !aix,!darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!solaris,!windows,!zos

package resource // import "go.opentelemetry.io/otel/sdk/resource"

// platformOSDescription is a placeholder implementation for OSes
// for which this project currently doesn't support os.description
// attribute detection. See build tags declaration early on this file
// for a list of unsupported OSes.
func platformOSDescription() (string, error) {
	return "<unknown>", nil
}
