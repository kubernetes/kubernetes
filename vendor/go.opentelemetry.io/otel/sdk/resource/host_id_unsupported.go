// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris && !windows
// +build !darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!solaris,!windows

package resource // import "go.opentelemetry.io/otel/sdk/resource"

// hostIDReaderUnsupported is a placeholder implementation for operating systems
// for which this project currently doesn't support host.id
// attribute detection. See build tags declaration early on this file
// for a list of unsupported OSes.
type hostIDReaderUnsupported struct{}

func (*hostIDReaderUnsupported) read() (string, error) {
	return "<unknown>", nil
}

var platformHostIDReader hostIDReader = &hostIDReaderUnsupported{}
