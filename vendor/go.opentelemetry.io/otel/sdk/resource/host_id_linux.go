// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package resource

var platformHostIDReader hostIDReader = &hostIDReaderLinux{
	readFile: readFile,
}
