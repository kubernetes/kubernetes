// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package resource

var platformHostIDReader hostIDReader = &hostIDReaderDarwin{
	execCommand: execCommand,
}
