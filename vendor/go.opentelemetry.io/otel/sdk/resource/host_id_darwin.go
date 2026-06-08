// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package resource // import "go.opentelemetry.io/otel/sdk/resource"

var platformHostIDReader hostIDReader = &hostIDReaderDarwin{
	execCommand: execCommand,
}
