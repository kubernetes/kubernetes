// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !darwin
// +build !dragonfly
// +build !freebsd
// +build !linux
// +build !netbsd
// +build !openbsd
// +build !solaris
// +build !windows

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
