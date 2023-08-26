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

//go:build windows
// +build windows

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"golang.org/x/sys/windows/registry"
)

// implements hostIDReader
type hostIDReaderWindows struct{}

// read reads MachineGuid from the windows registry key:
// SOFTWARE\Microsoft\Cryptography
func (*hostIDReaderWindows) read() (string, error) {
	k, err := registry.OpenKey(
		registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Cryptography`,
		registry.QUERY_VALUE|registry.WOW64_64KEY,
	)

	if err != nil {
		return "", err
	}
	defer k.Close()

	guid, _, err := k.GetStringValue("MachineGuid")
	if err != nil {
		return "", err
	}

	return guid, nil
}

var platformHostIDReader hostIDReader = &hostIDReaderWindows{}
