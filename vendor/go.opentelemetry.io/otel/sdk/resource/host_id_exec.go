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

//go:build bsd || darwin

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import "os/exec"

func execCommand(name string, arg ...string) (string, error) {
	cmd := exec.Command(name, arg...)
	b, err := cmd.Output()
	if err != nil {
		return "", err
	}

	return string(b), nil
}
