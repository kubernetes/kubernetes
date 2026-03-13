/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kernel

import (
	"fmt"
	"os"
	"strings"

	"k8s.io/apimachinery/pkg/util/version"
)

type readFileFunc func(string) ([]byte, error)

// GetVersion returns currently running kernel version.
func GetVersion() (*version.Version, error) {
	return getVersion(os.ReadFile)
}

// getVersion reads os release file from the give readFile function.
func getVersion(readFile readFileFunc) (*version.Version, error) {
	kernelVersionFile := "/proc/sys/kernel/osrelease"
	fileContent, err := readFile(kernelVersionFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read os-release file: %s", err.Error())
	}

	kernelVersion, err := version.ParseGeneric(strings.TrimSpace(string(fileContent)))
	if err != nil {
		return nil, fmt.Errorf("failed to parse kernel version: %s", err.Error())
	}

	return kernelVersion, nil
}
