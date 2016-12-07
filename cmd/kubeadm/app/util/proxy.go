/*
Copyright 2014 The Kubernetes Authors.

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

package util

import (
	"os"
	"strings"
)

// UnsetProxyEnvironmentVariables removing all *_proxy variables from
// environment. Helpful in situations where API clients must be forced
// to work over direct connections.
func UnsetProxyEnvironmentVariables() {
	for _, envVar := range os.Environ() {
		varName := strings.Split(envVar, "=")[0]
		if strings.HasSuffix(strings.ToLower(varName), "_proxy") {
			os.Unsetenv(varName)
		}
	}
}
