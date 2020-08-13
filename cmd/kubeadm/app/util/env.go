/*
Copyright 2019 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
)

// GetProxyEnvVars builds a list of environment variables in order to use the right proxy
func GetProxyEnvVars() []v1.EnvVar {
	envs := []v1.EnvVar{}
	for _, env := range os.Environ() {
		pos := strings.Index(env, "=")
		if pos == -1 {
			// malformed environment variable, skip it.
			continue
		}
		name := env[:pos]
		value := env[pos+1:]
		if strings.HasSuffix(strings.ToLower(name), "_proxy") && value != "" {
			envVar := v1.EnvVar{Name: name, Value: value}
			envs = append(envs, envVar)
		}
	}
	return envs
}
