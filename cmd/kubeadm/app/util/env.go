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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// GetProxyEnvVars builds a list of environment variables in order to use the right proxy
func GetProxyEnvVars() []kubeadmapi.EnvVar {
	envs := []kubeadmapi.EnvVar{}
	for _, env := range os.Environ() {
		pos := strings.Index(env, "=")
		if pos == -1 {
			// malformed environment variable, skip it.
			continue
		}
		name := env[:pos]
		value := env[pos+1:]
		if strings.HasSuffix(strings.ToLower(name), "_proxy") && value != "" {
			envVar := kubeadmapi.EnvVar{
				EnvVar: v1.EnvVar{Name: name, Value: value},
			}
			envs = append(envs, envVar)
		}
	}
	return envs
}

// MergeKubeadmEnvVars merges values of environment variable slices.
// The values defined in later slices overwrite values in previous ones.
func MergeKubeadmEnvVars(envList ...[]kubeadmapi.EnvVar) []v1.EnvVar {
	m := make(map[string]v1.EnvVar)
	merged := []v1.EnvVar{}
	for _, envs := range envList {
		for _, env := range envs {
			m[env.Name] = env.EnvVar
		}
	}
	for _, v := range m {
		merged = append(merged, v)
	}
	return merged
}
