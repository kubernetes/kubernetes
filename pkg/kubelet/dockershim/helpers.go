/*
Copyright 2016 The Kubernetes Authors.

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

package dockershim

import (
	"fmt"

	dockerapiversion "github.com/docker/engine-api/types/versions"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// apiVersion implements kubecontainer.Version interface by implementing
// Compare() and String(). It uses the compare function of engine-api to
// compare docker apiversions.
type apiVersion string

func (v apiVersion) String() string {
	return string(v)
}

func (v apiVersion) Compare(other string) (int, error) {
	if dockerapiversion.LessThan(string(v), other) {
		return -1, nil
	} else if dockerapiversion.GreaterThan(string(v), other) {
		return 1, nil
	}
	return 0, nil
}

// generateEnvList converts KeyValue list to a list of strings, in the form of
// '<key>=<value>', which can be understood by docker.
func generateEnvList(envs []*runtimeApi.KeyValue) (result []string) {
	for _, env := range envs {
		result = append(result, fmt.Sprintf("%s=%s", env.GetKey(), env.GetValue()))
	}
	return
}

// generateMountBindings converts the mount list to a list of strings that
// can be understood by docker.
// Each element in the string is in the form of:
// '<HostPath>:<ContainerPath>', or
// '<HostPath>:<ContainerPath>:ro', if the path is read only, or
// '<HostPath>:<ContainerPath>:Z', if the volume requires SELinux
// relabeling and the pod provides an SELinux label
func generateMountBindings(mounts []*runtimeApi.Mount) (result []string) {
	// TODO: resolve podHasSELinuxLabel
	for _, m := range mounts {
		bind := fmt.Sprintf("%s:%s", m.GetHostPath(), m.GetContainerPath())
		readOnly := m.GetReadonly()
		if readOnly {
			bind += ":ro"
		}
		if m.GetSelinuxRelabel() {
			if readOnly {
				bind += ",Z"
			} else {
				bind += ":Z"
			}
		}
		result = append(result, bind)
	}
	return
}
