/*
Copyright 2015 The Kubernetes Authors.

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

package cadvisor

import (
	"fmt"

	cadvisorfs "github.com/google/cadvisor/fs"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapi2 "github.com/google/cadvisor/info/v2"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// imageFsInfoProvider knows how to translate the configured runtime
// to its file system label for images.
type imageFsInfoProvider struct {
	runtime         string
	runtimeEndpoint string
}

// ImageFsInfoLabel returns the image fs label for the configured runtime.
// For remote runtimes, it handles additional runtimes natively understood by cAdvisor.
func (i *imageFsInfoProvider) ImageFsInfoLabel() (string, error) {
	switch i.runtime {
	case "docker":
		return cadvisorfs.LabelDockerImages, nil
	case "rkt":
		return cadvisorfs.LabelRktImages, nil
	case "remote":
		// TODO: pending rebase including https://github.com/google/cadvisor/pull/1741
		if i.runtimeEndpoint == "/var/run/crio.sock" {
			return "crio-images", nil
		}
	}
	return "", fmt.Errorf("no imagefs label for configured runtime")
}

// NewImageFsInfoProvider returns a provider for the specified runtime configuration.
func NewImageFsInfoProvider(runtime, runtimeEndpoint string) ImageFsInfoProvider {
	return &imageFsInfoProvider{runtime: runtime, runtimeEndpoint: runtimeEndpoint}
}

func CapacityFromMachineInfo(info *cadvisorapi.MachineInfo) v1.ResourceList {
	c := v1.ResourceList{
		v1.ResourceCPU: *resource.NewMilliQuantity(
			int64(info.NumCores*1000),
			resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(
			int64(info.MemoryCapacity),
			resource.BinarySI),
	}
	return c
}

func EphemeralStorageCapacityFromFsInfo(info cadvisorapi2.FsInfo) v1.ResourceList {
	c := v1.ResourceList{
		v1.ResourceEphemeralStorage: *resource.NewQuantity(
			int64(info.Capacity),
			resource.BinarySI),
	}
	return c
}
