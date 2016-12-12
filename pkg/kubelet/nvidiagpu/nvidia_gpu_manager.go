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

package nvidiagpu

import (
	"fmt"
	"sync"
	"os"
	"path/filepath"
	"regexp"

	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	dockertypes "github.com/docker/engine-api/types"
)


// TODO: If use NVML in the future, the implementation could be more complex,
// but also more powerful!

const (
	// All NVIDIA GPUs cards should be mounted with nvidiactl and nvidia-uvm
	NvidiaDeviceCtl string = "/dev/nvidiactl"
	NvidiaDeviceUVM string = "/dev/nvidia-uvm"
)

type NvidiaGPUManager struct {
	gpuPaths []string
	gpuMutex sync.Mutex

	dockerClient dockertools.DockerInterface
}

// Get all the paths of NVIDIA GPU card from /dev/
func (ngm *NvidiaGPUManager) discovery() error {
	var err error
	if ngm.gpuPaths == nil {
		err = filepath.Walk("/dev", func(path string, f os.FileInfo, err error) error {
			reg := regexp.MustCompile(`^nvidia[0-9]*$`)
			gpupath := reg.FindAllString(f.Name(), -1)
			if gpupath != nil && gpupath[0] != "" {
				ngm.gpuPaths = append(ngm.gpuPaths, "/dev/"+gpupath[0])
			}

			return nil
		})
	}

	return err
}

func Valid(path string) bool {
	reg := regexp.MustCompile(`^/dev/nvidia[0-9]*$`)
	check := reg.FindAllString(path, -1)

	return check != nil && check[0] != ""
}

func (ngm *NvidiaGPUManager) Init(dc dockertools.DockerInterface) error {
	if _, err := os.Stat(NvidiaDeviceCtl); err != nil {
		return err
	}
	if _, err := os.Stat(NvidiaDeviceUVM); err != nil {
		return err
	}

	ngm.gpuMutex.Lock()
	defer ngm.gpuMutex.Unlock()

	err := ngm.discovery()

	ngm.dockerClient = dc

	return err
}

func (ngm *NvidiaGPUManager) Shutdown() {
	ngm.gpuMutex.Lock()
	defer ngm.gpuMutex.Unlock()

	ngm.gpuPaths = nil
}

func (ngm *NvidiaGPUManager) Capacity() int {
	ngm.gpuMutex.Lock()
	defer ngm.gpuMutex.Unlock()

	return len(ngm.gpuPaths)
}

func (ngm *NvidiaGPUManager) isAvailable(path string) bool {
	containers, err := ngm.dockerClient.ListContainers(dockertypes.ContainerListOptions{All: true})

	if err != nil {
		return true
	}

	for i := range containers {
		containerJSON, err := ngm.dockerClient.InspectContainer(containers[i].ID)

		if err != nil {
			continue
		}

		devices := containerJSON.HostConfig.Devices

		if devices == nil {
			continue
		}

		for _, device := range devices {
			if Valid(device.PathOnHost) && path == device.PathOnHost {
				return false
			}
		}
	}

	return true
}

func (ngm *NvidiaGPUManager) AllocateGPUs(num int) (paths []string, err error) {
	if num <= 0 {
		return
	}

	ngm.gpuMutex.Lock()
	defer ngm.gpuMutex.Unlock()

	for _, path := range ngm.gpuPaths {
		if ngm.isAvailable(path) {
			paths = append(paths, path)
			if len(paths) == num {
				return
			}
		}
	}

	err = fmt.Errorf("Do not have sufficient GPUs!")

	return
}

func (ngm *NvidiaGPUManager) AvailableGPUs() (num int) {
	ngm.gpuMutex.Lock()
	defer ngm.gpuMutex.Unlock()

	for _, path := range ngm.gpuPaths {
		if ngm.isAvailable(path) {
			num++
		}
	}

	return
}

