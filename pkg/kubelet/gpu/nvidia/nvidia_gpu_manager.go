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

package nvidia

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sync"

	"k8s.io/kubernetes/pkg/kubelet/dockertools"
)

// TODO: If use NVML in the future, the implementation could be more complex,
// but also more powerful!

const (
	// All NVIDIA GPUs cards should be mounted with nvidiactl and nvidia-uvm
	// If the driver installed correctly, the 2 devices must be there.
	NvidiaCtlDevice string = "/dev/nvidiactl"
	NvidiaUVMDevice string = "/dev/nvidia-uvm"
)

// Manage GPU devices.
type NvidiaGPUManager struct {
	gpuPaths []string
	gpuMutex sync.Mutex

	// The interface which could get GPU mapping from all the containers.
	// TODO: Should make this independent of Docker in the future.
	dockerClient dockertools.DockerInterface
}

// Get all the paths of NVIDIA GPU card from /dev/
// TODO: Without NVML support we only can check whether there has GPU devices, but
// could not give a health check or get more information like GPU cores, memory, or
// family name. Need to support NVML in the future. But we do not need NVML until
// we want more features, features like schedule containers according to GPU family
// name.
func (ngm *NvidiaGPUManager) discovery() (err error) {
	if ngm.gpuPaths == nil {
		err = filepath.Walk("/dev", func(path string, f os.FileInfo, err error) error {
			reg := regexp.MustCompile(`^nvidia[0-9]*$`)
			gpupath := reg.FindAllString(f.Name(), -1)
			if gpupath != nil && gpupath[0] != "" {
				ngm.gpuPaths = append(ngm.gpuPaths, "/dev/"+gpupath[0])
			}

			return nil
		})

		if err != nil {
			return err
		}
	}

	return nil
}

func Valid(path string) bool {
	reg := regexp.MustCompile(`^/dev/nvidia[0-9]*$`)
	check := reg.FindAllString(path, -1)

	return check != nil && check[0] != ""
}

// Initialize the GPU devices, so far only needed to discover the GPU paths.
func (ngm *NvidiaGPUManager) Init(dc dockertools.DockerInterface) error {
	if _, err := os.Stat(NvidiaCtlDevice); err != nil {
		return err
	}

	if _, err := os.Stat(NvidiaUVMDevice); err != nil {
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

// Get how many GPU cards we have.
func (ngm *NvidiaGPUManager) Capacity() int {
	ngm.gpuMutex.Lock()
	defer ngm.gpuMutex.Unlock()

	return len(ngm.gpuPaths)
}

// Check whether the GPU device could be assigned to a container.
func (ngm *NvidiaGPUManager) isAvailable(path string) bool {
	containers, err := dockertools.GetKubeletDockerContainers(ngm.dockerClient, false)

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

// Return the GPU paths as needed, otherwise, return error.
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

	err = fmt.Errorf("Not enough GPUs!")

	return
}

// Return the count of GPUs which are free.
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
