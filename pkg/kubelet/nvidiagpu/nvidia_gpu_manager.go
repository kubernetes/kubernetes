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
	"sync"
	"os"
	"path/filepath"
	"regex"

	"k8s.io/kubernetes/pkg/api/v1"
)


// TODO: If use NVML in the future, the implementation could be more complex,
// but also more powerful!

const (
	// All NVIDIA GPUs cards should be mounted with nvidiactl and nvidia-uvm
	NvidiaDeviceCtl string = "/dev/nvidiactl"
	NvidiaDeviceUVM string = "/dev/nvidia-uvm"
)

type NvidiaGPU struct {
	gpuPaths []string
	gpuMutex sync.Mutex
}

// Get all the paths of NVIDIA GPU card from /dev/
func (nvidiaGPU *NvidiaGPU) discovery() error {
	var err error
	if gpuPaths == nil {
		err = filepath.Walk("/dev", func(path string, f os.FileInfo, err error) error {
			reg := regexp.MustCompile(`^nvidia[0-9]*$`)
			gpupath := reg.FindAllString(f.Name(), -1)
			if gpupath != nil && gpupath[0] != "" {
				nvidiaGPU.gpuPaths = append(gpuPaths, "/dev/"+gpupath[0])
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

func Init() error {
	if _, err := os.Stat(NvidiaDeviceCtl); err != nil {
		return err
	}
	if _, err := os.Stat(NvidiaDeviceUVM); err != nil {
		return err
	}

	err := discovery()

	return err
}

func (nvidiaGPU *NvidiaGPU) Init() error {
	nvidiaGPU.gpuMutex.Lock()
	defer nvidiaGPU.gpuMutex.Unlock()

	return discovery()
}

func (nvidiaGPU *NvidiaGPU) Shutdown() {
	nvidiaGPU.gpuMutex.Lock()
	defer nvidiaGPU.gpuMutex.Unlock()

	nvidiaGPU.gpugpuPaths = nil
}

func (nvidiaGPU *NvidiaGPU) Capacity() int {
	nvidiaGPU.gpuMutex.Lock()
	defer nvidiaGPU.gpuMutex.Unlock()

	return len(nvidiaGPU.gpuPaths)
}

func (nvidiaGPU *NvidiaGPU) isAvailable(path string) bool {
	for _, container range containers {
		for _, device range container.Devices {
			if device.PathOnHost == path {
				return false
			}
		}
	}

	return true
}

func (nvidiaGPU *NvidiaGPU) AllocateGPUs(int num) (paths []string) {
	if num <= 0 {
		return nil
	}

	for _, path range nvidiaGPU.gpuPaths {
		if nvidiaGPU.isAvailable(path) {
			paths = append(paths, path)
			if len(paths) == num {
			return
			}
		}
	}
}

func (nvidiaGPU *NvidiaGPU) AvailableGPUs() (num int) {
	for path range nvidiaGPU.gpuPaths {
		if nvidiaGPU.isAvailable(path) {
			num++
		}
	}
}

