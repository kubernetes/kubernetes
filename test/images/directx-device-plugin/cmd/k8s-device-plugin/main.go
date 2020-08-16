/*
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

package main

import (
	"flag"
	"fmt"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/images/directx-device-plugin/pkg/gpu-detection"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	dm "k8s.io/kubernetes/pkg/kubelet/cm/devicemanager"
	"os"
	"path"
	"time"
)

const (
	resourceName = "microsoft.com/directx"
	deviceClass  = "class/5B45201D-F2F2-4F3B-85BB-30FF1F953599"
)

// stubAllocFunc creates and returns allocation response for the input allocate request
func allocFunc(r *pluginapi.AllocateRequest, devs map[string]pluginapi.Device) (*pluginapi.AllocateResponse, error) {
	var responses pluginapi.AllocateResponse
	for _, req := range r.ContainerRequests {
		response := &pluginapi.ContainerAllocateResponse{}
		for _, requestID := range req.DevicesIDs {
			gpu := gpu_detection.GetGPUInfo(requestID)
			if gpu == nil {
				return nil, fmt.Errorf("invalid allocation request with non-existing device %s", requestID)
			}

			if getGPUHealth(gpu) != pluginapi.Healthy {
				return nil, fmt.Errorf("invalid allocation request with unhealthy device: %s", requestID)
			}

			response.Devices = append(response.Devices, &pluginapi.DeviceSpec{
				HostPath:      deviceClass,
				ContainerPath: "",
				Permissions:   "",
			})
			if response.Envs == nil {
				response.Envs = make(map[string]string)
			}
			response.Envs["DIRECTX_GPU_Name"] = gpu.Name
			response.Envs["DIRECTX_GPU_PNPDeviceID"] = gpu.PNPDeviceID
			response.Envs["DIRECTX_GPU_DriverVersion"] = gpu.DriverVersion
		}
		responses.ContainerResponses = append(responses.ContainerResponses, response)
	}

	return &responses, nil
}

func getGPUHealth(gpu *gpu_detection.GPUInfo) string {
	if gpu.IsStatusOK() {
		return pluginapi.Healthy
	}
	return pluginapi.Unhealthy
}

func main() {
	flag.Set("alsologtostderr", "true")
	devs := []*pluginapi.Device{}
	gpus:= gpu_detection.GetGPUList()
	pluginSocksDir := os.Getenv("PLUGIN_SOCK_DIR")
	if pluginSocksDir == "" {
		pluginSocksDir = pluginapi.DevicePluginPath
	}

	gpuMatchName := os.Getenv("DIRECTX_GPU_MATCH_NAME")
	if gpuMatchName == "" {
		gpuMatchName = "nvidia"
	}

	for _, gpuInfo := range gpus{
		if !gpuInfo.MatchName(gpuMatchName) {
			klog.Warningf("'%s' doesn't match  '%s', ignoring this gpu", gpuInfo.Name, gpuMatchName)
			continue
		}

		devs = append(devs, &pluginapi.Device{
			ID: gpuInfo.PNPDeviceID,
			Health: getGPUHealth(&gpuInfo),
		})
		klog.Infof("GPU %s id: %s", gpuInfo.Name, gpuInfo.PNPDeviceID)
	}

	klog.Infof("pluginSocksDir: %s", pluginSocksDir)
	socketPath := path.Join(pluginSocksDir, "directx.sock")
	klog.Infof("socketPath: %s", socketPath)
	dp1 := dm.NewDevicePluginStub(devs, socketPath, resourceName, false)
	if err := dp1.Start(); err != nil {
		panic(err)

	}

	dp1.SetAllocFunc(allocFunc)

	// todo: when kubelet will success to autodetect socket, change the pluginSockDir to detect DEPRECATION file
	if err := dp1.Register(pluginapi.KubeletSocket, resourceName, ""); err != nil {
		panic(err)
	}

	for {
		time.Sleep(time.Second*10)
		if _, err := os.Stat(socketPath); os.IsNotExist(err) {
			// exit if the socketPath is missing, cause by kubelet restart, we need to start again the plugin
			os.Exit(1)
		}
	}
}
