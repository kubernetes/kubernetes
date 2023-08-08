/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/fsnotify/fsnotify"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	plugin "k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/plugin/v1beta1"
)

const (
	resourceName = "example.com/resource"
)

// stubAllocFunc creates and returns allocation response for the input allocate request
func stubAllocFunc(r *pluginapi.AllocateRequest, devs map[string]pluginapi.Device) (*pluginapi.AllocateResponse, error) {
	var responses pluginapi.AllocateResponse
	for _, req := range r.ContainerRequests {
		response := &pluginapi.ContainerAllocateResponse{}
		for _, requestID := range req.DevicesIDs {
			dev, ok := devs[requestID]
			if !ok {
				return nil, fmt.Errorf("invalid allocation request with non-existing device %s", requestID)
			}

			if dev.Health != pluginapi.Healthy {
				return nil, fmt.Errorf("invalid allocation request with unhealthy device: %s", requestID)
			}

			// create fake device file
			fpath := filepath.Join("/tmp", dev.ID)

			// clean first
			if err := os.RemoveAll(fpath); err != nil {
				return nil, fmt.Errorf("failed to clean fake device file from previous run: %s", err)
			}

			f, err := os.Create(fpath)
			if err != nil && !os.IsExist(err) {
				return nil, fmt.Errorf("failed to create fake device file: %s", err)
			}

			f.Close()

			response.Mounts = append(response.Mounts, &pluginapi.Mount{
				ContainerPath: fpath,
				HostPath:      fpath,
			})
		}
		responses.ContainerResponses = append(responses.ContainerResponses, response)
	}

	return &responses, nil
}

func main() {
	devs := []*pluginapi.Device{
		{ID: "Dev-1", Health: pluginapi.Healthy},
		{ID: "Dev-2", Health: pluginapi.Healthy},
	}

	pluginSocksDir := os.Getenv("PLUGIN_SOCK_DIR")
	klog.Infof("pluginSocksDir: %s", pluginSocksDir)
	if pluginSocksDir == "" {
		klog.Errorf("Empty pluginSocksDir")
		return
	}

	socketPath := pluginSocksDir + "/dp." + fmt.Sprintf("%d", time.Now().Unix())

	dp1 := plugin.NewDevicePluginStub(devs, socketPath, resourceName, false, false)
	if err := dp1.Start(); err != nil {
		panic(err)

	}
	dp1.SetAllocFunc(stubAllocFunc)

	if registerControlFile := os.Getenv("REGISTER_CONTROL_FILE"); registerControlFile != "" {
		if err := handleRegistrationProcess(registerControlFile); err != nil {
			panic(err)
		}
	}

	if err := dp1.Register(pluginapi.KubeletSocket, resourceName, pluginapi.DevicePluginPath); err != nil {
		panic(err)
	}
	select {}
}

func handleRegistrationProcess(registerControlFile string) error {
	triggerPath := filepath.Dir(registerControlFile)

	klog.InfoS("Registration process will be managed explicitly", "triggerPath", triggerPath, "triggerEntry", registerControlFile)

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		klog.Errorf("Watcher creation failed: %v ", err)
		return err
	}

	defer watcher.Close()
	updateCh := make(chan bool)
	defer close(updateCh)

	go func() {
		klog.Infof("Starting watching routine")
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				klog.InfoS("Received event", "name", event.Name, "operation", event.Op)
				switch {
				case event.Op&fsnotify.Remove == fsnotify.Remove:
					if event.Name == registerControlFile {
						klog.InfoS("Expected delete", "name", event.Name, "operation", event.Op)
						updateCh <- true
						return
					}
					klog.InfoS("Spurious delete", "name", event.Name, "operation", event.Op)
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				klog.Errorf("error: %w", err)
				panic(err)
			}
		}
	}()

	err = watcher.Add(triggerPath)
	if err != nil {
		klog.Errorf("Failed to add watch to %q: %w", triggerPath, err)
		return err
	}

	klog.InfoS("Waiting for control file to be deleted", "path", registerControlFile)
	<-updateCh
	klog.InfoS("Control file was deleted, connecting!")
	return nil
}
