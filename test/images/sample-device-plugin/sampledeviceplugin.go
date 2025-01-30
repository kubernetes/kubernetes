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
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	plugin "k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/plugin/v1beta1"
)

const (
	resourceName = "example.com/resource"
	cdiPath      = "/var/run/cdi/example.com.json"
	cdiVersion   = "0.3.0"
	cdiPrefix    = "CDI-"
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

			if os.Getenv("CDI_ENABLED") != "" {
				// add the CDI device ID to the response.
				cdiDevice := &pluginapi.CDIDevice{
					Name: fmt.Sprintf("%s=%s", resourceName, cdiPrefix+dev.ID),
				}
				response.CDIDevices = append(response.CDIDevices, cdiDevice)
			}
		}
		responses.ContainerResponses = append(responses.ContainerResponses, response)
	}

	return &responses, nil
}

// stubAllocFunc creates and returns allocation response for the input allocate request
func stubRegisterControlFunc() bool {
	return false
}

func main() {
	// respond to syscalls for termination
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	devs := []*pluginapi.Device{
		{ID: "Dev-1", Health: pluginapi.Healthy},
		{ID: "Dev-2", Health: pluginapi.Healthy},
	}

	pluginSocksDir := os.Getenv("PLUGIN_SOCK_DIR")
	klog.Infof("pluginSocksDir: %s", pluginSocksDir)
	if pluginSocksDir == "" {
		pluginSocksDir = pluginapi.DevicePluginPath
	}

	socketPath := pluginSocksDir + "/dp." + fmt.Sprintf("%d", time.Now().Unix())

	dp1 := plugin.NewDevicePluginStub(devs, socketPath, resourceName, false, false)
	if err := dp1.Start(); err != nil {
		panic(err)

	}
	dp1.SetAllocFunc(stubAllocFunc)

	cdiEnabled := os.Getenv("CDI_ENABLED")
	klog.Infof("CDI_ENABLED: %s", cdiEnabled)
	if cdiEnabled != "" {
		if err := createCDIFile(devs); err != nil {
			panic(err)
		}
		defer func() {
			// Remove CDI file
			if _, err := os.Stat(cdiPath); err == nil || os.IsExist(err) {
				err := os.Remove(cdiPath)
				if err != nil {
					panic(err)
				}
			}
		}()
	}

	var registerControlFile string
	autoregister := true

	if registerControlFile = os.Getenv("REGISTER_CONTROL_FILE"); registerControlFile != "" {
		autoregister = false
		dp1.SetRegisterControlFunc(stubRegisterControlFunc)
	}

	if !autoregister {
		go dp1.Watch(pluginapi.KubeletSocket, resourceName, pluginapi.DevicePluginPath)

		triggerPath := filepath.Dir(registerControlFile)

		klog.InfoS("Registration process will be managed explicitly", "triggerPath", triggerPath, "triggerEntry", registerControlFile)

		watcher, err := fsnotify.NewWatcher()
		if err != nil {
			klog.Errorf("Watcher creation failed: %v ", err)
			panic(err)
		}
		defer watcher.Close()

		updateCh := make(chan bool)
		defer close(updateCh)

		go handleRegistrationProcess(registerControlFile, dp1, watcher, updateCh)

		err = watcher.Add(triggerPath)
		if err != nil {
			klog.Errorf("Failed to add watch to %q: %v", triggerPath, err)
			panic(err)
		}
		for {
			select {
			case received := <-updateCh:
				if received {
					if err := dp1.Register(pluginapi.KubeletSocket, resourceName, pluginapi.DevicePluginPath); err != nil {
						panic(err)
					}
					klog.InfoS("Control file was deleted, registration succeeded")
				}
			// Catch termination signals
			case sig := <-sigCh:
				klog.InfoS("Shutting down, received signal", "signal", sig)
				if err := dp1.Stop(); err != nil {
					panic(err)
				}
				return
			}
			time.Sleep(5 * time.Second)
		}
	} else {
		if err := dp1.Register(pluginapi.KubeletSocket, resourceName, pluginapi.DevicePluginPath); err != nil {
			panic(err)
		}

		go dp1.Watch(pluginapi.KubeletSocket, resourceName, pluginapi.DevicePluginPath)
		// Catch termination signals
		sig := <-sigCh
		klog.InfoS("Shutting down, received signal", "signal", sig)
		if err := dp1.Stop(); err != nil {
			panic(err)
		}
		return
	}
}

func handleRegistrationProcess(registerControlFile string, dpStub *plugin.Stub, watcher *fsnotify.Watcher, updateCh chan<- bool) {
	klog.InfoS("Starting watching routine")
	for {
		klog.InfoS("handleRegistrationProcess for loop")
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}
			klog.InfoS("Received event", "name", event.Name, "operation", event.Op)
			if event.Op&fsnotify.Remove == fsnotify.Remove {
				if event.Name == registerControlFile {
					klog.InfoS("Expected delete", "name", event.Name, "operation", event.Op)
					updateCh <- true
					continue
				}
				klog.InfoS("Spurious delete", "name", event.Name, "operation", event.Op)
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			klog.ErrorS(err, "error")
			panic(err)
		default:
			time.Sleep(5 * time.Second)
		}
	}
}

func createCDIFile(devs []*pluginapi.Device) error {
	content := fmt.Sprintf(`{"cdiVersion":"%s","kind":"%s","devices":[`, cdiVersion, resourceName)
	for i, dev := range devs {
		name := cdiPrefix + dev.ID
		content += fmt.Sprintf(`{"name":"%s","containerEdits":{"env":["CDI_DEVICE=%s"],"deviceNodes":[{"path":"/tmp/%s","type":"b","major":1,"minor":%d}]}}`, name, name, name, i)
		if i < len(devs)-1 {
			content += ","
		}
	}
	content += "]}"
	if err := os.WriteFile(cdiPath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to create CDI file: %s", err)
	}
	klog.InfoS("Created CDI file", "path", cdiPath, "devices", devs)
	return nil
}
