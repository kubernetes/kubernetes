/*
Copyright 2017 The Kubernetes Authors.

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

package deviceplugin

import (
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
)

// endpoint maps to a single registered device plugin. It is responsible
// for managing gRPC communications with the device plugin and caching
// device states reported by the device plugin.
type endpoint struct {
	client     pluginapi.DevicePluginClient
	clientConn *grpc.ClientConn

	socketPath   string
	resourceName string

	devices map[string]pluginapi.Device
	mutex   sync.Mutex

	callback MonitorCallback
}

// newEndpoint creates a new endpoint for the given resourceName.
func newEndpoint(socketPath, resourceName string, callback MonitorCallback) (*endpoint, error) {
	client, c, err := dial(socketPath)
	if err != nil {
		glog.Errorf("Can't create new endpoint with path %s err %v", socketPath, err)
		return nil, err
	}

	return &endpoint{
		client:     client,
		clientConn: c,

		socketPath:   socketPath,
		resourceName: resourceName,

		devices:  nil,
		callback: callback,
	}, nil
}

func (e *endpoint) getDevices() []pluginapi.Device {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	var devs []pluginapi.Device

	for _, d := range e.devices {
		devs = append(devs, d)
	}

	return devs
}

// list initializes ListAndWatch gRPC call for the device plugin and gets the
// initial list of the devices. Returns ListAndWatch gRPC stream on success.
func (e *endpoint) list() (pluginapi.DevicePlugin_ListAndWatchClient, error) {
	stream, err := e.client.ListAndWatch(context.Background(), &pluginapi.Empty{})
	if err != nil {
		glog.Errorf(errListAndWatch, e.resourceName, err)
		return nil, err
	}

	devs, err := stream.Recv()
	if err != nil {
		glog.Errorf(errListAndWatch, e.resourceName, err)
		return nil, err
	}

	devices := make(map[string]pluginapi.Device)
	var added, updated, deleted []pluginapi.Device
	for _, d := range devs.Devices {
		devices[d.ID] = *d
		added = append(added, *d)
	}

	e.mutex.Lock()
	e.devices = devices
	e.mutex.Unlock()

	e.callback(e.resourceName, added, updated, deleted)

	return stream, nil
}

// listAndWatch blocks on receiving ListAndWatch gRPC stream updates. Each ListAndWatch
// stream update contains a new list of device states. listAndWatch compares the new
// device states with its cached states to get list of new, updated, and deleted devices.
// It then issues a callback to pass this information to the device_plugin_handler which
// will adjust the resource available information accordingly.
func (e *endpoint) listAndWatch(stream pluginapi.DevicePlugin_ListAndWatchClient) {
	glog.V(3).Infof("Starting ListAndWatch")

	devices := make(map[string]pluginapi.Device)

	e.mutex.Lock()
	for _, d := range e.devices {
		devices[d.ID] = d
	}
	e.mutex.Unlock()

	for {
		response, err := stream.Recv()
		if err != nil {
			glog.Errorf(errListAndWatch, e.resourceName, err)
			return
		}

		devs := response.Devices
		glog.V(2).Infof("State pushed for device plugin %s", e.resourceName)

		newDevs := make(map[string]*pluginapi.Device)
		var added, updated []pluginapi.Device

		for _, d := range devs {
			dOld, ok := devices[d.ID]
			newDevs[d.ID] = d

			if !ok {
				glog.V(2).Infof("New device for Endpoint %s: %v", e.resourceName, d)

				devices[d.ID] = *d
				added = append(added, *d)

				continue
			}

			if d.Health == dOld.Health {
				continue
			}

			if d.Health == pluginapi.Unhealthy {
				glog.Errorf("Device %s is now Unhealthy", d.ID)
			} else if d.Health == pluginapi.Healthy {
				glog.V(2).Infof("Device %s is now Healthy", d.ID)
			}

			devices[d.ID] = *d
			updated = append(updated, *d)
		}

		var deleted []pluginapi.Device
		for id, d := range devices {
			if _, ok := newDevs[id]; ok {
				continue
			}

			glog.Errorf("Device %s was deleted", d.ID)

			deleted = append(deleted, d)
			delete(devices, id)
		}

		e.mutex.Lock()
		e.devices = devices
		e.mutex.Unlock()

		e.callback(e.resourceName, added, updated, deleted)
	}
}

// allocate issues Allocate gRPC call to the device plugin.
func (e *endpoint) allocate(devs []string) (*pluginapi.AllocateResponse, error) {
	return e.client.Allocate(context.Background(), &pluginapi.AllocateRequest{
		DevicesIDs: devs,
	})
}

func (e *endpoint) stop() {
	e.clientConn.Close()
}

// dial establishes the gRPC communication with the registered device plugin.
func dial(unixSocketPath string) (pluginapi.DevicePluginClient, *grpc.ClientConn, error) {
	c, err := grpc.Dial(unixSocketPath, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, nil, fmt.Errorf(errFailedToDialDevicePlugin+" %v", err)
	}

	return pluginapi.NewDevicePluginClient(c), c, nil
}
