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

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

type endpoint struct {
	client pluginapi.DevicePluginClient

	socketPath   string
	resourceName string

	devices map[string]*pluginapi.Device
	mutex   sync.Mutex

	callback MonitorCallback

	cancel context.CancelFunc
	ctx    context.Context
}

func newEndpoint(socketPath, resourceName string, callback MonitorCallback) (*endpoint, error) {
	client, err := dial(socketPath)
	if err != nil {
		return nil, err
	}

	ctx, stop := context.WithCancel(context.Background())

	return &endpoint{
		client: client,

		socketPath:   socketPath,
		resourceName: resourceName,

		devices:  nil,
		callback: callback,

		cancel: stop,
		ctx:    ctx,
	}, nil
}

func (e *endpoint) list() (pluginapi.DevicePlugin_ListAndWatchClient, error) {
	glog.V(2).Infof("Starting ListAndWatch")

	stream, err := e.client.ListAndWatch(e.ctx, &pluginapi.Empty{})
	if err != nil {
		glog.Errorf(ErrListAndWatch, e.resourceName, err)

		return nil, err
	}

	devs, err := stream.Recv()
	if err != nil {
		glog.Errorf(ErrListAndWatch, e.resourceName, err)
		return nil, err
	}

	devices := make(map[string]*pluginapi.Device)
	for _, d := range devs.Devices {
		devices[d.ID] = d
	}

	e.mutex.Lock()
	e.devices = devices
	e.mutex.Unlock()

	return stream, nil
}

func (e *endpoint) listAndWatch(stream pluginapi.DevicePlugin_ListAndWatchClient) {
	glog.V(2).Infof("Starting ListAndWatch")

	devices := make(map[string]*pluginapi.Device)

	e.mutex.Lock()
	for _, d := range e.devices {
		devices[d.ID] = CloneDevice(d)
	}
	e.mutex.Unlock()

	for {
		response, err := stream.Recv()
		if err != nil {
			glog.Errorf(ErrListAndWatch, e.resourceName, err)
			return
		}

		devs := response.Devices
		glog.V(2).Infof("State pushed for device plugin %s", e.resourceName)

		newDevs := make(map[string]*pluginapi.Device)
		var added, updated []*pluginapi.Device

		for _, d := range devs {
			dOld, ok := devices[d.ID]
			newDevs[d.ID] = d

			if !ok {
				glog.V(2).Infof("New device for Endpoint %s: %v", e.resourceName, d)

				devices[d.ID] = d
				added = append(added, CloneDevice(d))

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

			devices[d.ID] = d
			updated = append(updated, CloneDevice(d))
		}

		var deleted []*pluginapi.Device
		for id, d := range devices {
			if _, ok := newDevs[id]; ok {
				continue
			}

			glog.Errorf("Device %s was deleted", d.ID)

			deleted = append(deleted, CloneDevice(d))
			delete(devices, id)
		}

		e.mutex.Lock()
		e.devices = devices
		e.mutex.Unlock()

		e.callback(e.resourceName, added, updated, deleted)
	}

}

func (e *endpoint) allocate(devs []*pluginapi.Device) (*pluginapi.AllocateResponse, error) {
	var ids []string
	for _, d := range devs {
		ids = append(ids, d.ID)
	}

	return e.client.Allocate(context.Background(), &pluginapi.AllocateRequest{
		DevicesIDs: ids,
	})
}

func (e *endpoint) stop() {
	e.cancel()
}

func dial(unixSocketPath string) (pluginapi.DevicePluginClient, error) {
	c, err := grpc.Dial(unixSocketPath, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, fmt.Errorf(pluginapi.ErrFailedToDialDevicePlugin+" %v", err)
	}

	return pluginapi.NewDevicePluginClient(c), nil
}
