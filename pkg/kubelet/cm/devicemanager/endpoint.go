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

package devicemanager

import (
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
)

// endpoint maps to a single registered device plugin. It is responsible
// for managing gRPC communications with the device plugin and caching
// device states reported by the device plugin.
type endpoint interface {
	run()
	stop()
	allocate(devs []string) (*pluginapi.AllocateResponse, error)
	preStartContainer(devs []string) (*pluginapi.PreStartContainerResponse, error)
	getDevices() []pluginapi.Device
	callback(resourceName string, added, updated, deleted []pluginapi.Device)
	isStopped() bool
	stopGracePeriodExpired() bool
}

type endpointImpl struct {
	client     pluginapi.DevicePluginClient
	clientConn *grpc.ClientConn

	socketPath   string
	resourceName string
	stopTime     time.Time

	devices map[string]pluginapi.Device
	mutex   sync.Mutex

	cb monitorCallback
}

// newEndpoint creates a new endpoint for the given resourceName.
// This is to be used during normal device plugin registration.
func newEndpointImpl(socketPath, resourceName string, devices map[string]pluginapi.Device, callback monitorCallback) (*endpointImpl, error) {
	client, c, err := dial(socketPath)
	if err != nil {
		glog.Errorf("Can't create new endpoint with path %s err %v", socketPath, err)
		return nil, err
	}

	return &endpointImpl{
		client:     client,
		clientConn: c,

		socketPath:   socketPath,
		resourceName: resourceName,

		devices: devices,
		cb:      callback,
	}, nil
}

// newStoppedEndpointImpl creates a new endpoint for the given resourceName with stopTime set.
// This is to be used during Kubelet restart, before the actual device plugin re-registers.
func newStoppedEndpointImpl(resourceName string, devices map[string]pluginapi.Device) *endpointImpl {
	return &endpointImpl{
		resourceName: resourceName,
		devices:      devices,
		stopTime:     time.Now(),
	}
}

func (e *endpointImpl) callback(resourceName string, added, updated, deleted []pluginapi.Device) {
	e.cb(resourceName, added, updated, deleted)
}

func (e *endpointImpl) getDevices() []pluginapi.Device {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	var devs []pluginapi.Device

	for _, d := range e.devices {
		devs = append(devs, d)
	}

	return devs
}

// run initializes ListAndWatch gRPC call for the device plugin and
// blocks on receiving ListAndWatch gRPC stream updates. Each ListAndWatch
// stream update contains a new list of device states. listAndWatch compares the new
// device states with its cached states to get list of new, updated, and deleted devices.
// It then issues a callback to pass this information to the device manager which
// will adjust the resource available information accordingly.
func (e *endpointImpl) run() {
	stream, err := e.client.ListAndWatch(context.Background(), &pluginapi.Empty{})
	if err != nil {
		glog.Errorf(errListAndWatch, e.resourceName, err)

		return
	}

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
		// NOTE: Return a copy of 'devices' instead of returning a direct reference to local 'devices'
		e.devices = make(map[string]pluginapi.Device)
		for _, d := range devices {
			e.devices[d.ID] = d
		}
		e.mutex.Unlock()

		e.callback(e.resourceName, added, updated, deleted)
	}
}

func (e *endpointImpl) isStopped() bool {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	return !e.stopTime.IsZero()
}

func (e *endpointImpl) stopGracePeriodExpired() bool {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	return !e.stopTime.IsZero() && time.Since(e.stopTime) > endpointStopGracePeriod
}

// used for testing only
func (e *endpointImpl) setStopTime(t time.Time) {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	e.stopTime = t
}

// allocate issues Allocate gRPC call to the device plugin.
func (e *endpointImpl) allocate(devs []string) (*pluginapi.AllocateResponse, error) {
	if e.isStopped() {
		return nil, fmt.Errorf(errEndpointStopped, e)
	}
	return e.client.Allocate(context.Background(), &pluginapi.AllocateRequest{
		ContainerRequests: []*pluginapi.ContainerAllocateRequest{
			{DevicesIDs: devs},
		},
	})
}

// preStartContainer issues PreStartContainer gRPC call to the device plugin.
func (e *endpointImpl) preStartContainer(devs []string) (*pluginapi.PreStartContainerResponse, error) {
	if e.isStopped() {
		return nil, fmt.Errorf(errEndpointStopped, e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), pluginapi.KubeletPreStartContainerRPCTimeoutInSecs*time.Second)
	defer cancel()
	return e.client.PreStartContainer(ctx, &pluginapi.PreStartContainerRequest{
		DevicesIDs: devs,
	})
}

func (e *endpointImpl) stop() {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	if e.clientConn != nil {
		e.clientConn.Close()
	}
	e.stopTime = time.Now()
}

// dial establishes the gRPC communication with the registered device plugin. https://godoc.org/google.golang.org/grpc#Dial
func dial(unixSocketPath string) (pluginapi.DevicePluginClient, *grpc.ClientConn, error) {
	c, err := grpc.Dial(unixSocketPath, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithTimeout(10*time.Second),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, nil, fmt.Errorf(errFailedToDialDevicePlugin+" %v", err)
	}

	return pluginapi.NewDevicePluginClient(c), c, nil
}
