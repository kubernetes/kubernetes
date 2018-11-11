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
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"k8s.io/klog"

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
	callback(resourceName string, devices []pluginapi.Device)
	isStopped() bool
	stopGracePeriodExpired() bool
}

type endpointImpl struct {
	client     pluginapi.DevicePluginClient
	clientConn *grpc.ClientConn

	socketPath   string
	resourceName string
	stopTime     time.Time

	mutex sync.Mutex
	cb    monitorCallback
}

// newEndpoint creates a new endpoint for the given resourceName.
// This is to be used during normal device plugin registration.
func newEndpointImpl(socketPath, resourceName string, callback monitorCallback) (*endpointImpl, error) {
	client, c, err := dial(socketPath)
	if err != nil {
		klog.Errorf("Can't create new endpoint with path %s err %v", socketPath, err)
		return nil, err
	}

	return &endpointImpl{
		client:     client,
		clientConn: c,

		socketPath:   socketPath,
		resourceName: resourceName,

		cb: callback,
	}, nil
}

// newStoppedEndpointImpl creates a new endpoint for the given resourceName with stopTime set.
// This is to be used during Kubelet restart, before the actual device plugin re-registers.
func newStoppedEndpointImpl(resourceName string) *endpointImpl {
	return &endpointImpl{
		resourceName: resourceName,
		stopTime:     time.Now(),
	}
}

func (e *endpointImpl) callback(resourceName string, devices []pluginapi.Device) {
	e.cb(resourceName, devices)
}

// run initializes ListAndWatch gRPC call for the device plugin and
// blocks on receiving ListAndWatch gRPC stream updates. Each ListAndWatch
// stream update contains a new list of device states.
// It then issues a callback to pass this information to the device manager which
// will adjust the resource available information accordingly.
func (e *endpointImpl) run() {
	stream, err := e.client.ListAndWatch(context.Background(), &pluginapi.Empty{})
	if err != nil {
		klog.Errorf(errListAndWatch, e.resourceName, err)

		return
	}

	for {
		response, err := stream.Recv()
		if err != nil {
			klog.Errorf(errListAndWatch, e.resourceName, err)
			return
		}

		devs := response.Devices
		klog.V(2).Infof("State pushed for device plugin %s", e.resourceName)

		var newDevs []pluginapi.Device
		for _, d := range devs {
			newDevs = append(newDevs, *d)
		}

		e.callback(e.resourceName, newDevs)
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
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	c, err := grpc.DialContext(ctx, unixSocketPath, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, nil, fmt.Errorf(errFailedToDialDevicePlugin+" %v", err)
	}

	return pluginapi.NewDevicePluginClient(c), c, nil
}
