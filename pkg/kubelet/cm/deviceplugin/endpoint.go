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
type endpoint interface {
	Run()
	Stop() error

	Allocate(devs []string) (*pluginapi.AllocateResponse, error)
	ResourceName() string

	Store() deviceStore
	SetStore(deviceStore)
}

type endpointImpl struct {
	client     pluginapi.DevicePluginClient
	clientConn *grpc.ClientConn

	socketPath   string
	resourceName string

	stopChan chan interface{}
	ctx      context.Context
	cancel   context.CancelFunc

	sync.Mutex
	devStore deviceStore
}

// newEndpoint creates a new endpoint for the given resourceName.
func newEndpoint(socketPath, resourceName string) (*endpointImpl, error) {
	return newEndpointWithStore(socketPath, resourceName, nil)
}

func newEndpointWithStore(socketPath, resourceName string, devStore deviceStore) (*endpointImpl, error) {
	client, c, err := dial(socketPath)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &endpointImpl{
		client:     client,
		clientConn: c,

		socketPath:   socketPath,
		resourceName: resourceName,

		ctx:    ctx,
		cancel: cancel,

		devStore: devStore,
	}, nil
}

func (e *endpointImpl) Store() deviceStore {
	e.Lock()
	defer e.Unlock()

	return e.devStore
}

func (e *endpointImpl) SetStore(s deviceStore) {
	e.Lock()
	defer e.Unlock()

	e.devStore = s
}

func (e *endpointImpl) ResourceName() string {
	return e.resourceName
}

// run initializes ListAndWatch gRPC call for the device plugin and
// blocks on receiving ListAndWatch gRPC stream updates. Each ListAndWatch
// stream update contains a new list of device states. listAndWatch compares the new
// device states with its cached states to get list of new, updated, and deleted devices.
// It then issues a callback to pass this information to the device manager which
// will adjust the resource available information accordingly.
func (e *endpointImpl) Run() {
	glog.V(3).Infof("Starting to run endpoint %s", e.resourceName)
	e.stopChan = make(chan interface{}, 0)

	stream, err := e.client.ListAndWatch(e.ctx, &pluginapi.Empty{})
	if err != nil {
		glog.Errorf(errListAndWatch, e.resourceName, err)
		return
	}

	for {
		response, err := stream.Recv()
		if err != nil {
			glog.Errorf("Stopping to receive from Endpoint")
			s := e.Store()

			s.Callback(e.resourceName, nil, nil, s.Devices())
			e.clientConn.Close()

			glog.Errorf(errListAndWatch, e.resourceName, err)

			close(e.stopChan)
			return
		}

		glog.V(2).Infof("Endpoint %s updated", e.resourceName)

		s := e.Store()
		added, updated, deleted := s.Update(response.Devices)
		s.Callback(e.resourceName, added, updated, deleted)
	}
}

// allocate issues Allocate gRPC call to the device plugin.
func (e *endpointImpl) Allocate(devs []string) (*pluginapi.AllocateResponse, error) {
	return e.client.Allocate(context.Background(), &pluginapi.AllocateRequest{
		DevicesIDs: devs,
	})
}

func (e *endpointImpl) Stop() error {
	if e.stopChan == nil {
		e.cancel()
		e.clientConn.Close()

		return nil
	}

	e.cancel()
	e.clientConn.Close()

	select {
	case <-e.stopChan:
		break
	case <-time.After(5 * time.Second):
		return fmt.Errorf("Could not stop endpoint %s", e.resourceName)
	}

	return nil
}

// dial establishes the gRPC communication with the registered device plugin.
func dial(unixSocketPath string) (pluginapi.DevicePluginClient, *grpc.ClientConn, error) {
	c, err := grpc.Dial(unixSocketPath, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, nil, fmt.Errorf(errFailedToDialDevicePlugin+" %v", err)
	}

	return pluginapi.NewDevicePluginClient(c), c, nil
}
