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
	"os"
	"path/filepath"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

// NewManagerImpl creates a new manager on the socket `socketPath` and can
// rebuild state from devices and available []Device.
// f is the callback that is called when a device becomes unhealthy
// socketPath is present for testing purposes in production this is pluginapi.KubeletSocket
func NewManagerImpl(socketPath string, f MonitorCallback) (*ManagerImpl, error) {
	glog.V(2).Infof("Creating Device Plugin manager at %s", socketPath)

	if socketPath == "" || !filepath.IsAbs(socketPath) {
		return nil, fmt.Errorf(ErrBadSocket+" %v", socketPath)
	}

	dir, file := filepath.Split(socketPath)
	return &ManagerImpl{
		Endpoints: make(map[string]*endpoint),

		socketname: file,
		socketdir:  dir,
		callback:   f,
	}, nil
}

// Start starts the Device Plugin Manager
func (m *ManagerImpl) Start() error {
	glog.V(2).Infof("Starting Device Plugin manager")

	socketPath := filepath.Join(m.socketdir, m.socketname)
	os.MkdirAll(m.socketdir, 0755)

	if err := os.Remove(socketPath); err != nil && !os.IsNotExist(err) {
		glog.Errorf(ErrRemoveSocket+" %+v", err)
		return err
	}

	s, err := net.Listen("unix", socketPath)
	if err != nil {
		glog.Errorf(ErrListenSocket+" %+v", err)
		return err
	}

	m.server = grpc.NewServer([]grpc.ServerOption{}...)

	pluginapi.RegisterRegistrationServer(m.server, m)
	go m.server.Serve(s)

	return nil
}

// Devices is the map of devices that are known by the Device
// Plugin manager with the Kind of the devices as key
func (m *ManagerImpl) Devices() map[string][]*pluginapi.Device {
	glog.V(2).Infof("Devices called")

	m.mutex.Lock()
	defer m.mutex.Unlock()

	devs := make(map[string][]*pluginapi.Device)
	for k, e := range m.Endpoints {
		glog.V(2).Infof("Endpoint: %+v: %+v", k, e)
		e.mutex.Lock()
		devs[k] = copyDevices(e.devices)
		e.mutex.Unlock()
	}

	return devs
}

// Allocate is the call that you can use to allocate a set of Devices
func (m *ManagerImpl) Allocate(resourceName string,
	devs []*pluginapi.Device) (*pluginapi.AllocateResponse, error) {

	m.mutex.Lock()
	defer m.mutex.Unlock()

	if len(devs) == 0 {
		return nil, nil
	}

	glog.Infof("Recieved request for devices %v for device plugin %s",
		devs, resourceName)

	e, ok := m.Endpoints[resourceName]
	if !ok {
		return nil, fmt.Errorf("Unknown Device Plugin %s", resourceName)
	}

	return e.allocate(devs)
}

// Register registers a device plugin
func (m *ManagerImpl) Register(ctx context.Context,
	r *pluginapi.RegisterRequest) (*pluginapi.Empty, error) {

	glog.V(2).Infof("Got request for Device Plugin %s", r.ResourceName)

	if r.Version != pluginapi.Version {
		return &pluginapi.Empty{},
			fmt.Errorf(pluginapi.ErrUnsuportedVersion)
	}

	if err := IsResourceNameValid(r.ResourceName); err != nil {
		return &pluginapi.Empty{}, err
	}

	if _, ok := m.Endpoints[r.ResourceName]; ok {
		return &pluginapi.Empty{},
			fmt.Errorf(pluginapi.ErrDevicePluginAlreadyExists)
	}

	go m.addEndpoint(r)

	return &pluginapi.Empty{}, nil
}

// Stop is the function that can stop the gRPC server
func (m *ManagerImpl) Stop() error {
	for _, e := range m.Endpoints {
		e.stop()
	}

	m.server.Stop()

	return nil
}

func (m *ManagerImpl) addEndpoint(r *pluginapi.RegisterRequest) {
	socketPath := filepath.Join(m.socketdir, r.Endpoint)

	e, err := newEndpoint(socketPath, r.ResourceName, m.callback)
	if err != nil {
		glog.Errorf("Failed to dial device plugin with request %v: %v", r, err)
		return
	}

	stream, err := e.list()
	if err != nil {
		glog.Errorf("Failed to List devices for plugin %v: %v", r.ResourceName, err)
		return
	}

	go func() {
		e.listAndWatch(stream)

		m.mutex.Lock()
		e.mutex.Lock()

		delete(m.Endpoints, r.ResourceName)
		glog.V(2).Infof("Unregistered endpoint %v", e)

		e.mutex.Unlock()
		m.mutex.Unlock()
	}()

	m.mutex.Lock()
	e.mutex.Lock()

	m.Endpoints[r.ResourceName] = e
	glog.V(2).Infof("Registered endpoint %v", e)

	e.mutex.Unlock()
	m.mutex.Unlock()

}
