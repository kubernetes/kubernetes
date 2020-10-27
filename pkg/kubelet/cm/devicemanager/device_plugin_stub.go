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
	"net"
	"os"
	"path"
	"sync"
	"time"

	"google.golang.org/grpc"

	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	watcherapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
)

// Stub implementation for DevicePlugin.
type Stub struct {
	devs                       []*pluginapi.Device
	socket                     string
	resourceName               string
	preStartContainerFlag      bool
	getPreferredAllocationFlag bool

	stop   chan interface{}
	wg     sync.WaitGroup
	update chan []*pluginapi.Device

	server *grpc.Server

	// allocFunc is used for handling allocation request
	allocFunc stubAllocFunc

	// getPreferredAllocFunc is used for handling getPreferredAllocation request
	getPreferredAllocFunc stubGetPreferredAllocFunc

	registrationStatus chan watcherapi.RegistrationStatus // for testing
	endpoint           string                             // for testing

}

// stubGetPreferredAllocFunc is the function called when a getPreferredAllocation request is received from Kubelet
type stubGetPreferredAllocFunc func(r *pluginapi.PreferredAllocationRequest, devs map[string]pluginapi.Device) (*pluginapi.PreferredAllocationResponse, error)

func defaultGetPreferredAllocFunc(r *pluginapi.PreferredAllocationRequest, devs map[string]pluginapi.Device) (*pluginapi.PreferredAllocationResponse, error) {
	var response pluginapi.PreferredAllocationResponse

	return &response, nil
}

// stubAllocFunc is the function called when an allocation request is received from Kubelet
type stubAllocFunc func(r *pluginapi.AllocateRequest, devs map[string]pluginapi.Device) (*pluginapi.AllocateResponse, error)

func defaultAllocFunc(r *pluginapi.AllocateRequest, devs map[string]pluginapi.Device) (*pluginapi.AllocateResponse, error) {
	var response pluginapi.AllocateResponse

	return &response, nil
}

// NewDevicePluginStub returns an initialized DevicePlugin Stub.
func NewDevicePluginStub(devs []*pluginapi.Device, socket string, name string, preStartContainerFlag bool, getPreferredAllocationFlag bool) *Stub {
	return &Stub{
		devs:                       devs,
		socket:                     socket,
		resourceName:               name,
		preStartContainerFlag:      preStartContainerFlag,
		getPreferredAllocationFlag: getPreferredAllocationFlag,

		stop:   make(chan interface{}),
		update: make(chan []*pluginapi.Device),

		allocFunc:             defaultAllocFunc,
		getPreferredAllocFunc: defaultGetPreferredAllocFunc,
	}
}

// SetGetPreferredAllocFunc sets allocFunc of the device plugin
func (m *Stub) SetGetPreferredAllocFunc(f stubGetPreferredAllocFunc) {
	m.getPreferredAllocFunc = f
}

// SetAllocFunc sets allocFunc of the device plugin
func (m *Stub) SetAllocFunc(f stubAllocFunc) {
	m.allocFunc = f
}

// Start starts the gRPC server of the device plugin. Can only
// be called once.
func (m *Stub) Start() error {
	err := m.cleanup()
	if err != nil {
		return err
	}

	sock, err := net.Listen("unix", m.socket)
	if err != nil {
		return err
	}

	m.wg.Add(1)
	m.server = grpc.NewServer([]grpc.ServerOption{}...)
	pluginapi.RegisterDevicePluginServer(m.server, m)
	watcherapi.RegisterRegistrationServer(m.server, m)

	go func() {
		defer m.wg.Done()
		m.server.Serve(sock)
	}()
	_, conn, err := dial(m.socket)
	if err != nil {
		return err
	}
	conn.Close()
	klog.Infof("Starting to serve on %v", m.socket)

	return nil
}

// Stop stops the gRPC server. Can be called without a prior Start
// and more than once. Not safe to be called concurrently by different
// goroutines!
func (m *Stub) Stop() error {
	if m.server == nil {
		return nil
	}
	m.server.Stop()
	m.wg.Wait()
	m.server = nil
	close(m.stop) // This prevents re-starting the server.

	return m.cleanup()
}

// GetInfo is the RPC which return pluginInfo
func (m *Stub) GetInfo(ctx context.Context, req *watcherapi.InfoRequest) (*watcherapi.PluginInfo, error) {
	klog.Info("GetInfo")
	return &watcherapi.PluginInfo{
		Type:              watcherapi.DevicePlugin,
		Name:              m.resourceName,
		Endpoint:          m.endpoint,
		SupportedVersions: []string{pluginapi.Version}}, nil
}

// NotifyRegistrationStatus receives the registration notification from watcher
func (m *Stub) NotifyRegistrationStatus(ctx context.Context, status *watcherapi.RegistrationStatus) (*watcherapi.RegistrationStatusResponse, error) {
	if m.registrationStatus != nil {
		m.registrationStatus <- *status
	}
	if !status.PluginRegistered {
		klog.Infof("Registration failed: %v", status.Error)
	}
	return &watcherapi.RegistrationStatusResponse{}, nil
}

// Register registers the device plugin for the given resourceName with Kubelet.
func (m *Stub) Register(kubeletEndpoint, resourceName string, pluginSockDir string) error {
	if pluginSockDir != "" {
		if _, err := os.Stat(pluginSockDir + "DEPRECATION"); err == nil {
			klog.Info("Deprecation file found. Skip registration.")
			return nil
		}
	}
	klog.Info("Deprecation file not found. Invoke registration")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, kubeletEndpoint, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", addr)
		}))
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pluginapi.NewRegistrationClient(conn)
	reqt := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     path.Base(m.socket),
		ResourceName: resourceName,
		Options: &pluginapi.DevicePluginOptions{
			PreStartRequired:                m.preStartContainerFlag,
			GetPreferredAllocationAvailable: m.getPreferredAllocationFlag,
		},
	}

	_, err = client.Register(context.Background(), reqt)
	if err != nil {
		return err
	}
	return nil
}

// GetDevicePluginOptions returns DevicePluginOptions settings for the device plugin.
func (m *Stub) GetDevicePluginOptions(ctx context.Context, e *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	options := &pluginapi.DevicePluginOptions{
		PreStartRequired:                m.preStartContainerFlag,
		GetPreferredAllocationAvailable: m.getPreferredAllocationFlag,
	}
	return options, nil
}

// PreStartContainer resets the devices received
func (m *Stub) PreStartContainer(ctx context.Context, r *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	klog.Infof("PreStartContainer, %+v", r)
	return &pluginapi.PreStartContainerResponse{}, nil
}

// ListAndWatch lists devices and update that list according to the Update call
func (m *Stub) ListAndWatch(e *pluginapi.Empty, s pluginapi.DevicePlugin_ListAndWatchServer) error {
	klog.Info("ListAndWatch")

	s.Send(&pluginapi.ListAndWatchResponse{Devices: m.devs})

	for {
		select {
		case <-m.stop:
			return nil
		case updated := <-m.update:
			s.Send(&pluginapi.ListAndWatchResponse{Devices: updated})
		}
	}
}

// Update allows the device plugin to send new devices through ListAndWatch
func (m *Stub) Update(devs []*pluginapi.Device) {
	m.update <- devs
}

// GetPreferredAllocation gets the preferred allocation from a set of available devices
func (m *Stub) GetPreferredAllocation(ctx context.Context, r *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	klog.Infof("GetPreferredAllocation, %+v", r)

	devs := make(map[string]pluginapi.Device)

	for _, dev := range m.devs {
		devs[dev.ID] = *dev
	}

	return m.getPreferredAllocFunc(r, devs)
}

// Allocate does a mock allocation
func (m *Stub) Allocate(ctx context.Context, r *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	klog.Infof("Allocate, %+v", r)

	devs := make(map[string]pluginapi.Device)

	for _, dev := range m.devs {
		devs[dev.ID] = *dev
	}

	return m.allocFunc(r, devs)
}

func (m *Stub) cleanup() error {
	if err := os.Remove(m.socket); err != nil && !os.IsNotExist(err) {
		return err
	}

	return nil
}
