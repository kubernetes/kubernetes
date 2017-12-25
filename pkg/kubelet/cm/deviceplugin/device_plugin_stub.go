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
	"path"
	"path/filepath"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
)

// StubDevicePlugin is a stub implementation for a Device Plugin.
type StubDevicePlugin struct {
	devs   []*pluginapi.Device
	socket string

	stop   chan interface{}
	update chan []*pluginapi.Device

	server          *grpc.Server
	serverErrorChan chan error
}

// NewStubDevicePlugin returns an initialized DevicePlugin Stub.
func NewStubDevicePlugin(devs []*pluginapi.Device, socket string) *StubDevicePlugin {
	return &StubDevicePlugin{
		devs:   devs,
		socket: socket,

		stop:   make(chan interface{}),
		update: make(chan []*pluginapi.Device),

		serverErrorChan: make(chan error),
	}
}

// dial establishes the gRPC communication with the registered device plugin.
func dial(unixSocketPath string, timeout time.Duration) (*grpc.ClientConn, error) {
	c, err := grpc.Dial(unixSocketPath, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithTimeout(timeout),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, fmt.Errorf(errFailedToDialDevicePlugin+" %v", err)
	}

	return c, nil
}

// Start starts the gRPC server of the device plugin
func (m *StubDevicePlugin) Start() error {
	err := m.cleanup()
	if err != nil {
		return err
	}

	dir, _ := filepath.Split(m.socket)
	os.MkdirAll(dir, 0755)

	sock, err := net.Listen("unix", m.socket)
	if err != nil {
		return err
	}

	m.server = grpc.NewServer([]grpc.ServerOption{}...)
	pluginapi.RegisterDevicePluginServer(m.server, m)

	go func() {
		err := m.server.Serve(sock)
		m.serverErrorChan <- err
		close(m.serverErrorChan)
	}()

	go func() {
		for {
			select {
			case err := <-m.serverErrorChan:
				glog.Errorf("gRPC server ended unexpectedly")
				panic(err)
			case <-m.stop:
				m.server.Stop()
				return
			}
		}
	}()

	// Wait for server to start by launching a blocking connexion
	c, err := dial(m.socket, 5*time.Second)
	if err != nil {
		return err
	}
	c.Close()

	glog.V(2).Infof("Starting to serve on %s", m.socket)

	return nil
}

// Stop stops the gRPC server
func (m *StubDevicePlugin) Stop() error {
	glog.V(2).Infof("Stopping server")

	close(m.stop)
	<-m.serverErrorChan

	return m.cleanup()
}

// Register registers the device plugin for the given resourceName with Kubelet.
func (m *StubDevicePlugin) Register(kubeletEndpoint, resourceName string) error {
	c, err := dial(kubeletEndpoint, 5*time.Second)
	defer c.Close()

	if err != nil {
		return err
	}

	client := pluginapi.NewRegistrationClient(c)
	reqt := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     path.Base(m.socket),
		ResourceName: resourceName,
	}

	_, err = client.Register(context.Background(), reqt)
	if err != nil {
		return err
	}

	return nil
}

// ListAndWatch lists devices and update that list according to the Update call
func (m *StubDevicePlugin) ListAndWatch(e *pluginapi.Empty, s pluginapi.DevicePlugin_ListAndWatchServer) error {
	glog.V(2).Infof("ListAndWatch")
	var devs []*pluginapi.Device

	for _, d := range m.devs {
		devs = append(devs, &pluginapi.Device{
			ID:     d.ID,
			Health: pluginapi.Healthy,
		})
	}

	s.Send(&pluginapi.ListAndWatchResponse{Devices: devs})

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
func (m *StubDevicePlugin) Update(devs []*pluginapi.Device) {
	m.update <- devs
}

// Allocate does a mock allocation
func (m *StubDevicePlugin) Allocate(ctx context.Context, r *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	glog.V(2).Infof("Allocate, %+v", r)

	var response pluginapi.AllocateResponse
	return &response, nil
}

func (m *StubDevicePlugin) cleanup() error {
	if err := os.Remove(m.socket); err != nil && !os.IsNotExist(err) {
		return err
	}

	return nil
}
