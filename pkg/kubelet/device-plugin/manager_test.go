/*
Copyright 2016 The Kubernetes Authors.

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
	"log"
	"net"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

const (
	deviceVendor = "foo"
	deviceKind   = "device"
	deviceSock   = "device.sock"
	serverSock   = pluginapi.DevicePluginPath + deviceSock

	waitToKill = 1
)

var (
	deviceErrorChan = make(chan *pluginapi.Device)
	nDevices        = 3
)

type DevicePlugin struct {
	devs   []*pluginapi.Device
	server *grpc.Server
}

// TestManagerDiscovery tests that device plugin's Discovery method
// is called when registering
func TestManagerDiscovery(t *testing.T) {
	mgr, plugin, err := setup()
	require.NoError(t, err)

	devs, ok := mgr.Devices()[deviceKind]
	assert.True(t, ok)

	assert.Len(t, devs, nDevices)
	for _, d := range devs {
		_, ok = HasDevice(d, plugin.devs)
		assert.True(t, ok)
	}

	teardown(mgr, plugin)
}

// TestManagerAllocation tests that device plugin's Allocation and Deallocation method
// allocates correctly the devices
// This also tests the RM of the manager
func TestManagerAllocation(t *testing.T) {
	mgr, plugin, err := setup()
	require.NoError(t, err)

	for i := 1; i < nDevices; i++ {
		devs, resp, err := mgr.Allocate("device", i)
		require.NoError(t, err)

		assert.Len(t, devs, i)
		assert.Len(t, resp[0].Envs, 1)
		assert.Len(t, resp[0].Mounts, 1)

		assert.Len(t, mgr.Available()["device"], nDevices-i)

		// Deallocation test
		mgr.Deallocate(devs)
		assert.Len(t, mgr.Available()["device"], nDevices)
	}

	teardown(mgr, plugin)
}

// TestManagerAllocation tests that device plugin's Allocation and Deallocation method
func TestManagerMonitoring(t *testing.T) {
	mgr, plugin, err := setup()
	require.NoError(t, err)

	devs, _, err := mgr.Allocate("device", 1)
	require.NoError(t, err)

	// Monitoring test
	time.Sleep(waitToKill*time.Second + 500*time.Millisecond)
	unhealthyDev := devs[0]

	devs = mgr.Devices()[deviceKind]
	i, ok := HasDevice(unhealthyDev, devs)

	assert.True(t, ok)
	assert.Equal(t, pluginapi.Unhealthy, devs[i].Health)

	teardown(mgr, plugin)
}

func setup() (*Manager, *DevicePlugin, error) {
	mgr, err := NewManager(nil, nil, monitorCallback)
	fmt.Println(err)
	if err != nil {
		return nil, nil, err
	}

	plugin, err := StartDevicePluginServer()
	if err != nil {
		return nil, nil, err
	}

	err = DialRegistery()
	if err != nil {
		return nil, nil, err
	}

	// Give some time for the discovery phase to happen
	time.Sleep(time.Millisecond * 500)

	return mgr, plugin, nil
}

func teardown(mgr *Manager, plugin *DevicePlugin) {
	plugin.server.Stop()
	mgr.Stop()
}

// DevicePlugin implementation
func (d *DevicePlugin) Init(ctx context.Context,
	e *pluginapi.Empty) (*pluginapi.Empty, error) {

	for i := 0; i < nDevices; i++ {
		d.devs = append(d.devs, NewDevice(strconv.Itoa(i), deviceKind, deviceVendor))
	}

	return nil, nil
}

func (d *DevicePlugin) Discover(e *pluginapi.Empty,
	deviceStream pluginapi.DeviceManager_DiscoverServer) error {

	for _, dev := range d.devs {
		deviceStream.Send(dev)
	}

	return nil
}

func (d *DevicePlugin) Monitor(e *pluginapi.Empty,
	deviceStream pluginapi.DeviceManager_MonitorServer) error {

	for {
		d := <-deviceErrorChan
		time.Sleep(waitToKill * time.Second)

		h := NewDeviceHealth(d.Name, d.Kind, deviceVendor, pluginapi.Unhealthy)
		err := deviceStream.Send(h)

		if err != nil {
			log.Println("Error while monitoring: %+v", err)
		}
	}
}

func (d *DevicePlugin) Allocate(ctx context.Context,
	r *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {

	var response pluginapi.AllocateResponse

	response.Envs = append(response.Envs, &pluginapi.KeyValue{
		Key:   "TEST_ENV_VAR",
		Value: "FOO",
	})

	response.Mounts = append(response.Mounts, &pluginapi.Mount{
		Name:      "mount-abc",
		HostPath:  "/tmp",
		MountPath: "/device-plugin",
		ReadOnly:  false,
	})

	deviceErrorChan <- r.Devices[0]

	return &response, nil
}

func (d *DevicePlugin) Deallocate(ctx context.Context,
	r *pluginapi.DeallocateRequest) (*pluginapi.Error, error) {

	return &pluginapi.Error{}, nil
}

func StartDevicePluginServer() (*DevicePlugin, error) {
	os.Remove(serverSock)

	sock, err := net.Listen("unix", serverSock)
	if err != nil {
		return nil, err
	}

	plugin := &DevicePlugin{}
	plugin.server = grpc.NewServer([]grpc.ServerOption{}...)

	pluginapi.RegisterDeviceManagerServer(plugin.server, plugin)
	go plugin.server.Serve(sock)

	return plugin, nil
}

func DialRegistery() error {
	c, err := grpc.Dial(pluginapi.KubeletSocket, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return err
	}

	client := pluginapi.NewPluginRegistrationClient(c)
	resp, err := client.Register(context.Background(), &pluginapi.RegisterRequest{
		Version:    pluginapi.Version,
		Unixsocket: deviceSock,
		Vendor:     deviceVendor,
	})

	if err != nil {
		return err
	}

	if resp.Error != nil && resp.Error.Error {
		return fmt.Errorf("%s", resp.Error.Reason)
	}

	c.Close()

	return nil
}

func monitorCallback(d *pluginapi.Device) {
}
