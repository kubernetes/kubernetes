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
	DeviceVendor = "foo"
	DeviceKind   = "device"
	DeviceSock   = "device.sock"
	ServerSock   = pluginapi.DevicePluginPath + DeviceSock

	WaitToKill = 3
)

var (
	deviceErrorChan = make(chan *pluginapi.Device)
)

type DevicePluginServer struct {
}

func (d *DevicePluginServer) Init(ctx context.Context, e *pluginapi.Empty) (*pluginapi.Empty, error) {
	return nil, nil
}

func (d *DevicePluginServer) Stop(ctx context.Context, e *pluginapi.Empty) (*pluginapi.Empty, error) {
	return nil, nil
}

func (d *DevicePluginServer) Discover(e *pluginapi.Empty, deviceStream pluginapi.DeviceManager_DiscoverServer) error {
	for i := 0; i < 5; i++ {
		deviceStream.Send(&pluginapi.Device{
			Name:       strconv.Itoa(i),
			Kind:       DeviceKind,
			Vendor:     DeviceVendor,
			Properties: nil,
		})
	}

	return nil
}

func (d *DevicePluginServer) Monitor(e *pluginapi.Empty, deviceStream pluginapi.DeviceManager_MonitorServer) error {
	for {
		select {
		case d := <-deviceErrorChan:
			time.Sleep(WaitToKill * time.Second)

			err := deviceStream.Send(&pluginapi.DeviceHealth{
				Name:   d.Name,
				Kind:   d.Kind,
				Vendor: DeviceVendor,
				Health: pluginapi.Unhealthy,
			})

			if err != nil {
				log.Println("error while monitoring: %+v", err)
			}
		}

		time.Sleep(time.Second)
	}
}

func (d *DevicePluginServer) Allocate(ctx context.Context, r *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {

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

func (d *DevicePluginServer) Deallocate(ctx context.Context, r *pluginapi.DeallocateRequest) (*pluginapi.Error, error) {
	return &pluginapi.Error{}, nil
}

func StartDevicePluginServer(t *testing.T) {
	os.Remove(ServerSock)
	sock, err := net.Listen("unix", ServerSock)
	require.NoError(t, err)

	grpcServer := grpc.NewServer([]grpc.ServerOption{}...)
	pluginapi.RegisterDeviceManagerServer(grpcServer, &DevicePluginServer{})

	go grpcServer.Serve(sock)
}

func DialRegistery(t *testing.T) {
	c, err := grpc.Dial(pluginapi.KubeletSocket, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	require.NoError(t, err)

	client := pluginapi.NewPluginRegistrationClient(c)
	resp, err := client.Register(context.Background(), &pluginapi.RegisterRequest{
		Version:    pluginapi.Version,
		Unixsocket: DeviceSock,
		Vendor:     DeviceVendor,
	})

	require.Len(t, resp.Error, 0)
	require.NoError(t, err)
	c.Close()
}

func monitorCallback(d *pluginapi.Device) {
}

func TestManager(t *testing.T) {
	mgr, err := NewManager(monitorCallback)
	require.NoError(t, err)

	StartDevicePluginServer(t)
	DialRegistery(t)

	assert.Len(t, mgr.Devices()["device"], 5)

	devs, resp, err := mgr.Allocate("device", 1)

	require.NoError(t, err)
	assert.Len(t, resp[0].Envs, 1)
	assert.Len(t, resp[0].Mounts, 1)
	assert.Len(t, devs, 1)

	assert.Len(t, mgr.Available()["device"], 4)

	mgr.Deallocate(devs)
	assert.Len(t, mgr.Available()["device"], 5)

	time.Sleep((WaitToKill + 1) * time.Second)
	unhealthyDev := devs[0]

	devs = mgr.Devices()[DeviceKind]
	i, ok := hasDevice(unhealthyDev, devs)

	assert.True(t, ok)
	assert.Equal(t, pluginapi.Unhealthy, devs[i].Health)
}
