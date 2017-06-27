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
	"net"
	"os"
	//"strconv"
	"io/ioutil"
	"strings"
	"testing"
	"time"

	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

const (
	DeviceSock1 = "device.sock.solarflare.nic"
	ServerSock1 = pluginapi.DevicePluginPath + DeviceSock1
)

var discovered = 0

type DevicePluginServer1 struct {
}

func (d *DevicePluginServer1) Init(ctx context.Context, e *pluginapi.Empty) (*pluginapi.Empty, error) {
	glog.Errorf("ramki: Init\n")
	return nil, nil
}

func (d *DevicePluginServer1) Stop(ctx context.Context, e *pluginapi.Empty) (*pluginapi.Empty, error) {
	glog.Errorf("ramki: Stop\n")
	return nil, nil
}

func (d *DevicePluginServer1) Discover(e *pluginapi.Empty, deviceStream pluginapi.DeviceManager_DiscoverServer) error {
	glog.Errorf("ramki: Discover\n")

	// read the whole file at once
	//b, err := ioutil.ReadFile("/proc/devices")
	b, err := ioutil.ReadFile("/proc/devices")
	if err != nil {
		panic(err)
	}
	s := string(b)

	if strings.Index(s, "sfc_char") > 0 {
		deviceStream.Send(&pluginapi.Device{
			Name:       "/dev/sfc_char",
			Kind:       "device",
			Vendor:     "device",
			Properties: nil,
		})

		discovered++
	}

	if strings.Index(s, "sfc_affinity") > 0 {
		deviceStream.Send(&pluginapi.Device{
			Name:       "/dev/sfc_affnity",
			Kind:       "device",
			Vendor:     "device",
			Properties: nil,
		})

		discovered++
	}

	if strings.Index(s, "onload_epoll") > 0 {
		deviceStream.Send(&pluginapi.Device{
			Name:       "/dev/onload_epoll",
			Kind:       "device",
			Vendor:     "device",
			Properties: nil,
		})

		discovered++
	}

	if strings.Index(s, "onload_cplane") > 0 {
		deviceStream.Send(&pluginapi.Device{
			Name:       "/dev/onload_cplane",
			Kind:       "device",
			Vendor:     "device",
			Properties: nil,
		})

		discovered++
	}

	// '\n' is added to avoid a match with onload_cplane and onload_epoll
	if strings.Index(s, "onload\n") > 0 {
		deviceStream.Send(&pluginapi.Device{
			Name:       "/dev/onload",
			Kind:       "device",
			Vendor:     "device",
			Properties: nil,
		})

		discovered++
	}

	return nil
}

func (d *DevicePluginServer1) Monitor(e *pluginapi.Empty, deviceStream pluginapi.DeviceManager_MonitorServer) error {
	return nil
}

func (d *DevicePluginServer1) Allocate(ctx context.Context, r *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {

	glog.Errorf("ramki: Allocate\n")
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

	return &response, nil
}

func (d *DevicePluginServer1) Deallocate(ctx context.Context, r *pluginapi.DeallocateRequest) (*pluginapi.Error, error) {
	glog.Errorf("ramki: Deallocate\n")
	return &pluginapi.Error{}, nil
}

func StartDevicePluginServer1(t *testing.T) {
	glog.Errorf("ramki: StartDevicePluginServer:%s", ServerSock1)
	os.Remove(ServerSock1)
	sock, err := net.Listen("unix", ServerSock1)
	require.NoError(t, err)

	grpcServer := grpc.NewServer([]grpc.ServerOption{}...)
	pluginapi.RegisterDeviceManagerServer(grpcServer, &DevicePluginServer1{})

	go grpcServer.Serve(sock)
}

func DialRegistery1(t *testing.T) {
	glog.Errorf("ramki: DialRegistery\n")
	c, err := grpc.Dial(pluginapi.KubeletSocket, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	require.NoError(t, err)

	client := pluginapi.NewPluginRegistrationClient(c)
	resp, err := client.Register(context.Background(), &pluginapi.RegisterRequest{
		Version:    pluginapi.Version,
		Unixsocket: DeviceSock1,
		Vendor:     "device",
	})

	glog.Errorf("resp: %+v", resp)
	require.Len(t, resp.Error, 0)
	require.NoError(t, err)
	c.Close()
}

func TestManager1(t *testing.T) {

	glog.Errorf("ramki: TestManager\n")
	mgr, err := NewManager(monitorCallback)
	require.NoError(t, err)

	StartDevicePluginServer1(t)
	DialRegistery1(t)

	//assert.Len(t, mgr.Devices()["device"], 2)

	if discovered == 0 {
		return
	}

	devs, resp, err := mgr.Allocate("device", 1)

	require.NoError(t, err)
	assert.Len(t, resp[0].Envs, 1)
	assert.Len(t, resp[0].Mounts, 1)
	assert.Len(t, devs, 1)

	//assert.Len(t, mgr.Available()["device"], 1)

	mgr.Deallocate(devs)
	//assert.Len(t, mgr.Available()["device"], 2)
}
