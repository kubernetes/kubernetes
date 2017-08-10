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
	"io"
	"net"
	"path/filepath"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

func allocate(e *endpoint, devs []*pluginapi.Device) (*pluginapi.AllocateResponse, error) {
	return e.client.Allocate(context.Background(), &pluginapi.AllocateRequest{
		Devices: devs,
	})
}

func (s *registry) initiateCommunication(r *pluginapi.RegisterRequest,
	response *pluginapi.RegisterResponse) {

	connection, client, err := dial(s.socketdir, r.Unixsocket)
	if err != nil {
		response.Error = err.Error()
		return
	}

	devs, err := listDevs(client)
	if err != nil {
		response.Error = err.Error()
		return
	}

	if err := IsDevsValid(devs, r.Vendor); err != nil {
		response.Error = err.Error()
		return
	}

	s.Endpoints[r.Vendor] = &endpoint{
		c:          connection,
		client:     client,
		socketname: r.Unixsocket,
	}

	for _, d := range devs {
		s.Manager.addDevice(d)
	}

	go s.healthCheck(client, r.Vendor)
}

func (s *registry) healthCheck(client pluginapi.DeviceManagerClient, vendor string) {
Start:
	stream, err := client.HealthCheck(context.Background(), &pluginapi.Empty{})
	if err != nil {
		glog.Infof("Could not call healthCheck for device plugin with "+
			"Kind '%s' and with error %+v", err)
		return
	}

	for {
		dev, err := stream.Recv()
		glog.Infof("Read Unhealthy device %+v for device plugin for vendor '%s'", dev, vendor)
		if err == io.EOF {
			glog.Infof("End of Stream when healthChecking vendor %s "+
				", restarting healthCheck", vendor)
			time.Sleep(time.Second)
			goto Start
		}

		if err != nil {
			glog.Infof("healthCheck stoped unexpectedly for device plugin with "+
				"Kind '%s' and with error %+v", err)
			return
		}

		err = s.handleDeviceUnhealthy(dev, vendor)
		if err != nil {
			glog.Infof("+%v", err)
		}
	}
}

func (s *registry) handleDeviceUnhealthy(d *pluginapi.Device, vendor string) error {
	s.Manager.mutex.Lock()
	defer s.Manager.mutex.Unlock()

	glog.Infof("Unhealthy device %+v for device plugin for vendor '%s'", d, vendor)

	if err := IsDevValid(d, vendor); err != nil {
		return fmt.Errorf("%+v", err)
	}

	devs, ok := s.Manager.devices[d.Kind]
	if !ok {
		return fmt.Errorf(ErrDevicePluginUnknown+" %+v", d)
	}

	available, ok := s.Manager.available[d.Kind]
	if !ok {
		return fmt.Errorf(ErrDevicePluginUnknown+" %+v", d)
	}

	i, ok := HasDevice(d, devs)
	if !ok {
		return fmt.Errorf(ErrDeviceUnknown+" %+v", d)
	}

	devs[i].Health = pluginapi.Unhealthy

	j, ok := HasDevice(d, available)
	if ok {
		glog.Infof("Device %+v found in available pool, removing", d)
		s.Manager.available[vendor] = deleteDevAt(j, available)

		return nil
	}

	glog.Infof("Device %+v not found in available pool (might be used) using callback", d)
	s.Manager.callback(devs[i])

	return nil
}

func listDevs(client pluginapi.DeviceManagerClient) ([]*pluginapi.Device, error) {
	var devs []*pluginapi.Device

	stream, err := client.ListDevices(context.Background(), &pluginapi.Empty{})
	if err != nil {
		return nil, fmt.Errorf("Failed to discover devices: %v", err)
	}

	for {
		d, err := stream.Recv()
		if err == io.EOF {
			break
		}

		if err != nil {
			return nil, fmt.Errorf("Failed to Recv while processing device"+
				"plugin client with err %+v", err)
		}

		devs = append(devs, d)
	}

	return devs, nil
}

func dial(socketdir, unixSocket string) (*grpc.ClientConn,
	pluginapi.DeviceManagerClient, error) {

	socketPath := filepath.Join(socketdir, unixSocket)

	c, err := grpc.Dial(socketPath, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}))

	if err != nil {
		return nil, nil, fmt.Errorf(pluginapi.ErrFailedToDialDevicePlugin+" %v", err)
	}

	glog.Infof("Dialed device plugin: %+v", unixSocket)
	return c, pluginapi.NewDeviceManagerClient(c), nil
}
