package deviceplugin

import (
	"fmt"
	"io"
	"net"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

func allocate(e *Endpoint, devs []*pluginapi.Device) (*pluginapi.AllocateResponse, error) {
	return e.client.Allocate(context.Background(), &pluginapi.AllocateRequest{
		Devices: devs,
	})
}

func deallocate(e *Endpoint, devs []*pluginapi.Device) (*pluginapi.Error, error) {
	return e.client.Deallocate(context.Background(), &pluginapi.DeallocateRequest{
		Devices: devs,
	})
}

func (s *Registery) InitiateCommunication(r *pluginapi.RegisterRequest,
	response *pluginapi.RegisterResponse) {

	connection, client, err := dial(r.Unixsocket)
	if err != nil {
		response.Error = NewError(err.Error())
		return
	}

	if err := initProtocol(client); err != nil {
		response.Error = NewError(err.Error())
		return
	}

	devs, err := listDevs(client)
	if err != nil {
		response.Error = NewError(err.Error())
		return
	}

	if err := IsDevsValid(devs, r.Vendor); err != nil {
		response.Error = NewError(err.Error())
		return
	}

	s.Endpoints[r.Vendor] = &Endpoint{
		c:          connection,
		client:     client,
		socketname: r.Unixsocket,
	}

	for _, d := range devs {
		s.Manager.addDevice(d)
	}

	go s.Monitor(client, r.Vendor)
}

func (s *Registery) Monitor(client pluginapi.DeviceManagerClient, vendor string) {
Start:
	stream, err := client.Monitor(context.Background(), &pluginapi.Empty{})
	if err != nil {
		glog.Infof("Could not call monitor for device plugin with "+
			"Kind '%s' and with error %+v", err)
		return
	}

	for {
		health, err := stream.Recv()
		if err == io.EOF {
			glog.Infof("End of Stream when monitoring vendor %s "+
				", restarting monitor", vendor)
			goto Start
		}

		if err != nil {
			glog.Infof("Monitor stoped unexpectedly for device plugin with "+
				"Kind '%s' and with error %+v", err)
			return
		}

		s.handleDeviceUnhealthy(&pluginapi.Device{
			Name:   health.Name,
			Kind:   health.Kind,
			Vendor: health.Vendor,
		}, vendor)
	}
}

func (s *Registery) handleDeviceUnhealthy(d *pluginapi.Device, vendor string) {
	s.Manager.mutex.Lock()
	defer s.Manager.mutex.Unlock()

	glog.Infof("Unhealthy device %+v for device plugin for vendor '%s'", d, vendor)

	if err := IsDevValid(d, vendor); err != nil {
		glog.Infof("device is not valid %+v", err)
		return
	}

	devs, ok := s.Manager.devices[d.Kind]
	if !ok {
		glog.Infof("Manager does not have device plugin for device %+v", d)
		return
	}

	available, ok := s.Manager.available[d.Kind]
	if !ok {
		glog.Infof("Manager does not have device plugin for device %+v", d)
		return
	}

	i, ok := HasDevice(d, devs)
	if !ok {
		glog.Infof("Could not find device %+v for device plugin for "+
			"vendor %s", d, vendor)
		return
	}

	devs[i].Health = pluginapi.Unhealthy

	j, ok := HasDevice(d, available)
	if ok {
		glog.Infof("Device %+v found in available pool, removing", d)
		s.Manager.available[vendor] = deleteDevAt(j, available)
		return
	}

	glog.Infof("Device %+v not found in available pool (might be used) using callback", d)
	s.Manager.callback(devs[i])
}

func listDevs(client pluginapi.DeviceManagerClient) ([]*pluginapi.Device, error) {
	var devs []*pluginapi.Device

	stream, err := client.Discover(context.Background(), &pluginapi.Empty{})
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

func initProtocol(client pluginapi.DeviceManagerClient) error {
	_, err := client.Init(context.Background(), &pluginapi.Empty{})
	if err != nil {
		return fmt.Errorf("fail to start communication with device plugin: %v", err)
	}

	return nil
}

func dial(unixSocket string) (*grpc.ClientConn, pluginapi.DeviceManagerClient, error) {

	c, err := grpc.Dial(pluginapi.DevicePluginPath+unixSocket, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}))

	if err != nil {
		return nil, nil, fmt.Errorf("fail to dial device plugin: %v", err)
	}

	glog.Infof("Renaud, Dialed device plugin: %+v", unixSocket)
	return c, pluginapi.NewDeviceManagerClient(c), nil
}
