package deviceplugin

import (
	"net"
	"os"
	"strings"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

type Endpoint struct {
	c          *grpc.ClientConn
	client     pluginapi.DeviceManagerClient
	socketname string
}

type Registery struct {
	Endpoints map[string]*Endpoint // Key is Kind
	Manager   *Manager
	server    *grpc.Server
}

func newServer() *Registery {
	return &Registery{
		Endpoints: make(map[string]*Endpoint),
	}
}

func (m *Manager) startRegistery() error {
	os.Remove(pluginapi.KubeletSocket)

	s, err := net.Listen("unix", pluginapi.KubeletSocket)
	if err != nil {
		glog.Errorf("Failed to listen to socket while starting "+
			"device pluginregistery", err)
		return err
	}

	m.registry.server = grpc.NewServer([]grpc.ServerOption{}...)

	pluginapi.RegisterPluginRegistrationServer(m.registry.server, m.registry)
	go m.registry.server.Serve(s)

	return nil
}

func (s *Registery) Register(ctx context.Context,
	r *pluginapi.RegisterRequest) (*pluginapi.RegisterResponse, error) {

	response := &pluginapi.RegisterResponse{
		Version: pluginapi.Version,
	}

	if r.Version != pluginapi.Version {
		response.Error = NewError("Unsupported version")
		return response, nil
	}

	r.Vendor = strings.TrimSpace(r.Vendor)
	if err := IsVendorValid(r.Vendor); err != nil {
		response.Error = NewError(err.Error())
		return response, nil
	}

	if e, ok := s.Endpoints[r.Vendor]; ok {
		if e.socketname != r.Unixsocket {
			response.Error = NewError("A device plugin is already in charge of " +
				"this vendor on socket " + e.socketname)
			return response, nil
		}

		s.Manager.deleteDevices(r.Vendor)
	}

	s.InitiateCommunication(r, response)

	return response, nil
}

func (s *Registery) Heartbeat(ctx context.Context,
	r *pluginapi.HeartbeatRequest) (*pluginapi.HeartbeatResponse, error) {

	r.Vendor = strings.TrimSpace(r.Vendor)
	if err := IsVendorValid(r.Vendor); err != nil {
		return &pluginapi.HeartbeatResponse{
			Response: pluginapi.HeartbeatError,
			Error:    NewError(err.Error()),
		}, nil
	}

	if _, ok := s.Endpoints[r.Vendor]; ok {
		return &pluginapi.HeartbeatResponse{
			Response: pluginapi.HeartbeatOk,
		}, nil
	}

	return &pluginapi.HeartbeatResponse{
		Response: pluginapi.HeartbeatNeedsRegistration,
	}, nil
}
