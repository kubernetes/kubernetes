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
	"net"
	"os"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

type endpoint struct {
	c          *grpc.ClientConn
	client     pluginapi.DeviceManagerClient
	socketname string
}

type registry struct {
	socketname string
	socketdir  string

	Endpoints map[string]*endpoint // Key is Kind
	Manager   *Manager
	server    *grpc.Server
}

func newRegistry(socketPath string) (*registry, error) {
	if socketPath == "" || !filepath.IsAbs(socketPath) {
		return nil, fmt.Errorf(ErrBadSocket+" %v", socketPath)
	}

	dir, file := filepath.Split(socketPath)
	return &registry{
		Endpoints:  make(map[string]*endpoint),
		socketname: file,
		socketdir:  dir,
	}, nil
}

func (m *Manager) startRegistry() error {
	socketPath := filepath.Join(m.registry.socketdir, m.registry.socketname)
	os.MkdirAll(m.registry.socketdir, 0755)

	if err := os.Remove(socketPath); err != nil && !os.IsNotExist(err) {
		glog.Errorf(ErrRemoveSocket+" %+v", err)
		return err
	}

	s, err := net.Listen("unix", socketPath)
	if err != nil {
		glog.Errorf(ErrListenSocket+" %+v", err)
		return err
	}

	m.registry.server = grpc.NewServer([]grpc.ServerOption{}...)

	pluginapi.RegisterPluginRegistrationServer(m.registry.server, m.registry)
	go m.registry.server.Serve(s)

	return nil
}

func (s *registry) Register(ctx context.Context,
	r *pluginapi.RegisterRequest) (*pluginapi.RegisterResponse, error) {

	response := &pluginapi.RegisterResponse{
		Version: pluginapi.Version,
	}

	if r.Version != pluginapi.Version {
		response.Error = pluginapi.ErrUnsuportedVersion
		return response, nil
	}

	r.Vendor = strings.TrimSpace(r.Vendor)
	if err := IsVendorValid(r.Vendor); err != nil {
		response.Error = err.Error()
		return response, nil
	}

	if e, ok := s.Endpoints[r.Vendor]; ok {
		if e.socketname != r.Unixsocket {
			response.Error = pluginapi.ErrDevicePluginAlreadyExists + " " + e.socketname
			return response, nil
		}

		s.Manager.deleteDevices(r.Vendor)
	}

	s.initiateCommunication(r, response)

	return response, nil
}

func (s *registry) Heartbeat(ctx context.Context,
	r *pluginapi.HeartbeatRequest) (*pluginapi.HeartbeatResponse, error) {

	glog.Infof("Recieved connection from device plugin %+v", r)

	r.Vendor = strings.TrimSpace(r.Vendor)
	if err := IsVendorValid(r.Vendor); err != nil {
		return &pluginapi.HeartbeatResponse{
			Response: pluginapi.HeartbeatFailure,
			Error:    err.Error(),
		}, nil
	}

	if _, ok := s.Endpoints[r.Vendor]; ok {
		return &pluginapi.HeartbeatResponse{
			Response: pluginapi.HeartbeatOk,
		}, nil
	}

	return &pluginapi.HeartbeatResponse{
		Response: pluginapi.HeartbeatFailure,
	}, nil
}
