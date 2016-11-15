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

package remote

import (
	"fmt"
	"net"
	"os"
	"syscall"

	"github.com/golang/glog"
	"google.golang.org/grpc"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	"k8s.io/kubernetes/pkg/util/interrupt"
)

const (
	// defaultEndpoint is the default address of dockershim grpc server socket.
	defaultAddress = "/var/run/dockershim.sock"
	// unixProtocol is the network protocol of unix socket.
	unixProtocol = "unix"
)

// DockerServer is the grpc server of dockershim.
type DockerServer struct {
	// addr is the address to serve on.
	addr string
	// service is the docker service which implements runtime and image services.
	service DockerService
	// server is the grpc server.
	server *grpc.Server
}

// NewDockerServer creates the dockershim grpc server.
func NewDockerServer(addr string, s dockershim.DockerService) *DockerServer {
	return &DockerServer{
		addr:    addr,
		service: NewDockerService(s),
	}
}

// Start starts the dockershim grpc server.
func (s *DockerServer) Start() error {
	glog.V(2).Infof("Start dockershim grpc server")
	// Unlink to cleanup the previous socket file.
	err := syscall.Unlink(s.addr)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to unlink socket file %q: %v", s.addr, err)
	}
	l, err := net.Listen(unixProtocol, s.addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %q: %v", s.addr, err)
	}
	// Create the grpc server and register runtime and image services.
	s.server = grpc.NewServer()
	runtimeApi.RegisterRuntimeServiceServer(s.server, s.service)
	runtimeApi.RegisterImageServiceServer(s.server, s.service)
	go func() {
		// Use interrupt handler to make sure the server to be stopped properly.
		h := interrupt.New(nil, s.Stop)
		err := h.Run(func() error { return s.server.Serve(l) })
		if err != nil {
			glog.Errorf("Failed to serve connections: %v", err)
		}
	}()
	return nil
}

// Stop stops the dockershim grpc server.
func (s *DockerServer) Stop() {
	glog.V(2).Infof("Stop docker server")
	s.server.Stop()
}
