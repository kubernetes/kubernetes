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
	"time"

	"github.com/golang/glog"
	"google.golang.org/grpc"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	"k8s.io/kubernetes/pkg/kubelet/remote"
	"k8s.io/kubernetes/pkg/util/interrupt"
)

const (
	// defaultEndpoint is the default address of dockershim grpc server socket.
	defaultAddress = "/var/run/dockershim.sock"
	// unixProtocol is the network protocol of unix socket.
	unixProtocol = "unix"
	// startTimeout is the timeout to wait for dockershim grpc server to become ready.
	startTimeout = 2 * time.Minute
)

// DockerServer is the grpc server of dockershim.
type DockerServer struct {
	// addr is the address to serve on.
	addr string
	// service is the docker service which implements runtime and image services.
	service *DockerService
	// listener is the network listener. It should be closed properly before the
	// server is stopped.
	listener net.Listener
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
	s.listener, err = net.Listen(unixProtocol, s.addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %q: %v", s.addr, err)
	}
	// Create the grpc server and register runtime and image services.
	s.server = grpc.NewServer()
	runtimeApi.RegisterRuntimeServiceServer(s.server, s.service)
	runtimeApi.RegisterImageServiceServer(s.server, s.service)
	errCh := make(chan error)
	go func() {
		defer close(errCh)
		// Use interrupt handler to make sure the server to be stopped properly.
		h := interrupt.New(nil, s.Stop)
		err := h.Run(func() error { return s.server.Serve(s.listener) })
		if err != nil {
			errCh <- fmt.Errorf("failed to serve connections: %v", err)
		}
	}()
	// We need readinessCheck now because currently we start dockershim grpc server
	// in-process, and will connect the server right after starting it. So it must
	// be ready before the function returns.
	return s.readinessCheck(errCh)
}

// readinessCheck checks the readiness of dockershim grpc server. Once there is an error
// in errCh, the function will stop waiting and return the error.
func (s *DockerServer) readinessCheck(errCh <-chan error) error {
	glog.V(2).Infof("Running readiness check for docker server")
	endTime := time.Now().Add(startTimeout)
	for endTime.After(time.Now()) {
		select {
		case err := <-errCh:
			return err
		case <-time.After(time.Second):
			glog.V(4).Infof("Connecting to runtime service %q", s.addr)
			// The connection timeout doesn't matter here because
			// we don't really use any remote runtime function here.
			_, err := remote.NewRemoteRuntimeService(s.addr, 1*time.Minute)
			if err != nil {
				glog.V(4).Infof("Docker server is not ready yet: %v", err)
				continue
			}
			glog.V(4).Infof("Docker server is ready")
			return nil
		}
	}
	return fmt.Errorf("docker server start timeout %v", startTimeout)
}

// Stop stops the dockershim grpc server.
func (s *DockerServer) Stop() {
	glog.V(2).Infof("Stop docker server")
	s.server.Stop()
	s.listener.Close()
}
