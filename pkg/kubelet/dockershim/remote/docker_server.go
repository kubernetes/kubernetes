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

	"github.com/golang/glog"
	"google.golang.org/grpc"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/util/interrupt"
)

// DockerServer is the grpc server of dockershim.
type DockerServer struct {
	// endpoint is the endpoint to serve on.
	endpoint string
	// service is the docker service which implements runtime and image services.
	service DockerService
	// server is the grpc server.
	server *grpc.Server
}

// NewDockerServer creates the dockershim grpc server.
func NewDockerServer(endpoint string, s dockershim.DockerService) *DockerServer {
	return &DockerServer{
		endpoint: endpoint,
		service:  NewDockerService(s),
	}
}

// Start starts the dockershim grpc server.
func (s *DockerServer) Start() error {
	glog.V(2).Infof("Start dockershim grpc server")
	l, err := util.CreateListener(s.endpoint)
	if err != nil {
		return fmt.Errorf("failed to listen on %q: %v", s.endpoint, err)
	}
	// Create the grpc server and register runtime and image services.
	s.server = grpc.NewServer()
	runtimeapi.RegisterRuntimeServiceServer(s.server, s.service)
	runtimeapi.RegisterImageServiceServer(s.server, s.service)
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
