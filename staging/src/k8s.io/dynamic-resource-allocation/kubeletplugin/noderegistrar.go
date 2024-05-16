/*
Copyright 2022 The Kubernetes Authors.

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

package kubeletplugin

import (
	"fmt"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"
	registerapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
)

type nodeRegistrar struct {
	logger klog.Logger
	registrationServer
	server *grpcServer
}

// startRegistrar returns a running instance.
func startRegistrar(logger klog.Logger, grpcVerbosity int, interceptors []grpc.UnaryServerInterceptor, streamInterceptors []grpc.StreamServerInterceptor, driverName string, endpoint string, pluginRegistrationEndpoint endpoint) (*nodeRegistrar, error) {
	n := &nodeRegistrar{
		logger: logger,
		registrationServer: registrationServer{
			driverName:        driverName,
			endpoint:          endpoint,
			supportedVersions: []string{"1.0.0"}, // TODO: is this correct?
		},
	}
	s, err := startGRPCServer(logger, grpcVerbosity, interceptors, streamInterceptors, pluginRegistrationEndpoint, func(grpcServer *grpc.Server) {
		registerapi.RegisterRegistrationServer(grpcServer, n)
	})
	if err != nil {
		return nil, fmt.Errorf("start gRPC server: %v", err)
	}
	n.server = s
	return n, nil
}

// stop ensures that the registrar is not running anymore and cleans up all resources.
// It is idempotent and may be called with a nil pointer.
func (s *nodeRegistrar) stop() {
	if s == nil {
		return
	}
	s.server.stop()
}
