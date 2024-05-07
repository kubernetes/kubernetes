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
	"errors"
	"fmt"
	"net"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"

	drapbv1alpha2 "k8s.io/kubelet/pkg/apis/dra/v1alpha2"
	drapbv1alpha3 "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
	registerapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
)

// DRAPlugin gets returned by Start and defines the public API of the generic
// dynamic resource allocation plugin.
type DRAPlugin interface {
	// Stop ensures that all spawned goroutines are stopped and frees
	// resources.
	Stop()

	// RegistrationStatus returns the result of registration, nil if none
	// received yet.
	RegistrationStatus() *registerapi.RegistrationStatus

	// This unexported method ensures that we can modify the interface
	// without causing an API break of the package
	// (https://pkg.go.dev/golang.org/x/exp/apidiff#section-readme).
	internal()
}

// Option implements the functional options pattern for Start.
type Option func(o *options) error

// DriverName defines the driver name for the dynamic resource allocation driver.
// Must be set.
func DriverName(driverName string) Option {
	return func(o *options) error {
		o.driverName = driverName
		return nil
	}
}

// Logger overrides the default klog.Background logger.
func Logger(logger klog.Logger) Option {
	return func(o *options) error {
		o.logger = logger
		return nil
	}
}

// GRPCVerbosity sets the verbosity for logging gRPC calls. Default is 4. A negative
// value disables logging.
func GRPCVerbosity(level int) Option {
	return func(o *options) error {
		o.grpcVerbosity = level
		return nil
	}
}

// RegistrarSocketPath sets the file path for a Unix domain socket.
// If RegistrarListener is not used, then Start will remove
// a file at that path, should one exist, and creates a socket
// itself. Otherwise it uses the provided listener and only
// removes the socket at the specified path during shutdown.
//
// At least one of these two options is required.
func RegistrarSocketPath(path string) Option {
	return func(o *options) error {
		o.pluginRegistrationEndpoint.path = path
		return nil
	}
}

// RegistrarListener sets an already created listener for the plugin
// registrarion API. Can be combined with RegistrarSocketPath.
//
// At least one of these two options is required.
func RegistrarListener(listener net.Listener) Option {
	return func(o *options) error {
		o.pluginRegistrationEndpoint.listener = listener
		return nil
	}
}

// PluginSocketPath sets the file path for a Unix domain socket.
// If PluginListener is not used, then Start will remove
// a file at that path, should one exist, and creates a socket
// itself. Otherwise it uses the provided listener and only
// removes the socket at the specified path during shutdown.
//
// At least one of these two options is required.
func PluginSocketPath(path string) Option {
	return func(o *options) error {
		o.draEndpoint.path = path
		return nil
	}
}

// PluginListener sets an already created listener for the dynamic resource
// allocation plugin API. Can be combined with PluginSocketPath.
//
// At least one of these two options is required.
func PluginListener(listener net.Listener) Option {
	return func(o *options) error {
		o.draEndpoint.listener = listener
		return nil
	}
}

// KubeletPluginSocketPath defines how kubelet will connect to the dynamic
// resource allocation plugin. This corresponds to PluginSocketPath, except
// that PluginSocketPath defines the path in the filesystem of the caller and
// KubeletPluginSocketPath in the filesystem of kubelet.
func KubeletPluginSocketPath(path string) Option {
	return func(o *options) error {
		o.draAddress = path
		return nil
	}
}

// GRPCInterceptor is called for each incoming gRPC method call. This option
// may be used more than once and each interceptor will get called.
func GRPCInterceptor(interceptor grpc.UnaryServerInterceptor) Option {
	return func(o *options) error {
		o.interceptors = append(o.interceptors, interceptor)
		return nil
	}
}

// NodeV1alpha2 explicitly chooses whether the DRA gRPC API v1alpha2
// gets enabled.
func NodeV1alpha2(enabled bool) Option {
	return func(o *options) error {
		o.nodeV1alpha2 = enabled
		return nil
	}
}

// NodeV1alpha2 explicitly chooses whether the DRA gRPC API v1alpha3
// gets enabled.
func NodeV1alpha3(enabled bool) Option {
	return func(o *options) error {
		o.nodeV1alpha3 = enabled
		return nil
	}
}

type options struct {
	logger                     klog.Logger
	grpcVerbosity              int
	driverName                 string
	draEndpoint                endpoint
	draAddress                 string
	pluginRegistrationEndpoint endpoint
	interceptors               []grpc.UnaryServerInterceptor

	nodeV1alpha2, nodeV1alpha3 bool
}

// draPlugin combines the kubelet registration service and the DRA node plugin
// service.
type draPlugin struct {
	registrar *nodeRegistrar
	plugin    *grpcServer
}

// Start sets up two gRPC servers (one for registration, one for the DRA node
// client). By default, all APIs implemented by the nodeServer get registered.
func Start(nodeServer interface{}, opts ...Option) (result DRAPlugin, finalErr error) {
	d := &draPlugin{}

	o := options{
		logger:        klog.Background(),
		grpcVerbosity: 4,
		nodeV1alpha2:  true,
		nodeV1alpha3:  true,
	}
	for _, option := range opts {
		if err := option(&o); err != nil {
			return nil, err
		}
	}

	if o.driverName == "" {
		return nil, errors.New("driver name must be set")
	}
	if o.draAddress == "" {
		return nil, errors.New("DRA address must be set")
	}
	var emptyEndpoint endpoint
	if o.draEndpoint == emptyEndpoint {
		return nil, errors.New("a Unix domain socket path and/or listener must be set for the kubelet plugin")
	}
	if o.pluginRegistrationEndpoint == emptyEndpoint {
		return nil, errors.New("a Unix domain socket path and/or listener must be set for the registrar")
	}

	// Run the node plugin gRPC server first to ensure that it is ready.
	implemented := false
	plugin, err := startGRPCServer(klog.LoggerWithName(o.logger, "dra"), o.grpcVerbosity, o.interceptors, o.draEndpoint, func(grpcServer *grpc.Server) {
		if nodeServer, ok := nodeServer.(drapbv1alpha3.NodeServer); ok && o.nodeV1alpha3 {
			o.logger.V(5).Info("registering drapbv1alpha3.NodeServer")
			drapbv1alpha3.RegisterNodeServer(grpcServer, nodeServer)
			implemented = true
		}
		if nodeServer, ok := nodeServer.(drapbv1alpha2.NodeServer); ok && o.nodeV1alpha2 {
			o.logger.V(5).Info("registering drapbv1alpha2.NodeServer")
			drapbv1alpha2.RegisterNodeServer(grpcServer, nodeServer)
			implemented = true
		}
	})
	if err != nil {
		return nil, fmt.Errorf("start node client: %v", err)
	}
	d.plugin = plugin
	defer func() {
		// Clean up if we didn't finish succcessfully.
		if r := recover(); r != nil {
			plugin.stop()
			panic(r)
		}
		if finalErr != nil {
			plugin.stop()
		}
	}()
	if !implemented {
		return nil, errors.New("no supported DRA gRPC API is implemented and enabled")
	}

	// Now make it available to kubelet.
	registrar, err := startRegistrar(klog.LoggerWithName(o.logger, "registrar"), o.grpcVerbosity, o.interceptors, o.driverName, o.draAddress, o.pluginRegistrationEndpoint)
	if err != nil {
		return nil, fmt.Errorf("start registrar: %v", err)
	}
	d.registrar = registrar

	return d, nil
}

func (d *draPlugin) Stop() {
	if d == nil {
		return
	}
	d.registrar.stop()
	d.plugin.stop()
}

func (d *draPlugin) RegistrationStatus() *registerapi.RegistrationStatus {
	if d.registrar == nil {
		return nil
	}
	return d.registrar.status
}

func (d *draPlugin) internal() {}
