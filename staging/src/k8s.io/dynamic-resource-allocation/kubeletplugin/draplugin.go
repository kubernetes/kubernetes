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
	"context"
	"errors"
	"fmt"
	"net"
	"sync"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"

	resourceapi "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes"
	restproxypb "k8s.io/dynamic-resource-allocation/apis/restproxy/v1alpha1"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/dynamic-resource-allocation/restproxy"
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

	// PublishResources may be called one or more times to publish
	// resource information in ResourceSlice objects.
	//
	// PublishResources does not block, so it might still take a while
	// after it returns before all information is actually written
	// to the API server.
	//
	// The caller must not modify the content of the slice.
	PublishResources(ctx context.Context, nodeResources []*resourceapi.ResourceModel)

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
		o.unaryInterceptors = append(o.unaryInterceptors, interceptor)
		return nil
	}
}

// GRPCStreamInterceptor is called for each gRPC streaming method call. This option
// may be used more than once and each interceptor will get called.
func GRPCStreamInterceptor(interceptor grpc.StreamServerInterceptor) Option {
	return func(o *options) error {
		o.streamInterceptors = append(o.streamInterceptors, interceptor)
		return nil
	}
}

// NodeV1alpha3 explicitly chooses whether the DRA gRPC API v1alpha3
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
	unaryInterceptors          []grpc.UnaryServerInterceptor
	streamInterceptors         []grpc.StreamServerInterceptor

	nodeV1alpha3 bool
}

// draPlugin combines the kubelet registration service and the DRA node plugin
// service.
type draPlugin struct {
	ctx        context.Context // For new background activities that are started later.
	wg         sync.WaitGroup
	cancel     func(cause error)
	registrar  *nodeRegistrar
	plugin     *grpcServer
	driverName string
	*restproxy.RoundTripper

	// Information about resource publishing changes concurrently and thus
	// must be protected by the mutex.
	mutex                   sync.Mutex
	nodeName                string
	nodeUID                 types.UID
	publishResources        bool
	nodeResources           []*resourceapi.ResourceModel
	resourceSliceOwner      resourceslice.Owner
	resourceSliceController *resourceslice.Controller
}

// Start sets up two gRPC servers (one for registration, one for the DRA node
// client). By default, all APIs implemented by the nodeServer get registered.
//
// The context and/or DRAPlugin.Stop can be used to stop all background activity.
// Stop also blocks. A logger can be stored in the context to add values or
// a name to all log entries.
func Start(ctx context.Context, nodeServer interface{}, opts ...Option) (result DRAPlugin, finalErr error) {
	logger := klog.FromContext(ctx)
	o := options{
		logger:        klog.Background(),
		grpcVerbosity: 6, // Logs requests and responses, which can be large.
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

	d := &draPlugin{
		driverName: o.driverName,
	}

	// Stop calls cancel and therefore cancellation
	// and Stop both cause goroutines to stop.
	d.ctx, d.cancel = context.WithCancelCause(ctx)
	ctx = d.ctx
	logger.V(3).Info("Starting")
	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		defer logger.V(3).Info("Stopping")
		<-ctx.Done()
	}()

	// Clean up if we don't finish succcessfully.
	defer func() {
		if r := recover(); r != nil {
			d.Stop()
			panic(r)
		}
		if finalErr != nil {
			d.Stop()
		}
	}()

	// Prepare for sending REST calls through kubelet. The returned object implements the
	// restproxy gRPC service, so we register that below. The NodeObject call in that service
	// gets intercepted and, once called, will start publishing resources.
	d.RoundTripper = restproxy.StartRoundTripper(klog.NewContext(ctx, klog.LoggerWithName(logger, "roundtripper")))

	// Run the node plugin gRPC server first to ensure that it is ready.
	implemented := false
	plugin, err := startGRPCServer(klog.NewContext(ctx, klog.LoggerWithName(logger, "dra")), o.grpcVerbosity, o.unaryInterceptors, o.streamInterceptors, o.draEndpoint, func(grpcServer *grpc.Server) {
		if nodeServer, ok := nodeServer.(drapbv1alpha3.NodeServer); ok && o.nodeV1alpha3 {
			logger.V(5).Info("registering drapbv1alpha3.NodeServer")
			drapbv1alpha3.RegisterNodeServer(grpcServer, nodeServer)
			implemented = true
		}
		restproxypb.RegisterRESTServer(grpcServer, d)
	})
	if err != nil {
		return nil, fmt.Errorf("start node client: %v", err)
	}
	d.plugin = plugin
	if !implemented {
		return nil, errors.New("no supported DRA gRPC API is implemented and enabled")
	}

	// Now make it available to kubelet.
	registrar, err := startRegistrar(klog.NewContext(ctx, klog.LoggerWithName(logger, "registrar")), o.grpcVerbosity, o.unaryInterceptors, o.streamInterceptors, o.driverName, o.draAddress, o.pluginRegistrationEndpoint)
	if err != nil {
		return nil, fmt.Errorf("start registrar: %v", err)
	}
	d.registrar = registrar

	// startGRPCServer and startRegistrar don't implement cancellation
	// themselves, we add that for both here.
	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		<-ctx.Done()
		d.plugin.stop()
		d.registrar.stop()
	}()

	return d, nil
}

func (d *draPlugin) NodeObject(ctx context.Context, req *restproxypb.NodeObjectRequest) (*restproxypb.NodeObjectResponse, error) {
	if req.Name == "" || req.Uid == "" {
		return nil, errors.New("name and UID must be set")
	}

	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.nodeName = req.Name
	d.nodeUID = types.UID(req.Uid)
	d.updatePublishResources()

	return &restproxypb.NodeObjectResponse{}, nil
}

func (d *draPlugin) PublishResources(ctx context.Context, nodeResources []*resourceapi.ResourceModel) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.publishResources = true
	d.nodeResources = nodeResources
	d.updatePublishResources()
}

// updatePublishResources is called each time something changed that might
// affect how resources are published.
func (d *draPlugin) updatePublishResources() {
	ctx := d.ctx
	logger := klog.FromContext(ctx)
	if ctx.Err() != nil {
		// Already stopped, don't do anything.
		return
	}

	if !d.publishResources {
		// Nothing to publish (yet).
		return
	}

	if d.nodeUID == "" || d.nodeName == "" {
		// Still need the NodeObject call.
		return
	}

	owner := resourceslice.Owner{
		APIVersion: "v1",
		Kind:       "Node",
		Name:       d.nodeName,
		UID:        d.nodeUID,
	}

	resources := &resourceslice.Resources{NodeResources: d.nodeResources}

	if d.resourceSliceOwner == owner && d.resourceSliceController != nil {
		// Update the existing controller.
		d.resourceSliceController.Update(resources)
		return
	}

	if d.resourceSliceOwner != owner && d.resourceSliceController != nil {
		// Node object changed? Unlikely, but in this case, we stop
		// syncing with the old owner and start anew. The old
		// ResourceSlice objects should get deleted automatically.
		d.resourceSliceController.Stop()
		d.resourceSliceController = nil
	}

	restConfig := d.RoundTripper.NewRESTConfig()
	kubeClient, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		// Should not happen.
		utilruntime.HandleError(fmt.Errorf("creating client set for REST config provided by REST proxy round-tripper: %v", err))
		return
	}
	d.resourceSliceController = resourceslice.StartController(klog.NewContext(ctx, klog.LoggerWithName(logger, "ResourceSlice controller")), kubeClient, d.driverName, owner, resources)
	d.resourceSliceOwner = owner
}

func (d *draPlugin) Stop() {
	if d == nil {
		return
	}
	d.cancel(errors.New("DRA plugin was stopped"))
	d.registrar.stop()
	d.plugin.stop()
	d.RoundTripper.Stop()

	// d.resourceSliceController is set concurrently.
	d.mutex.Lock()
	d.resourceSliceController.Stop()
	d.mutex.Unlock()

	d.wg.Wait()
}

func (d *draPlugin) RegistrationStatus() *registerapi.RegistrationStatus {
	if d.registrar == nil {
		return nil
	}
	return d.registrar.status
}

func (d *draPlugin) internal() {}
