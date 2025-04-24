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
	"os"
	"path"
	"sync"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"

	"go.etcd.io/etcd/client/pkg/v3/fileutil"
	resourceapi "k8s.io/api/resource/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	registerapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
)

const (
	// KubeletPluginsDir is the default directory for [PluginDataDirectoryPath].
	KubeletPluginsDir = "/var/lib/kubelet/plugins"
	// KubeletRegistryDir is the default for [RegistrarDirectoryPath]
	KubeletRegistryDir = "/var/lib/kubelet/plugins_registry"
)

// DRAPlugin is the interface that needs to be implemented by a DRA driver to
// use this helper package. The helper package then implements the gRPC
// interface expected by the kubelet by wrapping the DRAPlugin implementation.
type DRAPlugin interface {
	// PrepareResourceClaims is called to prepare all resources allocated
	// for the given ResourceClaims. This is used to implement
	// the gRPC NodePrepareResources call.
	//
	// It gets called with the complete list of claims that are needed
	// by some pod. In contrast to the gRPC call, the helper has
	// already retrieved the actual ResourceClaim objects.
	//
	// In addition to that, the helper also:
	// - verifies that all claims are really allocated
	// - increments a numeric counter for each call and
	//   adds its value to a per-context logger with "requestID" as key
	// - adds the method name with "method" as key to that logger
	// - logs the gRPC call and response (configurable with GRPCVerbosity)
	// - serializes all gRPC calls unless the driver explicitly opted out of that
	//
	// This call must be idempotent because the kubelet might have to ask
	// for preparation multiple times, for example if it gets restarted.
	//
	// A DRA driver should verify that all devices listed in a
	// [resourceapi.DeviceRequestAllocationResult] are not already in use
	// for some other ResourceClaim. Kubernetes tries very hard to ensure
	// that, but if something went wrong, then the DRA driver is the last
	// line of defense against using the same device for two different
	// unrelated workloads.
	//
	// If an error is returned, the result is ignored. Otherwise the result
	// must have exactly one entry for each claim, identified by the UID of
	// the corresponding ResourceClaim. For each claim, preparation
	// can be either successful (no error set in the per-ResourceClaim PrepareResult)
	// or can be reported as failed.
	//
	// It is possible to create the CDI spec files which define the CDI devices
	// on-the-fly in PrepareResourceClaims. UnprepareResourceClaims then can
	// remove them. Container runtimes may cache CDI specs but must reload
	// files in case of a cache miss. To avoid false cache hits, the unique
	// name in the CDI device ID should not be reused. A DRA driver can use
	// the claim UID for it.
	PrepareResourceClaims(ctx context.Context, claims []*resourceapi.ResourceClaim) (result map[types.UID]PrepareResult, err error)

	// UnprepareResourceClaims must undo whatever work PrepareResourceClaims did.
	//
	// At the time when this gets called, the original ResourceClaims may have
	// been deleted already. They also don't get cached by the kubelet. Therefore
	// parameters for each ResourcClaim are only the UID, namespace and name.
	// It is the responsibility of the DRA driver to cache whatever additional
	// information it might need about prepared resources.
	//
	// This call must be idempotent because the kubelet might have to ask
	// for un-preparation multiple times, for example if it gets restarted.
	// Therefore it is not an error if this gets called for a ResourceClaim
	// which is not currently prepared.
	//
	// As with PrepareResourceClaims, the helper takes care of logging
	// and serialization.
	//
	// The conventions for returning one overall error and several per-ResourceClaim
	// errors are the same as in PrepareResourceClaims.
	UnprepareResourceClaims(ctx context.Context, claims []NamespacedObject) (result map[types.UID]error, err error)
}

// PrepareResult contains the result of preparing one particular ResourceClaim.
type PrepareResult struct {
	// Err, if non-nil, describes a problem that occurred while preparing
	// the ResourceClaim. The devices are then ignored and the kubelet will
	// try to prepare the ResourceClaim again later.
	Err error

	// Devices contains the IDs of CDI devices associated with specific requests
	// in a ResourceClaim. Those IDs will be passed on to the container runtime
	// by the kubelet.
	//
	// The empty slice is also valid.
	Devices []Device
}

// Device provides the CDI device IDs for one request in a ResourceClaim.
type Device struct {
	// Requests lists the names of requests or subrequests in the
	// ResourceClaim that this device is associated with. The subrequest
	// name may be included here, but it is also okay to just return
	// the request name.
	//
	// A DRA driver can get this string from the Request field in
	// [resourceapi.DeviceRequestAllocationResult], which includes the
	// subrequest name if there is one.
	//
	// If empty, the device is associated with all requests.
	Requests []string

	// PoolName identifies the DRA driver's pool which contains the device.
	// Must not be empty.
	PoolName string

	// DeviceName identifies the device inside that pool.
	// Must not be empty.
	DeviceName string

	// CDIDeviceIDs lists all CDI devices associated with this DRA device.
	// Each ID must be of the form "<vendor ID>/<class>=<unique name>".
	// May be empty.
	CDIDeviceIDs []string
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

// GRPCVerbosity sets the verbosity for logging gRPC calls.
// Default is 6, which includes gRPC calls and their responses.
// A negative value disables logging.
func GRPCVerbosity(level int) Option {
	return func(o *options) error {
		o.grpcVerbosity = level
		return nil
	}
}

// RegistrarDirectoryPath sets the path to the directory where the kubelet
// expects to find registration sockets of plugins. Typically this is
// /var/lib/kubelet/plugins_registry with /var/lib/kubelet being the kubelet's
// data directory.
//
// This is also the default. Some Kubernetes clusters may use a different data directory.
// This path must be the same inside and outside of the driver's container.
// The directory must exist.
func RegistrarDirectoryPath(path string) Option {
	return func(o *options) error {
		o.pluginRegistrationEndpoint.dir = path
		return nil
	}
}

// RegistrarSocketFilename sets the name of the socket inside the directory where
// the kubelet watches for registration sockets (see RegistrarDirectoryPath).
//
// Usually DRA drivers should not need this option. It is provided to
// support updates from an installation which used an older release of
// of the helper code.
//
// The default is <driver name>-reg.sock. When rolling updates are enabled,
// it is <driver name>-<uid>-reg.sock.
//
// This option and [RollingUpdate] are mutually exclusive.
func RegistrarSocketFilename(name string) Option {
	return func(o *options) error {
		o.pluginRegistrationEndpoint.file = name
		return nil
	}
}

// RegistrarListener configures how to create the registrar socket.
// The default is to remove the file if it exists and to then
// create a socket.
//
// This is used in Kubernetes for end-to-end testing. The default should
// be fine for DRA drivers.
func RegistrarListener(listen func(ctx context.Context, path string) (net.Listener, error)) Option {
	return func(o *options) error {
		o.pluginRegistrationEndpoint.listenFunc = listen
		return nil
	}
}

// PluginDataDirectoryPath sets the path where the DRA driver creates the
// "dra.sock" socket that the kubelet connects to for the DRA-specific gRPC calls.
// It is also used to coordinate between different Pods when using rolling
// updates. It must not be shared with other kubelet plugins.
//
// The default is /var/lib/kubelet/plugins/<driver name>. This directory
// does not need to be inside the kubelet data directory, as long as
// the kubelet can access it.
//
// This path must be the same inside and outside of the driver's container.
// The directory must exist.
func PluginDataDirectoryPath(path string) Option {
	return func(o *options) error {
		o.pluginDataDirectoryPath = path
		return nil
	}
}

// PluginListener configures how to create the registrar socket.
// The default is to remove the file if it exists and to then
// create a socket.
//
// This is used in Kubernetes for end-to-end testing. The default should
// be fine for DRA drivers.
func PluginListener(listen func(ctx context.Context, path string) (net.Listener, error)) Option {
	return func(o *options) error {
		o.draEndpointListen = listen
		return nil
	}
}

// RollingUpdate can be used to enable support for running two plugin instances
// in parallel while a newer instance replaces the older. When enabled, both
// instances must share the same plugin data directory and driver name.
// They create different sockets to allow the kubelet to connect to both at
// the same time.
//
// There is no guarantee which of the two instances are used by kubelet.
// For example, it can happen that a claim gets prepared by one instance
// and then needs to be unprepared by the other. Kubelet then may fall back
// to the first one again for some other operation. In practice this means
// that each instance must be entirely stateless across method calls.
// Serialization (on by default, see [Serialize]) ensures that methods
// are serialized across all instances through file locking. The plugin
// implementation can load shared state from a file at the start
// of a call, execute and then store the updated shared state again.
//
// Passing a non-empty uid enables rolling updates, an empty uid disables it.
// The uid must be the pod UID. A DaemonSet can pass that into the driver container
// via the downward API (https://kubernetes.io/docs/concepts/workloads/pods/downward-api/#downwardapi-fieldRef).
//
// Because new instances cannot remove stale sockets of older instances,
// it is important that each pod shuts down cleanly: it must catch SIGINT/TERM
// and stop the helper instead of quitting immediately.
//
// This depends on support in the kubelet which was added in Kubernetes 1.33.
// Don't use this if it is not certain that the kubelet has that support!
//
// This option and [RegistrarSocketFilename] are mutually exclusive.
func RollingUpdate(uid types.UID) Option {
	return func(o *options) error {
		o.rollingUpdateUID = uid

		// TODO: ask the kubelet whether that pod is still running and
		// clean up leftover sockets?
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

// NodeV1beta1 explicitly chooses whether the DRA gRPC API v1beta1
// gets enabled. True by default.
//
// This is used in Kubernetes for end-to-end testing. The default should
// be fine for DRA drivers.
func NodeV1beta1(enabled bool) Option {
	return func(o *options) error {
		o.nodeV1beta1 = enabled
		return nil
	}
}

// KubeClient grants the plugin access to the API server. This is needed
// for syncing ResourceSlice objects. It's the responsibility of the DRA driver
// developer to ensure that this client has permission to read, write,
// patch and list such objects. It also needs permission to read node objects.
// Ideally, a validating admission policy should be used to limit write
// access to ResourceSlices which belong to the node.
func KubeClient(kubeClient kubernetes.Interface) Option {
	return func(o *options) error {
		o.kubeClient = kubeClient
		return nil
	}
}

// NodeName tells the plugin on which node it is running. This is needed for
// syncing ResourceSlice objects.
func NodeName(nodeName string) Option {
	return func(o *options) error {
		o.nodeName = nodeName
		return nil
	}
}

// NodeUID tells the plugin the UID of the v1.Node object. This is used
// when syncing ResourceSlice objects, but doesn't have to be used. If
// not supplied, the controller will look up the object once.
func NodeUID(nodeUID types.UID) Option {
	return func(o *options) error {
		o.nodeUID = nodeUID
		return nil
	}
}

// Serialize overrides whether the helper serializes the prepare and unprepare
// calls. The default is to serialize.
//
// A DRA driver can opt out of that to speed up parallel processing, but then
// must handle concurrency itself.
func Serialize(enabled bool) Option {
	return func(o *options) error {
		o.serialize = enabled
		return nil
	}
}

// FlockDir changes where lock files are created and locked. A lock file
// is needed when serializing gRPC calls and rolling updates are enabled.
// The directory must exist and be reserved for exclusive use by the
// driver. The default is the plugin data directory.
func FlockDirectoryPath(path string) Option {
	return func(o *options) error {
		o.flockDirectoryPath = path
		return nil
	}
}

type options struct {
	logger                     klog.Logger
	grpcVerbosity              int
	driverName                 string
	nodeName                   string
	nodeUID                    types.UID
	pluginRegistrationEndpoint endpoint
	pluginDataDirectoryPath    string
	rollingUpdateUID           types.UID
	draEndpointListen          func(ctx context.Context, path string) (net.Listener, error)
	unaryInterceptors          []grpc.UnaryServerInterceptor
	streamInterceptors         []grpc.StreamServerInterceptor
	kubeClient                 kubernetes.Interface
	serialize                  bool
	flockDirectoryPath         string
	nodeV1beta1                bool
}

// Helper combines the kubelet registration service and the DRA node plugin
// service and implements them by calling a [DRAPlugin] implementation.
type Helper struct {
	// backgroundCtx is for activities that are started later.
	backgroundCtx context.Context
	// cancel cancels the backgroundCtx.
	cancel           func(cause error)
	wg               sync.WaitGroup
	registrar        *nodeRegistrar
	pluginServer     *grpcServer
	plugin           DRAPlugin
	driverName       string
	nodeName         string
	nodeUID          types.UID
	kubeClient       kubernetes.Interface
	serialize        bool
	grpcMutex        sync.Mutex
	grpcLockFilePath string

	// Information about resource publishing changes concurrently and thus
	// must be protected by the mutex. The controller gets started only
	// if needed.
	mutex                   sync.Mutex
	resourceSliceController *resourceslice.Controller
}

// Start sets up two gRPC servers (one for registration, one for the DRA node
// client) and implements them by calling a [DRAPlugin] implementation.
//
// The context and/or DRAPlugin.Stop can be used to stop all background activity.
// Stop also blocks. A logger can be stored in the context to add values or
// a name to all log entries.
//
// If the plugin will be used to publish resources, [KubeClient] and [NodeName]
// options are mandatory. Otherwise only [DriverName] is mandatory.
func Start(ctx context.Context, plugin DRAPlugin, opts ...Option) (result *Helper, finalErr error) {
	logger := klog.FromContext(ctx)
	o := options{
		logger:        klog.Background(),
		grpcVerbosity: 6, // Logs requests and responses, which can be large.
		serialize:     true,
		nodeV1beta1:   true,
		pluginRegistrationEndpoint: endpoint{
			dir: KubeletRegistryDir,
		},
	}
	for _, option := range opts {
		if err := option(&o); err != nil {
			return nil, err
		}
	}

	if o.driverName == "" {
		return nil, errors.New("driver name must be set")
	}
	if o.rollingUpdateUID != "" && o.pluginRegistrationEndpoint.file != "" {
		return nil, errors.New("rolling updates and explicit registration socket filename are mutually exclusive")
	}
	uidPart := ""
	if o.rollingUpdateUID != "" {
		uidPart = "-" + string(o.rollingUpdateUID)
	}
	if o.pluginRegistrationEndpoint.file == "" {
		o.pluginRegistrationEndpoint.file = o.driverName + uidPart + "-reg.sock"
	}
	if o.pluginDataDirectoryPath == "" {
		o.pluginDataDirectoryPath = path.Join(KubeletPluginsDir, o.driverName)
	}

	d := &Helper{
		driverName: o.driverName,
		nodeName:   o.nodeName,
		nodeUID:    o.nodeUID,
		kubeClient: o.kubeClient,
		serialize:  o.serialize,
		plugin:     plugin,
	}
	if o.rollingUpdateUID != "" {
		dir := o.pluginDataDirectoryPath
		if o.flockDirectoryPath != "" {
			dir = o.flockDirectoryPath
		}
		// Enable file locking, required for concurrently running pods.
		d.grpcLockFilePath = path.Join(dir, "serialize.lock")
	}

	// Stop calls cancel and therefore both cancellation
	// and Stop cause goroutines to stop.
	ctx, cancel := context.WithCancelCause(ctx)
	d.backgroundCtx, d.cancel = ctx, cancel
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

	// Run the node plugin gRPC server first to ensure that it is ready.
	var supportedServices []string
	draEndpoint := endpoint{
		dir:        o.pluginDataDirectoryPath,
		file:       "dra" + uidPart + ".sock", // "dra" is hard-coded. The directory is unique, so we get a unique full path also without the UID.
		listenFunc: o.draEndpointListen,
	}
	pluginServer, err := startGRPCServer(klog.LoggerWithName(logger, "dra"), o.grpcVerbosity, o.unaryInterceptors, o.streamInterceptors, draEndpoint, func(grpcServer *grpc.Server) {
		if o.nodeV1beta1 {
			logger.V(5).Info("registering v1beta1.DRAPlugin gRPC service")
			drapb.RegisterDRAPluginServer(grpcServer, &nodePluginImplementation{Helper: d})
			supportedServices = append(supportedServices, drapb.DRAPluginService)
		}
	})
	if err != nil {
		return nil, fmt.Errorf("start node client: %v", err)
	}
	d.pluginServer = pluginServer
	if len(supportedServices) == 0 {
		return nil, errors.New("no supported DRA gRPC API is implemented and enabled")
	}

	// Now make it available to kubelet.
	registrar, err := startRegistrar(klog.LoggerWithName(logger, "registrar"), o.grpcVerbosity, o.unaryInterceptors, o.streamInterceptors, o.driverName, supportedServices, draEndpoint.path(), o.pluginRegistrationEndpoint)
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

		// Time to stop.
		d.pluginServer.stop()
		d.registrar.stop()

		// d.resourceSliceController is set concurrently.
		d.mutex.Lock()
		d.resourceSliceController.Stop()
		d.mutex.Unlock()
	}()

	return d, nil
}

// Stop ensures that all spawned goroutines are stopped and frees resources.
func (d *Helper) Stop() {
	if d == nil {
		return
	}
	d.cancel(errors.New("DRA plugin was stopped"))
	// Wait for goroutines in Start to clean up and exit.
	d.wg.Wait()
}

// PublishResources may be called one or more times to publish
// resource information in ResourceSlice objects. If it never gets
// called, then the kubelet plugin does not manage any ResourceSlice
// objects.
//
// PublishResources does not block, so it might still take a while
// after it returns before all information is actually written
// to the API server.
//
// It is the responsibility of the caller to ensure that the pools and
// slices described in the driver resources parameters are valid
// according to the restrictions defined in the resource.k8s.io API.
//
// Invalid ResourceSlices will be rejected by the apiserver during
// publishing, which happens asynchronously and thus does not
// get returned as error here. The only error returned here is
// when publishing was not set up properly, for example missing
// [KubeClient] or [NodeName] options.
//
// The caller may modify the resources after this call returns.
func (d *Helper) PublishResources(_ context.Context, resources resourceslice.DriverResources) error {
	if d.kubeClient == nil {
		return errors.New("no KubeClient found to publish resources")
	}
	if d.nodeName == "" {
		return errors.New("no NodeName was set to publish resources")
	}

	d.mutex.Lock()
	defer d.mutex.Unlock()

	owner := resourceslice.Owner{
		APIVersion: "v1",
		Kind:       "Node",
		Name:       d.nodeName,
		UID:        d.nodeUID, // Optional, will be determined by controller if empty.
	}
	driverResources := &resourceslice.DriverResources{
		Pools: resources.Pools,
	}

	if d.resourceSliceController == nil {
		// Start publishing the information. The controller is using
		// our background context, not the one passed into this
		// function, and thus is connected to the lifecycle of the
		// plugin.
		//
		// TODO: don't delete ResourceSlices, not even on a clean shutdown.
		// We either support rolling updates and want to hand over seamlessly
		// or don't and then perhaps restart the pod quickly enough that
		// the kubelet hasn't deleted ResourceSlices yet.
		controllerCtx := d.backgroundCtx
		controllerLogger := klog.FromContext(controllerCtx)
		controllerLogger = klog.LoggerWithName(controllerLogger, "ResourceSlice controller")
		controllerCtx = klog.NewContext(controllerCtx, controllerLogger)
		var err error
		if d.resourceSliceController, err = resourceslice.StartController(controllerCtx,
			resourceslice.Options{
				DriverName: d.driverName,
				KubeClient: d.kubeClient,
				Owner:      &owner,
				Resources:  driverResources,
			}); err != nil {
			return fmt.Errorf("start ResourceSlice controller: %w", err)
		}
		return nil
	}

	// Inform running controller about new information.
	d.resourceSliceController.Update(driverResources)

	return nil
}

// RegistrationStatus returns the result of registration, nil if none received yet.
func (d *Helper) RegistrationStatus() *registerapi.RegistrationStatus {
	if d.registrar == nil {
		return nil
	}
	// TODO: protect against concurrency issues.
	return d.registrar.status
}

// serializeGRPCIfEnabled locks a mutex if serialization is enabled.
// Either way it returns a method that the caller must invoke
// via defer.
func (d *Helper) serializeGRPCIfEnabled() (func(), error) {
	if !d.serialize {
		return func() {}, nil
	}

	// If rolling updates are enabled, we cannot do only in-memory locking.
	// We must use file locking.
	if d.grpcLockFilePath != "" {
		file, err := fileutil.LockFile(d.grpcLockFilePath, os.O_RDWR|os.O_CREATE, 0666)
		if err != nil {
			return nil, fmt.Errorf("lock file: %w", err)
		}
		return func() {
			_ = file.Close()
		}, nil
	}

	d.grpcMutex.Lock()
	return d.grpcMutex.Unlock, nil
}

// nodePluginImplementation is a thin wrapper around the helper instance.
// It prevents polluting the public API with these implementation details.
type nodePluginImplementation struct {
	*Helper
}

// NodePrepareResources implements [drapb.NodePrepareResources].
func (d *nodePluginImplementation) NodePrepareResources(ctx context.Context, req *drapb.NodePrepareResourcesRequest) (*drapb.NodePrepareResourcesResponse, error) {
	// Do slow API calls before serializing.
	claims, err := d.getResourceClaims(ctx, req.Claims)
	if err != nil {
		return nil, fmt.Errorf("get resource claims: %w", err)
	}

	unlock, err := d.serializeGRPCIfEnabled()
	if err != nil {
		return nil, fmt.Errorf("serialize gRPC: %w", err)
	}
	defer unlock()

	result, err := d.plugin.PrepareResourceClaims(ctx, claims)
	if err != nil {
		return nil, fmt.Errorf("prepare resource claims: %w", err)
	}

	resp := &drapb.NodePrepareResourcesResponse{Claims: map[string]*drapb.NodePrepareResourceResponse{}}
	for uid, claimResult := range result {
		var devices []*drapb.Device
		for _, result := range claimResult.Devices {
			device := &drapb.Device{
				RequestNames: stripSubrequestNames(result.Requests),
				PoolName:     result.PoolName,
				DeviceName:   result.DeviceName,
				CDIDeviceIDs: result.CDIDeviceIDs,
			}
			devices = append(devices, device)
		}
		resp.Claims[string(uid)] = &drapb.NodePrepareResourceResponse{
			Error:   errorString(claimResult.Err),
			Devices: devices,
		}
	}
	return resp, nil
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func stripSubrequestNames(names []string) []string {
	stripped := make([]string, len(names))
	for i, name := range names {
		stripped[i] = resourceclaim.BaseRequestRef(name)
	}
	return stripped
}

func (d *nodePluginImplementation) getResourceClaims(ctx context.Context, claims []*drapb.Claim) ([]*resourceapi.ResourceClaim, error) {
	var resourceClaims []*resourceapi.ResourceClaim
	for _, claimReq := range claims {
		claim, err := d.kubeClient.ResourceV1beta1().ResourceClaims(claimReq.Namespace).Get(ctx, claimReq.Name, metav1.GetOptions{})
		if err != nil {
			return resourceClaims, fmt.Errorf("retrieve claim %s/%s: %w", claimReq.Namespace, claimReq.Name, err)
		}
		if claim.Status.Allocation == nil {
			return resourceClaims, fmt.Errorf("claim %s/%s not allocated", claimReq.Namespace, claimReq.Name)
		}
		if claim.UID != types.UID(claimReq.UID) {
			return resourceClaims, fmt.Errorf("claim %s/%s got replaced", claimReq.Namespace, claimReq.Name)
		}
		resourceClaims = append(resourceClaims, claim)
	}
	return resourceClaims, nil
}

// NodeUnprepareResources implements [draapi.NodeUnprepareResources].
func (d *nodePluginImplementation) NodeUnprepareResources(ctx context.Context, req *drapb.NodeUnprepareResourcesRequest) (*drapb.NodeUnprepareResourcesResponse, error) {
	unlock, err := d.serializeGRPCIfEnabled()
	if err != nil {
		return nil, fmt.Errorf("serialize gRPC: %w", err)
	}
	defer unlock()

	claims := make([]NamespacedObject, 0, len(req.Claims))
	for _, claim := range req.Claims {
		claims = append(claims, NamespacedObject{UID: types.UID(claim.UID), NamespacedName: types.NamespacedName{Name: claim.Name, Namespace: claim.Namespace}})
	}
	result, err := d.plugin.UnprepareResourceClaims(ctx, claims)
	if err != nil {
		return nil, fmt.Errorf("unprepare resource claims: %w", err)
	}

	resp := &drapb.NodeUnprepareResourcesResponse{Claims: map[string]*drapb.NodeUnprepareResourceResponse{}}
	for uid, err := range result {
		resp.Claims[string(uid)] = &drapb.NodeUnprepareResourceResponse{
			Error: errorString(err),
		}
	}
	return resp, nil
}
