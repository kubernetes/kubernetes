/*
Copyright 2019 The Kubernetes Authors.

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

package plugin

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	grpcstats "google.golang.org/grpc/stats"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	timedworkers "k8s.io/kubernetes/pkg/controller/tainteviction" // TODO (?): move this common helper somewhere else?
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	"k8s.io/utils/ptr"
)

// DRAPluginManager keeps track of how to reach plugins registered for DRA drivers.
// Each plugin has a gRPC endpoint. There may be more than one plugin per driver.
//
// To be informed about available plugins, the DRAPluginManager implements the
// [cache.PluginHandler] interface and needs to be added to the
// plugin manager.
//
// The null DRAPluginManager is not usable, use NewDRAPluginManager.
type DRAPluginManager struct {
	// backgroundCtx is used for all future activities of the DRAPluginManager.
	// This is necessary because it implements APIs which don't
	// provide a context.
	backgroundCtx context.Context
	cancel        func(err error)
	kubeClient    kubernetes.Interface
	getNode       func() (*v1.Node, error)
	wipingDelay   time.Duration
	streamHandler StreamHandler

	wg    sync.WaitGroup
	mutex sync.RWMutex

	// driver name -> DRAPlugin in the order in which they got added
	store map[string][]*monitoredPlugin

	// pendingWipes tracks at which time ResourceSlices for a
	// DRA driver should be removed. The removal then happens in
	// the background in a callback function that is invoked
	// by the TimedWorkerQueue.
	//
	// TimedWorkerQueue uses namespace/name as key. We use
	// the driver name as name with no namespace.
	pendingWipes *timedworkers.TimedWorkerQueue
}

var _ cache.PluginHandler = &DRAPluginManager{}

// monitoredPlugin tracks whether the gRPC connection of a plugin is
// currently connected. Fot that it implements the [grpcstats.Handler]
// interface.
//
// The tagging functions might be useful for contextual logging. But
// for now all that matters is HandleConn.
type monitoredPlugin struct {
	*DRAPlugin
	pm *DRAPluginManager

	// connected is protected by store.mutex.
	connected bool
}

var _ grpcstats.Handler = &monitoredPlugin{}

func (m *monitoredPlugin) TagRPC(ctx context.Context, info *grpcstats.RPCTagInfo) context.Context {
	return ctx
}

func (m *monitoredPlugin) HandleRPC(context.Context, grpcstats.RPCStats) {
}

func (m *monitoredPlugin) TagConn(ctx context.Context, info *grpcstats.ConnTagInfo) context.Context {
	return ctx
}

func (m *monitoredPlugin) HandleConn(_ context.Context, stats grpcstats.ConnStats) {
	connected := false
	switch stats.(type) {
	case *grpcstats.ConnBegin:
		connected = true
	case *grpcstats.ConnEnd:
		// We have to ask for a reconnect, otherwise gRPC wouldn't try and
		// thus we wouldn't be notified about a restart of the plugin.
		m.conn.Connect()
	default:
		return
	}
	if m.pm.backgroundCtx.Err() != nil {
		// Shutting down, no longer interested in connection changes...
		return
	}
	logger := klog.FromContext(m.pm.backgroundCtx)
	m.pm.mutex.Lock()
	defer m.pm.mutex.Unlock()
	logger.V(2).Info("Connection changed", "driverName", m.driverName, "endpoint", m.endpoint, "connected", connected)
	m.connected = connected
	m.pm.sync(m.driverName)
}

// NewDRAPluginManager creates a new DRAPluginManager, with support for wiping ResourceSlices
// when the plugin(s) for a DRA driver are not available too long.
//
// The context can be used to cancel all background activities.
// If desired, Stop can be called in addition or instead of canceling
// the context. It then also waits for background activities to stop.
func NewDRAPluginManager(ctx context.Context, kubeClient kubernetes.Interface, getNode func() (*v1.Node, error), streamHandler StreamHandler, wipingDelay time.Duration) *DRAPluginManager {
	ctx, cancel := context.WithCancelCause(ctx)
	pm := &DRAPluginManager{
		backgroundCtx: klog.NewContext(ctx, klog.LoggerWithName(klog.FromContext(ctx), "DRA registration handler")),
		cancel:        cancel,
		kubeClient:    kubeClient,
		getNode:       getNode,
		wipingDelay:   wipingDelay,
		streamHandler: streamHandler,
	}
	pm.pendingWipes = timedworkers.CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *timedworkers.WorkArgs) error {
		pm.wipeResourceSlices(ctx, args.Object.Name)
		return nil
	})

	// When kubelet starts up, no DRA driver has registered yet. None of
	// the drivers are usable until they come back, which might not happen
	// at all. Therefore it is better to not advertise any local resources
	// because pods could get stuck on the node waiting for the driver
	// to start up.
	//
	// This has to run in the background.
	pm.wg.Add(1)
	go func() {
		defer pm.wg.Done()

		ctx := pm.backgroundCtx
		logger := klog.LoggerWithName(klog.FromContext(ctx), "startup")
		ctx = klog.NewContext(ctx, logger)
		pm.wipeResourceSlices(ctx, "" /* all drivers */)
	}()

	return pm
}

// Stop cancels any remaining background activities and blocks until all goroutines have stopped,
// with one caveat: goroutines created dynamically for wiping ResourceSlices are not tracked.
// They won't do anything because of the context cancellation.
func (pm *DRAPluginManager) Stop() {
	defer pm.wg.Wait() // Must run after unlocking our mutex.
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	logger := klog.FromContext(pm.backgroundCtx)
	pm.cancel(errors.New("Stop was called"))

	// Close all connections, otherwise gRPC keeps doing things in the background.
	// Also cancel all pending wiping.
	for driverName, plugins := range pm.store {
		workerArg := timedworkers.NewWorkArgs(driverName, "")
		pm.pendingWipes.CancelWork(logger, workerArg.KeyFromWorkArgs())
		for _, plugin := range plugins {
			if err := plugin.conn.Close(); err != nil {
				logger.Error(err, "Closing gRPC connection", "driverName", plugin.driverName, "endpoint", plugin.endpoint)
			}
		}
	}
}

// wipeResourceSlices deletes ResourceSlices of the node, optionally just for a specific driver.
//
// It gets called in a stand-alone goroutine at kubelet startup and as callback
// of a TimedWorkersQueue. In both cases the caller has no way of handling errors,
// so wipeResourceSlices must implement it's own retry mechanism.
//
// Can be canceled by canceling the context.
func (pm *DRAPluginManager) wipeResourceSlices(ctx context.Context, driver string) {
	if pm.kubeClient == nil {
		return
	}
	logger := klog.FromContext(ctx)

	backoff := wait.Backoff{
		Duration: time.Second,
		Factor:   2,
		Jitter:   0.2,
		Cap:      5 * time.Minute,
		Steps:    100,
	}

	// Error logging is done inside the loop. Context cancellation doesn't get logged.
	_ = wait.ExponentialBackoffWithContext(ctx, backoff, func(ctx context.Context) (bool, error) {
		node, err := pm.getNode()
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			logger.Error(err, "Unexpected error checking for node")
			return false, nil
		}
		fieldSelector := fields.Set{resourceapi.ResourceSliceSelectorNodeName: node.Name}
		if driver != "" {
			fieldSelector[resourceapi.ResourceSliceSelectorDriver] = driver
		}

		err = pm.kubeClient.ResourceV1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: fieldSelector.String()})
		switch {
		case err == nil:
			logger.V(3).Info("Deleted ResourceSlices", "fieldSelector", fieldSelector)
			return true, nil
		case apierrors.IsUnauthorized(err):
			// This can happen while kubelet is still figuring out
			// its credentials.
			logger.V(5).Info("Deleting ResourceSlice failed, retrying", "fieldSelector", fieldSelector, "err", err)
			return false, nil
		case apierrors.IsNotFound(err):
			logger.V(5).Info("ResourceSlices not found, nothing to delete.", "fieldSelector", fieldSelector)
			return true, nil
		default:
			// Log and retry for other errors.
			logger.V(3).Info("Deleting ResourceSlice failed, retrying", "fieldSelector", fieldSelector, "err", err)
			return false, nil
		}
	})
}

// GetPlugin returns a wrapper around those gRPC methods of a DRA
// driver kubelet plugin which need to be called by kubelet. The wrapper
// handles gRPC connection management and logging. Connections are reused
// across different calls.
//
// It returns an informative error message including the driver name
// with an explanation why the driver is not usable.
func (pm *DRAPluginManager) GetPlugin(driverName string) (*DRAPlugin, error) {
	if driverName == "" {
		return nil, errors.New("DRA driver name is empty")
	}
	plugin := pm.get(driverName)
	if plugin == nil {
		return nil, fmt.Errorf("DRA driver %s is not registered", driverName)
	}
	return plugin, nil
}

// get lets you retrieve a DRA DRAPlugin by name.
func (pm *DRAPluginManager) get(driverName string) *DRAPlugin {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	logger := klog.FromContext(pm.backgroundCtx)

	plugins := pm.store[driverName]
	if len(plugins) == 0 {
		logger.V(5).Info("No plugin registered", "driverName", driverName)
		return nil
	}

	// Heuristic: pick the most recent one. It's most likely
	// the newest, except when kubelet got restarted and registered
	// all running plugins in random order.
	//
	// Prefer plugins which are connected, otherwise also
	// disconnected ones.
	for i := len(plugins) - 1; i >= 0; i-- {
		if plugin := plugins[i]; plugin.connected {
			logger.V(5).Info("Preferring connected plugin", "driverName", driverName, "endpoint", plugin.endpoint)
			return plugin.DRAPlugin
		}
	}
	plugin := plugins[len(plugins)-1]
	logger.V(5).Info("No plugin connected, using latest one", "driverName", driverName, "endpoint", plugin.endpoint)
	return plugin.DRAPlugin
}

// RegisterPlugin implements [cache.PluginHandler].
// It is called by the plugin manager when a plugin is ready to be registered.
//
// Plugins of a DRA driver are required to register under the name of
// the DRA driver.
//
// DRA uses the version array in the registration API to enumerate all gRPC
// services that the plugin provides, using the "<gRPC package name>.<service
// name>" format (e.g. "v1beta1.DRAPlugin"). This allows kubelet to determine
// in advance which version to use resp. which optional services the plugin
// supports.
func (pm *DRAPluginManager) RegisterPlugin(driverName string, endpoint string, supportedServices []string, pluginClientTimeout *time.Duration) error {
	chosenService, err := pm.validateSupportedServices(driverName, supportedServices)
	if err != nil {
		return fmt.Errorf("invalid supported gRPC versions of DRA driver plugin %s at endpoint %s: %w", driverName, endpoint, err)
	}

	timeout := ptr.Deref(pluginClientTimeout, defaultClientCallTimeout)

	// Storing endpoint of newly registered DRA DRAPlugin into the map, where the DRA driver name will be the key
	// under which the manager will be able to get a plugin when it needs to call it.
	if err := pm.add(driverName, endpoint, chosenService, timeout); err != nil {
		// No wrapping, the error already contains details.
		return err
	}

	return nil
}

func (pm *DRAPluginManager) add(driverName string, endpoint string, chosenService string, clientCallTimeout time.Duration) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	p := &DRAPlugin{
		driverName:        driverName,
		endpoint:          endpoint,
		chosenService:     chosenService,
		clientCallTimeout: clientCallTimeout,
		backgroundCtx:     pm.backgroundCtx,
	}
	if pm.store == nil {
		pm.store = make(map[string][]*monitoredPlugin)
	}
	for _, oldP := range pm.store[driverName] {
		if oldP.endpoint == endpoint {
			// One plugin instance cannot hijack the endpoint of another instance.
			return fmt.Errorf("endpoint %s already registered for DRA driver plugin %s", endpoint, driverName)
		}
	}

	logger := klog.FromContext(pm.backgroundCtx)

	mp := &monitoredPlugin{
		DRAPlugin: p,
		pm:        pm,
	}

	// The gRPC connection gets created once. gRPC then connects to the gRPC server on demand.
	target := "unix:" + endpoint
	logger.V(4).Info("Creating new gRPC connection", "target", target)
	conn, err := grpc.NewClient(
		target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithChainUnaryInterceptor(newMetricsInterceptor(driverName)),
		grpc.WithStatsHandler(mp),
	)
	if err != nil {
		return fmt.Errorf("create gRPC connection to DRA driver %s plugin at endpoint %s: %w", driverName, endpoint, err)
	}
	p.conn = conn

	if utilfeature.DefaultFeatureGate.Enabled(features.ResourceHealthStatus) {
		pm.wg.Add(1)
		go func() {
			defer pm.wg.Done()
			streamCtx, streamCancel := context.WithCancel(p.backgroundCtx)
			p.SetHealthStream(streamCtx, streamCancel)

			wait.UntilWithContext(streamCtx, func(ctx context.Context) {
				logger.V(4).Info("Attempting to start WatchResources health stream")
				stream, err := p.NodeWatchResources(ctx)
				if err != nil {
					logger.V(3).Error(err, "Failed to establish WatchResources stream, will retry")
					return
				}

				logger.V(2).Info("Successfully started WatchResources health stream")

				err = pm.streamHandler.HandleWatchResourcesStream(ctx, stream, driverName)
				logger.V(2).Info("WatchResources health stream has ended", "error", err)

			}, 5*time.Second)
		}()
	}

	// Ensure that gRPC tries to connect even if we don't call any gRPC method.
	// This is necessary to detect early whether a plugin is really available.
	// This is currently an experimental gRPC method. Should it be removed we
	// would need to do something else, like sending a fake gRPC method call.
	conn.Connect()

	pm.store[p.driverName] = append(pm.store[p.driverName], mp)
	logger.V(3).Info("Registered DRA plugin", "driverName", p.driverName, "endpoint", p.endpoint, "chosenService", p.chosenService, "numPlugins", len(pm.store[p.driverName]))
	pm.sync(p.driverName)
	return nil
}

// DeRegisterPlugin implements [cache.PluginHandler].
//
// The plugin manager calls it after it has detected that
// the plugin removed its registration socket,
// signaling that it is no longer available.
func (pm *DRAPluginManager) DeRegisterPlugin(driverName, endpoint string) {
	// remove could be removed (no pun intended) but is kept for the sake of symmetry.
	pm.remove(driverName, endpoint)
}

func (pm *DRAPluginManager) remove(driverName, endpoint string) {
	logger := klog.FromContext(pm.backgroundCtx)
	var p *monitoredPlugin
	defer func() {
		// Defer is used to avoid holding the mutex while closing the connection.
		// This is necessary because closing the connection may cause deadlock
		// with DRAPlugin.HandleConn, which can be called while holding the mutex.
		if p != nil && p.conn != nil {
			// Close the gRPC connection, so that it doesn't leak and
			// doesn't try to reconnect to an unregistered plugin.
			if err := p.conn.Close(); err != nil {
				logger.Error(err, "Closing gRPC connection", "driverName", driverName, "endpoint", endpoint)
			}
		}
	}()
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	plugins := pm.store[driverName]
	i := slices.IndexFunc(plugins, func(mp *monitoredPlugin) bool { return mp.driverName == driverName && mp.endpoint == endpoint })
	if i == -1 {
		return
	}
	p = plugins[i]

	last := len(plugins) == 1
	if last {
		delete(pm.store, driverName)
	} else {
		pm.store[driverName] = slices.Delete(plugins, i, i+1)
	}

	// Cancel the plugin's health stream if it was active.
	healthCancel := p.HealthStreamCancel()
	if healthCancel != nil {
		logger.V(4).Info("Canceling health stream during deregistration")
		healthCancel()
	}

	logger.V(3).Info("Unregistered DRA plugin", "driverName", driverName, "endpoint", endpoint, "numPlugins", len(pm.store[driverName]))
	pm.sync(driverName)
}

// sync must be called each time the information about a plugin changes.
// The mutex must be locked for writing.
func (pm *DRAPluginManager) sync(driverName string) {
	if pm.kubeClient == nil {
		// Cannot wipe.
		return
	}
	ctx := pm.backgroundCtx
	logger := klog.FromContext(pm.backgroundCtx)
	workerArgs := timedworkers.NewWorkArgs(driverName, "")

	// Is the DRA driver usable again?
	if pm.usable(driverName) {
		// Yes: cancel any pending ResourceSlice wiping for the DRA driver.
		pm.pendingWipes.CancelWork(logger, workerArgs.KeyFromWorkArgs())
		return
	}

	// No: ensure that we wipe ResourceSlices of the driver.
	// If this was already queued earlier, the original timeout
	// continues to apply because nothing changed.
	if pm.pendingWipes.GetWorkerUnsafe(workerArgs.KeyFromWorkArgs()) != nil {
		// Already queued or potentially already running.
		//
		// There's a small time-of-check-time-of-use race here,
		// but that's fine: if wiping starts after we retrieve
		// the pointer and before checking it, the work gets
		// done, which is what we want.
		return
	}
	now := time.Now()
	fireAt := now.Add(pm.wipingDelay)
	logger = klog.LoggerWithName(logger, "driver-cleanup")
	logger = klog.LoggerWithValues(logger, "driverName", driverName)
	ctx = klog.NewContext(ctx, logger)
	pm.pendingWipes.AddWork(ctx, timedworkers.NewWorkArgs(driverName, ""), now, fireAt)
}

// usable returns true if at least one endpoint is ready to handle gRPC calls for the DRA driver.
// Must be called while holding the mutex.
func (pm *DRAPluginManager) usable(driverName string) bool {
	for _, mp := range pm.store[driverName] {
		if mp.connected {
			return true
		}
	}
	return false
}

// ValidatePlugin implements [cache.PluginHandler].
//
// The plugin manager calls it upon detection of a new registration socket
// opened by DRA plugin.
func (pm *DRAPluginManager) ValidatePlugin(driverName string, endpoint string, supportedServices []string) error {
	_, err := pm.validateSupportedServices(driverName, supportedServices)
	if err != nil {
		return fmt.Errorf("invalid supported gRPC versions of DRA driver plugin %s at endpoint %s: %w", driverName, endpoint, err)
	}

	return err
}

// validateSupportedServices identifies the highest supported gRPC service for
// NodePrepareResources and NodeUnprepareResources and returns its name
// (e.g. [drapbv1beta1.DRAPluginService]). An error is returned if the plugin
// is unusable.
func (pm *DRAPluginManager) validateSupportedServices(driverName string, supportedServices []string) (string, error) {
	if len(supportedServices) == 0 {
		return "", errors.New("empty list of supported gRPC services (aka supported versions)")
	}

	// Pick most recent version if available.
	chosenService := ""
	for _, service := range supportedServices {
		if slices.Contains(servicesSupportedByKubelet, service) {
			chosenService = service
			break
		}
	}

	// Fall back to alpha if necessary because
	// plugins at that time didn't advertise gRPC services.
	if chosenService == "" {
		return "", fmt.Errorf("none of services supported by the plugin (%q) are supported by the kubelet (%q)", supportedServices, servicesSupportedByKubelet)
	}

	return chosenService, nil
}
