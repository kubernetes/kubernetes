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

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	drapbv1alpha4 "k8s.io/kubelet/pkg/apis/dra/v1alpha4"
	drapbv1beta1 "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
)

// DRAPluginManager keeps track of how to reach plugins registered for DRA drivers.
// Each plugin has a gRPC endpoint. There may be more than one plugin per driver.
//
// To be informed about available plugins, the DRAPluginManager implements the
// [cache.PluginHandler] interface and needs to be added to the
// plugin manager.
//
// The null DRAPluginManager is not usable, use NewPluginManager.
type DRAPluginManager struct {
	// backgroundCtx is used for all future activities of the DRAPluginManager.
	// This is necessary because it implements APIs which don't
	// provide a context.
	backgroundCtx context.Context
	cancel        func(err error)
	kubeClient    kubernetes.Interface
	getNode       func() (*v1.Node, error)
	wipingDelay   time.Duration

	wg    sync.WaitGroup
	mutex sync.RWMutex

	// driver name -> Plugin in the order in which they got added
	store map[string][]*DRAPlugin

	// pendingWipes maps a driver name to a cancel function for
	// wiping of that plugin's ResourceSlices. Entries get added
	// in DeRegisterPlugin and check in RegisterPlugin. If
	// wiping is pending during RegisterPlugin, it gets canceled.
	//
	// Must use pointers to functions because the entries have to
	// be comparable.
	pendingWipes map[string]*context.CancelCauseFunc
}

var _ cache.PluginHandler = &DRAPluginManager{}

// NewDRAPluginManager creates a new DRAPluginManager, with support for wiping ResourceSlices
// when the plugin(s) for a DRA driver are not available too long.
//
// The context can be used to cancel all background activities.
// If desired, Stop can be called in addition or instead of canceling
// the context. It then also waits for background activities to stop.
func NewDRAPluginManager(ctx context.Context, kubeClient kubernetes.Interface, getNode func() (*v1.Node, error), wipingDelay time.Duration) *DRAPluginManager {
	ctx, cancel := context.WithCancelCause(ctx)
	pm := &DRAPluginManager{
		backgroundCtx: klog.NewContext(ctx, klog.LoggerWithName(klog.FromContext(ctx), "DRA registration handler")),
		cancel:        cancel,
		kubeClient:    kubeClient,
		getNode:       getNode,
		wipingDelay:   wipingDelay,
		pendingWipes:  make(map[string]*context.CancelCauseFunc),
	}

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
		pm.wipeResourceSlices(ctx, 0 /* no delay */, "" /* all drivers */)
	}()

	return pm
}

// Stop cancels any remaining background activities and blocks until all goroutines have stopped.
func (pm *DRAPluginManager) Stop() {
	pm.cancel(errors.New("Stop was called"))
	pm.wg.Wait()
}

// wipeResourceSlices deletes ResourceSlices of the node, optionally just for a specific driver.
// Wiping will delay for a while and can be canceled by canceling the context.
func (pm *DRAPluginManager) wipeResourceSlices(ctx context.Context, delay time.Duration, driver string) {
	if pm.kubeClient == nil {
		return
	}
	logger := klog.FromContext(ctx)

	if delay != 0 {
		// Before we start deleting, give the driver time to bounce back.
		// Perhaps it got removed as part of a DaemonSet update and the
		// replacement pod is about to start.
		logger.V(4).Info("Starting to wait before wiping ResourceSlices", "delay", delay)
		select {
		case <-ctx.Done():
			logger.V(4).Info("Aborting wiping of ResourceSlices", "reason", context.Cause(ctx))
		case <-time.After(delay):
			logger.V(4).Info("Starting to wipe ResourceSlices after waiting", "delay", delay)
		}
	}

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

		err = pm.kubeClient.ResourceV1beta1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: fieldSelector.String()})
		switch {
		case err == nil:
			logger.V(3).Info("Deleted ResourceSlices", "fieldSelector", fieldSelector)
			return true, nil
		case apierrors.IsUnauthorized(err):
			// This can happen while kubelet is still figuring out
			// its credentials.
			logger.V(5).Info("Deleting ResourceSlice failed, retrying", "fieldSelector", fieldSelector, "err", err)
			return false, nil
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

// get lets you retrieve a DRA Plugin by name.
func (pm *DRAPluginManager) get(driverName string) *DRAPlugin {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	plugins := pm.store[driverName]
	if len(plugins) == 0 {
		return nil
	}
	// Heuristic: pick the most recent one. It's most likely
	// the newest, except when kubelet got restarted and registered
	// all running plugins in random order.
	return plugins[len(plugins)-1]
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
	// Prepare a context with its own logger for the plugin.
	//
	// The lifecycle of the plugin's background activities is tied to our
	// root context, so canceling that will also cancel the plugin.
	//
	// The logger injects the driver name and endpoint as additional values
	// into all log output related to the plugin.
	ctx := pm.backgroundCtx
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "driverName", driverName, "endpoint", endpoint)
	ctx = klog.NewContext(ctx, logger)

	chosenService, err := pm.validateSupportedServices(driverName, supportedServices)
	if err != nil {
		return fmt.Errorf("invalid supported gRPC versions of DRA driver plugin %s at endpoint %s: %w", driverName, endpoint, err)
	}

	var timeout time.Duration
	if pluginClientTimeout == nil {
		timeout = defaultClientCallTimeout
	} else {
		timeout = *pluginClientTimeout
	}

	ctx, cancel := context.WithCancelCause(ctx)

	plugin := &DRAPlugin{
		driverName:        driverName,
		backgroundCtx:     ctx,
		cancel:            cancel,
		conn:              nil,
		endpoint:          endpoint,
		chosenService:     chosenService,
		clientCallTimeout: timeout,
	}

	// Storing endpoint of newly registered DRA Plugin into the map, where the DRA driver name will be the key
	// under which the manager will be able to get a plugin when it needs to call it.
	if err := pm.add(plugin); err != nil {
		cancel(err)
		// No wrapping, the error already contains details.
		return err
	}

	return nil
}

func (pm *DRAPluginManager) add(p *DRAPlugin) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	if pm.store == nil {
		pm.store = make(map[string][]*DRAPlugin)
	}
	for _, oldP := range pm.store[p.driverName] {
		if oldP.endpoint == p.endpoint {
			// One plugin instance cannot hijack the endpoint of another instance.
			return fmt.Errorf("endpoint %s already registered for plugin %s", p.endpoint, p.driverName)
		}
	}

	logger := klog.FromContext(p.backgroundCtx)
	pm.store[p.driverName] = append(pm.store[p.driverName], p)
	logger.V(3).Info("Registered DRA plugin", "numInstances", len(pm.store[p.driverName]))
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
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	plugins := pm.store[driverName]
	i := slices.IndexFunc(plugins, func(p *DRAPlugin) bool { return p.driverName == driverName && p.endpoint == endpoint })
	if i == -1 {
		return
	}
	p := plugins[i]
	last := len(plugins) == 1
	if last {
		delete(pm.store, driverName)
	} else {
		pm.store[driverName] = slices.Delete(plugins, i, i+1)
	}
	if p.cancel != nil {
		// This cancels background attempts to establish a connection to the plugin.
		// TODO: remove this in favor of non-blocking connection management.
		p.cancel(errors.New("plugin got removed"))
	}
	logger := klog.FromContext(p.backgroundCtx)
	logger.V(3).Info("Unregistered DRA plugin", "numInstances", len(pm.store[driverName]))
	pm.sync(driverName)
}

// sync must be called each time the information about a plugin changes.
// The mutex must be locked for writing.
func (pm *DRAPluginManager) sync(driverName string) {
	ctx := pm.backgroundCtx
	logger := klog.FromContext(ctx)

	// Is the DRA driver usable again?
	if pm.usable(driverName) {
		// Yes: cancel any pending ResourceSlice wiping for the DRA driver.
		if cancel := pm.pendingWipes[driverName]; cancel != nil {
			(*cancel)(errors.New("new plugin instance registered"))
			delete(pm.pendingWipes, driverName)
		}
		return
	}

	// No: prepare for canceling the background wiping. This needs to run
	// in the context of the DRAPluginManager.
	logger = klog.LoggerWithName(logger, "driver-cleanup")
	logger = klog.LoggerWithValues(logger, "driverName", driverName)
	ctx, cancel := context.WithCancelCause(pm.backgroundCtx)
	ctx = klog.NewContext(ctx, logger)

	// Clean up the ResourceSlices for the deleted Plugin since it
	// may have died without doing so itself and might never come
	// back.
	//
	// May get canceled if the plugin comes back quickly enough.
	if cancel := pm.pendingWipes[driverName]; cancel != nil {
		(*cancel)(errors.New("plugin deregistered a second time"))
	}
	pm.pendingWipes[driverName] = &cancel

	pm.wg.Add(1)
	go func() {
		defer pm.wg.Done()
		defer func() {
			pm.mutex.Lock()
			defer pm.mutex.Unlock()

			// Cancel our own context, but remove it from the map only if it
			// is the current entry. Perhaps it already got replaced.
			cancel(errors.New("wiping done"))
			if pm.pendingWipes[driverName] == &cancel {
				delete(pm.pendingWipes, driverName)
			}
		}()
		pm.wipeResourceSlices(ctx, pm.wipingDelay, driverName)
	}()
}

// usable returns true if at least one endpoint is ready to handle gRPC calls for the DRA driver.
// Must be called while holding the mutex.
func (pm *DRAPluginManager) usable(driverName string) bool {
	return len(pm.store[driverName]) > 0
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
	for _, service := range []string{
		// Sorted by most recent first, oldest last.
		drapbv1beta1.DRAPluginService,
		drapbv1alpha4.NodeService,
	} {
		if slices.Contains(supportedServices, service) {
			chosenService = service
			break
		}
	}

	// Fall back to alpha if necessary because
	// plugins at that time didn't advertise gRPC services.
	if chosenService == "" {
		chosenService = drapbv1alpha4.NodeService
	}

	return chosenService, nil
}
