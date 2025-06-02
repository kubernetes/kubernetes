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

// Store keeps track of how to reach plugins registered for DRA drivers.
// Each plugin has a gRPC endpoint. There may be more than one plugin per driver.
//
// To be informed about available plugins, the store implements the
// [cache.PluginHandler] interface and needs to be added to the
// plugin manager.
//
// The null Store is not usable, use NewStore.
type Store struct {
	// backgroundCtx is used for all future activities of the store.
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
	store map[string][]*Plugin

	// pendingWipes maps a driver name to a cancel function for
	// wiping of that plugin's ResourceSlices. Entries get added
	// in DeRegisterPlugin and check in RegisterPlugin. If
	// wiping is pending during RegisterPlugin, it gets canceled.
	//
	// Must use pointers to functions because the entries have to
	// be comparable.
	pendingWipes map[string]*context.CancelCauseFunc
}

var _ cache.PluginHandler = &Store{}

// NewStore creates a new store with support for wiping ResourceSlices
// when the plugin(s) for a DRA driver are not available too long.
//
// The context can be used to cancel all background activities.
// If desired, Stop can be called in addition or instead of canceling
// the context. It then also waits for background activities to stop.
func NewStore(ctx context.Context, kubeClient kubernetes.Interface, getNode func() (*v1.Node, error), wipingDelay time.Duration) *Store {
	ctx, cancel := context.WithCancelCause(ctx)
	s := &Store{
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
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()

		ctx := s.backgroundCtx
		logger := klog.LoggerWithName(klog.FromContext(ctx), "startup")
		ctx = klog.NewContext(ctx, logger)
		s.wipeResourceSlices(ctx, 0 /* no delay */, "" /* all drivers */)
	}()

	return s
}

// Stop cancels any remaining background activities and blocks until all goroutines have stopped.
func (s *Store) Stop() {
	s.cancel(errors.New("Stop was called"))
	s.wg.Wait()
}

// wipeResourceSlices deletes ResourceSlices of the node, optionally just for a specific driver.
// Wiping will delay for a while and can be canceled by canceling the context.
func (s *Store) wipeResourceSlices(ctx context.Context, delay time.Duration, driver string) {
	if s.kubeClient == nil {
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
		node, err := s.getNode()
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

		err = s.kubeClient.ResourceV1beta1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: fieldSelector.String()})
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

// GetDRAPlugin returns a wrapper around those gRPC methods of a DRA
// driver kubelet plugin which need to be called by kubelet. The wrapper
// handles gRPC connection management and logging. Connections are reused
// across different calls.
//
// It returns an informative error message including the driver name
// with an explanation why the driver is not usable.
func (s *Store) GetDRAPlugin(driverName string) (*Plugin, error) {
	if driverName == "" {
		return nil, errors.New("DRA driver name is empty")
	}
	plugin := s.get(driverName)
	if plugin == nil {
		return nil, fmt.Errorf("DRA driver %s is not registered", driverName)
	}
	return plugin, nil
}

// get lets you retrieve a DRA Plugin by name.
func (s *Store) get(driverName string) *Plugin {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	plugins := s.store[driverName]
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
func (s *Store) RegisterPlugin(driverName string, endpoint string, supportedServices []string, pluginClientTimeout *time.Duration) error {
	// Prepare a context with its own logger for the plugin.
	//
	// The lifecycle of the plugin's background activities is tied to our
	// root context, so canceling that will also cancel the plugin.
	//
	// The logger injects the driver name and endpoint as additional values
	// into all log output related to the plugin.
	ctx := s.backgroundCtx
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "driverName", driverName, "endpoint", endpoint)
	ctx = klog.NewContext(ctx, logger)

	chosenService, err := s.validateSupportedServices(driverName, supportedServices)
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

	plugin := &Plugin{
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
	if err := s.add(plugin); err != nil {
		cancel(err)
		// No wrapping, the error already contains details.
		return err
	}

	return nil
}

func (s *Store) add(p *Plugin) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.store == nil {
		s.store = make(map[string][]*Plugin)
	}
	for _, oldP := range s.store[p.driverName] {
		if oldP.endpoint == p.endpoint {
			// One plugin instance cannot hijack the endpoint of another instance.
			return fmt.Errorf("endpoint %s already registered for plugin %s", p.endpoint, p.driverName)
		}
	}

	logger := klog.FromContext(p.backgroundCtx)
	s.store[p.driverName] = append(s.store[p.driverName], p)
	logger.V(3).Info("Registered DRA plugin", "numInstances", len(s.store[p.driverName]))
	s.sync(p.driverName)
	return nil
}

// DeRegisterPlugin implements [cache.PluginHandler].
//
// The plugin manager calls it after it has detected that
// the plugin removed its registration socket,
// signaling that it is no longer available.
func (s *Store) DeRegisterPlugin(driverName, endpoint string) {
	// remove could be removed (no pun intended) but is kept for the sake of symmetry.
	s.remove(driverName, endpoint)
}

func (s *Store) remove(driverName, endpoint string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	plugins := s.store[driverName]
	i := slices.IndexFunc(plugins, func(p *Plugin) bool { return p.driverName == driverName && p.endpoint == endpoint })
	if i == -1 {
		return
	}
	p := plugins[i]
	last := len(plugins) == 1
	if last {
		delete(s.store, driverName)
	} else {
		s.store[driverName] = slices.Delete(plugins, i, i+1)
	}
	if p.cancel != nil {
		// This cancels background attempts to establish a connection to the plugin.
		// TODO: remove this in favor of non-blocking connection management.
		p.cancel(errors.New("plugin got removed"))
	}
	logger := klog.FromContext(p.backgroundCtx)
	logger.V(3).Info("Unregistered DRA plugin", "numInstances", len(s.store[driverName]))
	s.sync(driverName)
}

// sync must be called each time the information about a plugin changes.
// The mutex must be locked for writing.
func (s *Store) sync(driverName string) {
	ctx := s.backgroundCtx
	logger := klog.FromContext(ctx)

	// Is the DRA driver usable again?
	if s.usable(driverName) {
		// Yes: cancel any pending ResourceSlice wiping for the DRA driver.
		if cancel := s.pendingWipes[driverName]; cancel != nil {
			(*cancel)(errors.New("new plugin instance registered"))
			delete(s.pendingWipes, driverName)
		}
		return
	}

	// No: prepare for canceling the background wiping. This needs to run
	// in the context of the store.
	logger = klog.LoggerWithName(logger, "driver-cleanup")
	logger = klog.LoggerWithValues(logger, "driverName", driverName)
	ctx, cancel := context.WithCancelCause(s.backgroundCtx)
	ctx = klog.NewContext(ctx, logger)

	// Clean up the ResourceSlices for the deleted Plugin since it
	// may have died without doing so itself and might never come
	// back.
	//
	// May get canceled if the plugin comes back quickly enough.
	if cancel := s.pendingWipes[driverName]; cancel != nil {
		(*cancel)(errors.New("plugin deregistered a second time"))
	}
	s.pendingWipes[driverName] = &cancel

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer func() {
			s.mutex.Lock()
			defer s.mutex.Unlock()

			// Cancel our own context, but remove it from the map only if it
			// is the current entry. Perhaps it already got replaced.
			cancel(errors.New("wiping done"))
			if s.pendingWipes[driverName] == &cancel {
				delete(s.pendingWipes, driverName)
			}
		}()
		s.wipeResourceSlices(ctx, s.wipingDelay, driverName)
	}()
}

// usable returns true if at least one endpoint is ready to handle gRPC calls for the DRA driver.
// Must be called while holding the mutex.
func (s *Store) usable(driverName string) bool {
	return len(s.store[driverName]) > 0
}

// ValidatePlugin implements [cache.PluginHandler].
//
// The plugin manager calls it upon detection of a new registration socket
// opened by DRA plugin.
func (s *Store) ValidatePlugin(driverName string, endpoint string, supportedServices []string) error {
	_, err := s.validateSupportedServices(driverName, supportedServices)
	if err != nil {
		return fmt.Errorf("invalid supported gRPC versions of DRA driver plugin %s at endpoint %s: %w", driverName, endpoint, err)
	}

	return err
}

// validateSupportedServices identifies the highest supported gRPC service for
// NodePrepareResources and NodeUnprepareResources and returns its name
// (e.g. [drapbv1beta1.DRAPluginService]). An error is returned if the plugin
// is unusable.
func (s *Store) validateSupportedServices(driverName string, supportedServices []string) (string, error) {
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
