/*
Copyright 2025 The Kubernetes Authors.

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
)

// CleanupHandler is the handler which is responsible for handling
// resource cleanups for DRA plugins. It is used to cleanup
// plugin resource slices when a plugin is deregistered or when the
// connection to the plugin is lost.
// It manages the cleanup process, including delaying the cleanup
// to allow the plugin to potentially recover and cancel pending
// cleanup.
type cleanupHandler struct {
	// backgroundCtx is used for all future activities of the handler.
	// This is necessary because it implements APIs which don't
	// provide a context.
	backgroundCtx context.Context

	cancel      func(err error)
	kubeClient  kubernetes.Interface
	getNode     func() (*v1.Node, error)
	wipingDelay time.Duration

	wg    sync.WaitGroup
	mutex sync.Mutex

	// pendingWipes maps a plugin name to a cancel function for
	// wiping of that plugin's ResourceSlices. Entries get added
	// in DeRegisterPlugin and check in RegisterPlugin. If
	// wiping is pending during RegisterPlugin, it gets canceled.
	//
	// Must use pointers to functions because the entries have to
	// be comparable.
	pendingWipes map[string]*context.CancelCauseFunc
}

// newCleanupHandler returns new cleanup handler.
//
// Must only be called once per process because it manages global state.
func newCleanupHandler(ctx context.Context, kubeClient kubernetes.Interface, getNode func() (*v1.Node, error), wipingDelay time.Duration) *cleanupHandler {
	ctx, cancel := context.WithCancelCause(ctx)
	handler := &cleanupHandler{
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
	handler.wg.Add(1)
	go func() {
		defer handler.wg.Done()

		logger := klog.LoggerWithName(klog.FromContext(handler.backgroundCtx), "startup")
		ctx := klog.NewContext(handler.backgroundCtx, logger)
		handler.wipeResourceSlices(ctx, 0 /* no delay */, "" /* all drivers */)
	}()

	return handler
}

// Stop cancels any remaining background activities and blocks until all goroutines have stopped.
func (c *cleanupHandler) Stop() {
	c.cancel(errors.New("Stop was called"))
	c.wg.Wait()
}

// wipeResourceSlices deletes ResourceSlices of the node, optionally just for a specific driver.
// Wiping will delay for a while and can be canceled by canceling the context.
func (c *cleanupHandler) wipeResourceSlices(ctx context.Context, delay time.Duration, driver string) {
	if c.kubeClient == nil {
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
		node, err := c.getNode()
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

		err = c.kubeClient.ResourceV1beta1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: fieldSelector.String()})
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

func (c *cleanupHandler) cancelPendingWipe(pluginName, errMsg string) {
	if c == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if cancel, ok := c.pendingWipes[pluginName]; ok {
		if cancel != nil {
			(*cancel)(errors.New(errMsg))
		}
		delete(c.pendingWipes, pluginName)
	}
}

// cleanupResourceSlices initiates the cleanup process for ResourceSlices
// associated with a given plugin. It prepares a new context and logger for
// the cleanup operation, ensuring that any previous cleanup for the same
// plugin is canceled if still pending. The cleanup is performed
// asynchronously in a background goroutine, which will remove the
// ResourceSlices after a delay unless the plugin is back online before the
// cleanup completes.
func (c *cleanupHandler) cleanupResourceSlices(pluginName string) {
	if c == nil {
		return
	}
	if draPlugins.pluginConnected(pluginName) {
		// If at least one plugin is connected, no cleanup is needed.
		return
	}

	// Prepare for canceling the background wiping. This needs to run
	// in the context of the registration handler, the one from
	// the plugin is canceled.
	logger := klog.FromContext(c.backgroundCtx)
	logger = klog.LoggerWithName(logger, "driver-cleanup")
	logger = klog.LoggerWithValues(logger, "pluginName", pluginName)

	logger.V(3).Info("Starting ResourceSlices cleanup")

	// Clean up the ResourceSlices for the Plugin since it
	// may have died without doing so itself and might never come
	// back.
	//
	// May get canceled if the plugin comes back quickly enough
	// (see RegisterPlugin).
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if cancel := c.pendingWipes[pluginName]; cancel != nil {
		logger.V(3).Info("Previously started cleanup detected, do nothing")
		return
	}

	ctx, cancel := context.WithCancelCause(c.backgroundCtx)
	ctx = klog.NewContext(ctx, logger)
	c.pendingWipes[pluginName] = &cancel

	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		defer c.cancelPendingWipe(pluginName, "cleanup done")
		c.wipeResourceSlices(ctx, c.wipingDelay, pluginName)
	}()
}

// isCleanupPending checks if a cleanup operation is pending for the
// specified plugin.
func (c *cleanupHandler) isCleanupPending(pluginName string) bool {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if c.pendingWipes == nil {
		return false
	}
	cancel, ok := c.pendingWipes[pluginName]
	return ok && cancel != nil
}
