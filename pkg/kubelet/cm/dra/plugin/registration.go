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

package plugin

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	drapbv1alpha4 "k8s.io/kubelet/pkg/apis/dra/v1alpha4"
	drapbv1beta1 "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
)

// defaultClientCallTimeout is the default amount of time that a DRA driver has
// to respond to any of the gRPC calls. kubelet uses this value by passing nil
// to RegisterPlugin. Some tests use a different, usually shorter timeout to
// speed up testing.
//
// This is half of the kubelet retry period (according to
// https://github.com/kubernetes/kubernetes/commit/0449cef8fd5217d394c5cd331d852bd50983e6b3).
const defaultClientCallTimeout = 45 * time.Second

// RegistrationHandler is the handler which is fed to the pluginwatcher API.
type RegistrationHandler struct {
	// backgroundCtx is used for all future activities of the handler.
	// This is necessary because it implements APIs which don't
	// provide a context.
	backgroundCtx  context.Context
	cleanupHandler *cleanupHandler
}

var _ cache.PluginHandler = &RegistrationHandler{}

// NewPluginHandler returns new registration handler.
//
// Must only be called once per process because it manages global state.
// If a kubeClient is provided, then it synchronizes ResourceSlices
// with the resource information provided by plugins.
func NewRegistrationHandler(kubeClient kubernetes.Interface, getNode func() (*v1.Node, error), wipingDelay time.Duration) *RegistrationHandler {
	// The context and thus logger should come from the caller.
	return newRegistrationHandler(context.TODO(), kubeClient, getNode, wipingDelay)
}

func newRegistrationHandler(ctx context.Context, kubeClient kubernetes.Interface, getNode func() (*v1.Node, error), wipingDelay time.Duration) *RegistrationHandler {
	return &RegistrationHandler{
		backgroundCtx:  klog.NewContext(ctx, klog.LoggerWithName(klog.FromContext(ctx), "DRA registration handler")),
		cleanupHandler: newCleanupHandler(ctx, kubeClient, getNode, wipingDelay),
	}
}

// Stop stops the cleanup handler, which will cancel all pending cleanups.
func (h *RegistrationHandler) Stop() {
	h.cleanupHandler.Stop()
}

// RegisterPlugin is called when a plugin can be registered.
//
// DRA uses the version array in the registration API to enumerate all gRPC
// services that the plugin provides, using the "<gRPC package name>.<service
// name>" format (e.g. "v1beta1.DRAPlugin"). This allows kubelet to determine
// in advance which version to use resp. which optional services the plugin
// supports.
func (h *RegistrationHandler) RegisterPlugin(pluginName string, endpoint string, supportedServices []string, pluginClientTimeout *time.Duration) error {
	// Prepare a context with its own logger for the plugin.
	//
	// The lifecycle of the plugin's background activities is tied to our
	// root context, so canceling that will also cancel the plugin.
	//
	// The logger injects the plugin name as additional value
	// into all log output related to the plugin.
	ctx := h.backgroundCtx
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "pluginName", pluginName, "endpoint", endpoint)
	ctx = klog.NewContext(ctx, logger)

	logger.V(3).Info("Register new DRA plugin")

	chosenService, err := h.validateSupportedServices(pluginName, supportedServices)
	if err != nil {
		return fmt.Errorf("version check of plugin %s failed: %w", pluginName, err)
	}

	var timeout time.Duration
	if pluginClientTimeout == nil {
		timeout = defaultClientCallTimeout
	} else {
		timeout = *pluginClientTimeout
	}

	ctx, cancel := context.WithCancelCause(ctx)

	pluginInstance := &Plugin{
		name:              pluginName,
		backgroundCtx:     ctx,
		conn:              nil,
		endpoint:          endpoint,
		chosenService:     chosenService,
		clientCallTimeout: timeout,
		cleanupHandler:    h.cleanupHandler,
	}

	// Storing endpoint of newly registered DRA Plugin into the map, where plugin name will be the key
	// all other DRA components will be able to get the actual socket of DRA plugins by its name.
	if err := draPlugins.add(pluginInstance); err != nil {
		cancel(err)
		// No wrapping, the error already contains details.
		return err
	}

	// Create a new gRPC connection to the plugin as soon as it's registered
	// to provide an effective connection monitoring.
	// If this is not done, the connection will be created lazily when
	// the first gRPC call is made, which can lead to inconsistent behavior
	// if the connection drops between registration and the first gRPC call.
	// It doesn't wait for the connection to be established, so it doesn't
	// block the registration process.
	if _, err = pluginInstance.getOrCreateGRPCConn(); err != nil {
		cancel(err)
		return err
	}

	if pluginInstance.isConnected() {
		// Now cancel any pending ResourceSlice wiping for this plugin.
		// Only needs to be done once.
		h.cleanupHandler.cancelPendingWipe(pluginName, "new plugin instance registered")
	} else {
		// If the plugin is not connected, it means that the gRPC connection
		// is not established yet, so we need to set up the cleanup for
		// ResourceSlices. This will ensure that if the plugin never comes
		// online, the ResourceSlices will be cleaned up after a delay.
		logger.V(3).Info("Plugin is not connected, scheduling cleanup of ResourceSlices")
		h.cleanupHandler.cleanupResourceSlices(pluginName)
	}

	return nil
}

// validateSupportedServices identifies the highest supported gRPC service for
// NodePrepareResources and NodeUnprepareResources and returns its name
// (e.g. [drapbv1beta1.DRAPluginService]). An error is returned if the plugin
// is unusable.
func (h *RegistrationHandler) validateSupportedServices(pluginName string, supportedServices []string) (string, error) {
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

// DeRegisterPlugin is called when a plugin has removed its socket,
// signaling it is no longer available.
func (h *RegistrationHandler) DeRegisterPlugin(pluginName, endpoint string) {
	if p, last := draPlugins.remove(pluginName, endpoint); p != nil {
		// This logger includes endpoint and pluginName.
		logger := klog.FromContext(p.backgroundCtx)
		logger.V(3).Info("Deregister DRA plugin", "lastInstance", last)
		if !last {
			return
		}
		h.cleanupHandler.cleanupResourceSlices(pluginName)
	}

	logger := klog.FromContext(h.backgroundCtx)
	logger.V(3).Info("Deregister DRA plugin not necessary, was already removed")
}

// ValidatePlugin is called by kubelet's plugin watcher upon detection
// of a new registration socket opened by DRA plugin.
func (h *RegistrationHandler) ValidatePlugin(pluginName string, endpoint string, supportedServices []string) error {
	_, err := h.validateSupportedServices(pluginName, supportedServices)
	if err != nil {
		return fmt.Errorf("invalid versions of plugin %s: %w", pluginName, err)
	}

	return err
}
