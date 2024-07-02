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
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
)

// RegistrationHandler is the handler which is fed to the pluginwatcher API.
type RegistrationHandler struct {
	ctx        context.Context
	kubeClient kubernetes.Interface
	getNode    func() (*v1.Node, error)
}

var _ cache.PluginHandler = &RegistrationHandler{}

// NewPluginHandler returns new registration handler.
//
// Must only be called once per process because it manages global state.
// If a kubeClient is provided, then it synchronizes ResourceSlices
// with the resource information provided by plugins.
func NewRegistrationHandler(kubeClient kubernetes.Interface, getNode func() (*v1.Node, error)) *RegistrationHandler {
	handler := &RegistrationHandler{
		// The context and thus logger should come from the caller.
		ctx:        klog.NewContext(context.TODO(), klog.LoggerWithName(klog.TODO(), "DRA registration handler")),
		kubeClient: kubeClient,
		getNode:    getNode,
	}

	// When kubelet starts up, no DRA driver has registered yet. None of
	// the drivers are usable until they come back, which might not happen
	// at all. Therefore it is better to not advertise any local resources
	// because pods could get stuck on the node waiting for the driver
	// to start up.
	//
	// This has to run in the background.
	go handler.wipeResourceSlices("")

	return handler
}

// wipeResourceSlices deletes ResourceSlices of the node, optionally just for a specific driver.
func (h *RegistrationHandler) wipeResourceSlices(pluginName string) {
	if h.kubeClient == nil {
		return
	}
	ctx := h.ctx
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
		node, err := h.getNode()
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			logger.Error(err, "Unexpected error checking for node")
		}
		fieldSelector := fields.Set{"nodeName": node.Name}
		if pluginName != "" {
			fieldSelector["driverName"] = pluginName
		}

		err = h.kubeClient.ResourceV1alpha3().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: fieldSelector.String()})
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

// RegisterPlugin is called when a plugin can be registered.
func (h *RegistrationHandler) RegisterPlugin(pluginName string, endpoint string, versions []string, pluginClientTimeout *time.Duration) error {
	ctx := h.ctx
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "pluginName", pluginName)
	ctx = klog.NewContext(ctx, logger)
	logger.V(3).Info("Register new DRA plugin", "endpoint", endpoint)

	highestSupportedVersion, err := h.validateVersions("registering", pluginName, versions)
	if err != nil {
		return err
	}

	var timeout time.Duration
	if pluginClientTimeout == nil {
		timeout = PluginClientTimeout
	} else {
		timeout = *pluginClientTimeout
	}

	ctx, cancel := context.WithCancelCause(ctx)

	pluginInstance := &Plugin{
		ctx:                     ctx,
		cancel:                  cancel,
		conn:                    nil,
		endpoint:                endpoint,
		highestSupportedVersion: highestSupportedVersion,
		clientTimeout:           timeout,
	}

	// Storing endpoint of newly registered DRA Plugin into the map, where plugin name will be the key
	// all other DRA components will be able to get the actual socket of DRA plugins by its name.
	if draPlugins.add(pluginName, pluginInstance) {
		logger.V(1).Info("Already registered, previous plugin was replaced")
	}

	return nil
}

func (h *RegistrationHandler) validateVersions(
	what string,
	pluginName string,
	versions []string,
) (*utilversion.Version, error) {
	if len(versions) == 0 {
		return nil, fmt.Errorf("%s DRA plugin %q: empty list for supported versions", what, pluginName)
	}

	// Validate version
	newPluginHighestVersion, err := utilversion.HighestSupportedVersion(versions)
	if err != nil {
		return nil, fmt.Errorf("%s DRA plugin %q: none of the versions specified %q are supported: %v", what, pluginName, versions, err)
	}

	existingPlugin := draPlugins.get(pluginName)
	if existingPlugin == nil {
		return newPluginHighestVersion, nil
	}
	if existingPlugin.highestSupportedVersion.LessThan(newPluginHighestVersion) {
		return newPluginHighestVersion, nil
	}
	return nil, fmt.Errorf("%s DRA plugin %q: another plugin with the same name is already registered with a higher supported version: %q < %q", what, pluginName, newPluginHighestVersion, existingPlugin.highestSupportedVersion)
}

// DeRegisterPlugin is called when a plugin has removed its socket,
// signaling it is no longer available.
func (h *RegistrationHandler) DeRegisterPlugin(pluginName string) {
	if p := draPlugins.delete(pluginName); p != nil {
		logger := klog.FromContext(p.ctx)
		logger.V(3).Info("Deregister DRA plugin", "endpoint", p.endpoint)

		// Did exist before. Let's clean up ResourceSlices for it because
		// it might have died without doing that itself and might not
		// come back.
		go h.wipeResourceSlices(pluginName)

		return
	}

	logger := klog.FromContext(h.ctx)
	logger.V(3).Info("Deregister DRA plugin not necessary, was already removed")
}

// ValidatePlugin is called by kubelet's plugin watcher upon detection
// of a new registration socket opened by DRA plugin.
func (h *RegistrationHandler) ValidatePlugin(pluginName string, endpoint string, versions []string) error {
	klog.InfoS("Validate DRA plugin", "name", pluginName, "endpoint", endpoint, "versions", strings.Join(versions, ","))

	_, err := h.validateVersions("validating", pluginName, versions)
	if err != nil {
		return err
	}

	return err
}
