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

package v1beta1

import (
	"context"
	"fmt"
	"os"
	"time"

	core "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	api "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
)

func (s *server) GetPluginHandler() cache.PluginHandler {
	// Use context.TODO() because we currently do not have a proper context to pass in.
	// Replace this with an appropriate context when refactoring this function to accept a context parameter.
	logger := klog.FromContext(context.TODO())
	if f, err := os.Create(s.socketDir + "DEPRECATION"); err != nil {
		logger.Error(err, "Failed to create deprecation file at socket dir", "path", s.socketDir)
	} else {
		f.Close()
		logger.V(4).Info("Created deprecation file", "path", f.Name())
	}
	return s
}

func (s *server) RegisterPlugin(pluginName string, endpoint string, versions []string, pluginClientTimeout *time.Duration) error {
	// Use context.TODO() because we currently do not have a proper context to pass in.
	// Replace this with an appropriate context when refactoring this function to accept a context parameter.
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	logger.V(2).Info("Registering plugin at endpoint", "plugin", pluginName, "endpoint", endpoint)
	return s.connectClient(ctx, pluginName, endpoint)
}

func (s *server) DeRegisterPlugin(pluginName, endpoint string) {
	logger := klog.FromContext(context.TODO())
	logger.V(2).Info("Deregistering plugin", "plugin", pluginName, "endpoint", endpoint)
	client := s.getClient(pluginName)
	if client != nil {
		if err := s.disconnectClient(logger, pluginName, client); err != nil {
			logger.Error(err, "disconnecting client", "plugin", pluginName, "endpoing", endpoint)
		}
	}
}

func (s *server) ValidatePlugin(pluginName string, endpoint string, versions []string) error {
	// Use context.TODO() because we currently do not have a proper context to pass in.
	// Replace this with an appropriate context when refactoring this function to accept a context parameter.
	logger := klog.FromContext(context.TODO())
	logger.V(2).Info("Got plugin at endpoint with versions", "plugin", pluginName, "endpoint", endpoint, "versions", versions)

	if !s.isVersionCompatibleWithPlugin(versions...) {
		return fmt.Errorf("manager version, %s, is not among plugin supported versions %v", api.Version, versions)
	}

	if !v1helper.IsExtendedResourceName(core.ResourceName(pluginName)) {
		return fmt.Errorf("invalid name of device plugin socket: %s", fmt.Sprintf(errInvalidResourceName, pluginName))
	}

	logger.V(2).Info("Device plugin validated", "plugin", pluginName, "endpoint", endpoint, "versions", versions)
	return nil
}

func (s *server) connectClient(ctx context.Context, name string, socketPath string) error {
	logger := klog.FromContext(ctx)
	c := NewPluginClient(name, socketPath, s.chandler)

	s.registerClient(logger, name, c)
	if err := c.Connect(ctx); err != nil {
		s.deregisterClient(logger, name)
		logger.Error(err, "Failed to connect to new client", "resource", name)
		return err
	}

	logger.V(2).Info("Connected to new client", "resource", name)
	go func() {
		s.runClient(ctx, name, c)
	}()

	return nil
}

func (s *server) disconnectClient(logger klog.Logger, name string, c Client) error {
	s.deregisterClient(logger, name)
	return c.Disconnect(logger)
}
func (s *server) registerClient(logger klog.Logger, name string, c Client) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.clients[name] = c
	logger.V(2).Info("Registered client", "name", name)
}

func (s *server) deregisterClient(logger klog.Logger, name string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	delete(s.clients, name)
	logger.V(2).Info("Deregistered client", "name", name)
}

func (s *server) runClient(ctx context.Context, name string, c Client) {
	logger := klog.FromContext(ctx)
	c.Run(ctx)

	c = s.getClient(name)
	if c == nil {
		return
	}

	if err := s.disconnectClient(logger, name, c); err != nil {
		logger.Error(err, "Unable to disconnect client", "resource", name, "client", c)
	}
}

func (s *server) getClient(name string) Client {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.clients[name]
}
