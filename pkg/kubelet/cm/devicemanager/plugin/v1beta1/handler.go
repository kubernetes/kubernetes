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

func (s *server) GetPluginHandler(ctx context.Context) cache.PluginHandler {
	logger := klog.FromContext(ctx)
	if f, err := os.Create(s.socketDir + "DEPRECATION"); err != nil {
		logger.Error(err, "Failed to create deprecation file at socket dir", "path", s.socketDir)
	} else {
		f.Close()
		logger.V(4).Info("Created deprecation file", "path", f.Name())
	}
	return s
}

func (s *server) RegisterPlugin(ctx context.Context, pluginName string, endpoint string, versions []string, pluginClientTimeout *time.Duration) error {
	logger := klog.FromContext(ctx)
	logger.V(2).Info("Registering plugin at endpoint", "plugin", pluginName, "endpoint", endpoint)
	return s.connectClient(ctx, pluginName, endpoint)
}

func (s *server) DeRegisterPlugin(ctx context.Context, pluginName, endpoint string) {
	logger := klog.FromContext(ctx)
	logger.V(2).Info("Deregistering plugin", "plugin", pluginName, "endpoint", endpoint)
	// endpoint in DeRegisterPlugin is the socket path
	client := s.getClient(pluginName, endpoint)
	if client != nil {
		if err := s.disconnectClient(logger, pluginName, client); err != nil {
			logger.Error(err, "disconnecting client", "plugin", pluginName, "endpoint", endpoint)
		}
	}
}

func (s *server) ValidatePlugin(ctx context.Context, pluginName string, endpoint string, versions []string) error {
	logger := klog.FromContext(ctx)
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
		// Need to re-connect the client if connection fails
		s.deregisterClient(logger, name, socketPath)
		logger.Error(err, "Failed to connect to new client", "resource", name, "socketPath", socketPath)
		return err
	}

	logger.V(2).Info("Connected to new client", "resource", name, "socketPath", socketPath)
	go func() {
		s.runClient(ctx, name, c)
	}()

	return nil
}

func (s *server) disconnectClient(logger klog.Logger, name string, c Client) error {
	s.deregisterClient(logger, name, c.SocketPath())
	return c.Disconnect(logger)
}
func (s *server) registerClient(logger klog.Logger, name string, c Client) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.clients[name] = append(s.clients[name], c)
	logger.V(2).Info("Registered client", "name", name, "socketPath", c.SocketPath())
}

func (s *server) deregisterClient(logger klog.Logger, name string, socketPath string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	// When a client is deregistered, we will rebuild the clients array, removing the given client.
	// We intentionally avoid mutating in place.
	// We only remove the connection when both the client name and socket path matches.
	// This ensures if there is two connections with same client name, only that specific client is removed.
	var newClients []Client
	for _, c := range s.clients[name] {
		if c.SocketPath() == socketPath {
			logger.V(2).Info("Deregistered client", "name", name, "socketPath", socketPath)
			continue
		}
		newClients = append(newClients, c)
	}

	if len(newClients) == 0 {
		delete(s.clients, name)
	} else {
		s.clients[name] = newClients
	}
}

func (s *server) runClient(ctx context.Context, name string, c Client) {
	logger := klog.FromContext(ctx)
	c.Run(ctx)

	c = s.getClient(name, c.SocketPath())
	if c == nil {
		return
	}

	if err := s.disconnectClient(logger, name, c); err != nil {
		logger.Error(err, "Unable to disconnect client", "resource", name, "client", c)
	}
}

func (s *server) getClient(name string, socketPath string) Client {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	for _, c := range s.clients[name] {
		if c.SocketPath() == socketPath {
			return c
		}
	}
	return nil
}
