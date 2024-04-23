/*
Copyright 2016 The Kubernetes Authors.

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

package rest

import (
	"fmt"
	"net/http"
	"sync"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog/v2"
)

type AuthProvider interface {
	// WrapTransport allows the plugin to create a modified RoundTripper that
	// attaches authorization headers (or other info) to requests.
	WrapTransport(http.RoundTripper) http.RoundTripper
	// Login allows the plugin to initialize its configuration. It must not
	// require direct user interaction.
	Login() error
}

// Factory generates an AuthProvider plugin.
//
//	clusterAddress is the address of the current cluster.
//	config is the initial configuration for this plugin.
//	persister allows the plugin to save updated configuration.
type Factory func(clusterAddress string, config map[string]string, persister AuthProviderConfigPersister) (AuthProvider, error)

// AuthProviderConfigPersister allows a plugin to persist configuration info
// for just itself.
type AuthProviderConfigPersister interface {
	Persist(map[string]string) error
}

type noopPersister struct{}

func (n *noopPersister) Persist(_ map[string]string) error {
	// no operation persister
	return nil
}

// All registered auth provider plugins.
var pluginsLock sync.Mutex
var plugins = make(map[string]Factory)

func RegisterAuthProviderPlugin(name string, plugin Factory) error {
	pluginsLock.Lock()
	defer pluginsLock.Unlock()
	if _, found := plugins[name]; found {
		return fmt.Errorf("auth Provider Plugin %q was registered twice", name)
	}
	//nolint:logchck // For development and debugging only
	klog.Background().V(4).Info("Registered Auth Provider Plugin", "plugin-name", name)
	plugins[name] = plugin
	return nil
}

func GetAuthProvider(clusterAddress string, apc *clientcmdapi.AuthProviderConfig, persister AuthProviderConfigPersister) (AuthProvider, error) {
	pluginsLock.Lock()
	defer pluginsLock.Unlock()
	p, ok := plugins[apc.Name]
	if !ok {
		return nil, fmt.Errorf("no Auth Provider found for name %q", apc.Name)
	}
	if persister == nil {
		persister = &noopPersister{}
	}
	return p(clusterAddress, apc.Config, persister)
}
