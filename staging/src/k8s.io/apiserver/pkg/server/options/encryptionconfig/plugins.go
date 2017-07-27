/*
Copyright 2017 The Kubernetes Authors.

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

package encryptionconfig

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"sync"

	"github.com/golang/glog"

	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
)

// Factory is a function that returns an envelope Service for encryption providers.
// The config parameter provides an io.Reader handler to the factory in
// order to load specific configurations. If no configuration is provided
// the parameter is nil.
type Factory func(config io.Reader) (envelope.Service, error)

type cloudKMSFactory func(name string) (envelope.Service, error)

// KMSPlugins contains all registered KMS options.
type KMSPlugins struct {
	lock     sync.RWMutex
	registry map[string]Factory
	cloudKMS cloudKMSFactory
}

var (
	// PluginEnabledFn checks whether a plugin is enabled.  By default, if you ask about it, it's enabled.
	PluginEnabledFn = func(name string, config io.Reader) bool {
		return true
	}

	// KMSPluginRegistry contains the registered KMS plugins which can be used for configuring
	// encryption providers.
	KMSPluginRegistry = KMSPlugins{}
)

// PluginEnabledFunc is a function type that can provide an external check on whether an admission plugin may be enabled
type PluginEnabledFunc func(name string, config io.Reader) bool

// Register registers a plugin Factory by name. This
// is expected to happen during app startup.
func (ps *KMSPlugins) Register(name string, plugin Factory) {
	ps.lock.Lock()
	defer ps.lock.Unlock()
	_, found := ps.registry[name]
	if ps.registry == nil {
		ps.registry = map[string]Factory{}
	}
	if found {
		glog.Fatalf("KMS plugin %q was registered twice", name)
	}
	glog.V(1).Infof("Registered KMS plugin %q", name)
	ps.registry[name] = plugin
}

// RegisterCloudProvidedKMSPlugin registers the cloud's KMS provider as
// an envelope.Service. This service is provided by the cloudprovider interface.
func (ps *KMSPlugins) RegisterCloudProvidedKMSPlugin(cloudKMSGetter cloudKMSFactory) {
	ps.cloudKMS = cloudKMSGetter
}

// getPlugin creates an instance of the named plugin.  It returns `false` if the
// the name is not known. The error is returned only when the named provider was
// known but failed to initialize.  The config parameter specifies the io.Reader
// handler of the configuration file for the cloud provider, or nil for no configuration.
func (ps *KMSPlugins) getPlugin(name string, config io.Reader) (envelope.Service, bool, error) {
	f, found := ps.fetchPluginFromRegistry(name)
	if !found {
		return nil, false, nil
	}

	config1, config2, err := splitStream(config)
	if err != nil {
		return nil, true, err
	}
	if !PluginEnabledFn(name, config1) {
		return nil, true, nil
	}

	ret, err := f(config2)
	return ret, true, err
}

// fetchPluginFromRegistry tries to get a registered plugin with the requested name.
func (ps *KMSPlugins) fetchPluginFromRegistry(name string) (Factory, bool) {
	ps.lock.RLock()
	defer ps.lock.RUnlock()
	// Map lookup defaults to single value context
	f, found := ps.registry[name]
	return f, found
}

// getCloudProvidedPlugin creates an instance of the named cloud provided KMS plugin.
func (ps *KMSPlugins) getCloudProvidedPlugin(name string) (envelope.Service, error) {
	if ps.cloudKMS == nil {
		return nil, fmt.Errorf("no cloud registered for KMS plugins")
	}
	return ps.cloudKMS(name)
}

// splitStream reads the stream bytes and constructs two copies of it.
func splitStream(config io.Reader) (io.Reader, io.Reader, error) {
	if config == nil || reflect.ValueOf(config).IsNil() {
		return nil, nil, nil
	}

	configBytes, err := ioutil.ReadAll(config)
	if err != nil {
		return nil, nil, err
	}

	return bytes.NewBuffer(configBytes), bytes.NewBuffer(configBytes), nil
}
