/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package secret

import (
	"fmt"
	"sync"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/errors"
)

type SecretGenerator interface {
}

// SecretPlugin is an interface to secret plugins that can be used
// to manage secrets.
type SecretPlugin interface {
	// Init initializes the plugin.  This will be called exactly once
	// before any New* calls are made - implementations of plugins may
	// depend on this.
	Init()

	// Name returns the plugin's secret type.  Plugins should use namespaced names
	// such as "example.com/secret".  The "kubernetes.io" namespace is
	// reserved for plugins which are bundled with kubernetes.
	SecretType() api.SecretType

	// GenerateSecret generates a secret.
	// The returned secret will be persisted. Return nil if the secret should
	// not be updated.
	GenerateSecret(secret api.Secret) (*api.Secret, error)

	// RevokeSecret revokes a secret. The action depends on the type of secret,
	// e.g. revoking a server certificate.
	RevokeSecret(secret api.Secret) error
}

// SecretPluginMgr tracks registered plugins.
type SecretPluginMgr struct {
	mutex   sync.Mutex
	plugins map[api.SecretType]SecretPlugin
}

// InitPlugins initializes each plugin.  All plugins must have unique names.
// This must be called exactly once before any New* methods are called on any
// plugins.
func (pm *SecretPluginMgr) InitPlugins(plugins []SecretPlugin) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	if pm.plugins == nil {
		pm.plugins = map[api.SecretType]SecretPlugin{}
	}

	allErrs := []error{}
	for _, plugin := range plugins {
		name := plugin.SecretType()
		if !util.IsQualifiedName(string(name)) {
			allErrs = append(allErrs, fmt.Errorf("secret plugin has invalid type: %#v", plugin))
			continue
		}

		if _, found := pm.plugins[name]; found {
			allErrs = append(allErrs, fmt.Errorf("secret plugin %q was registered more than once", name))
			continue
		}
		plugin.Init()
		pm.plugins[name] = plugin
		glog.V(1).Infof("Loaded secret plugin %q", name)
	}
	return errors.NewAggregate(allErrs)
}

// FindPluginByType looks for a plugin with the given type.
// If no plugin is registered with the type, return error.
func (pm *SecretPluginMgr) FindPluginByType(secretType api.SecretType) (SecretPlugin, error) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	pl, ok := pm.plugins[secretType]
	if !ok {
		return nil, fmt.Errorf("no secret plugin matched")
	}
	return pl, nil
}
