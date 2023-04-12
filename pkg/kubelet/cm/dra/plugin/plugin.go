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
	"errors"
	"fmt"
	"strings"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2"
)

const (
	// DRAPluginName is the name of the in-tree DRA Plugin.
	DRAPluginName = "kubernetes.io/dra"
)

// draPlugins map keeps track of all registered DRA plugins on the node
// and their corresponding sockets.
var draPlugins = &PluginsStore{}

// RegistrationHandler is the handler which is fed to the pluginwatcher API.
type RegistrationHandler struct{}

// NewPluginHandler returns new registration handler.
func NewRegistrationHandler() *RegistrationHandler {
	return &RegistrationHandler{}
}

// RegisterPlugin is called when a plugin can be registered.
func (h *RegistrationHandler) RegisterPlugin(pluginName string, endpoint string, versions []string) error {
	klog.InfoS("Register new DRA plugin", "name", pluginName, "endpoint", endpoint)

	highestSupportedVersion, err := h.validateVersions("RegisterPlugin", pluginName, versions)
	if err != nil {
		return err
	}

	// Storing endpoint of newly registered DRA Plugin into the map, where plugin name will be the key
	// all other DRA components will be able to get the actual socket of DRA plugins by its name.
	draPlugins.Set(pluginName, &Plugin{
		endpoint:                endpoint,
		highestSupportedVersion: highestSupportedVersion,
	})

	return nil
}

// Return the highest supported version.
func highestSupportedVersion(versions []string) (*utilversion.Version, error) {
	if len(versions) == 0 {
		return nil, errors.New(log("DRA plugin reporting empty array for supported versions"))
	}

	var highestSupportedVersion *utilversion.Version

	var theErr error

	for i := len(versions) - 1; i >= 0; i-- {
		currentHighestVer, err := utilversion.ParseGeneric(versions[i])
		if err != nil {
			theErr = err

			continue
		}

		if currentHighestVer.Major() > 1 {
			// DRA currently only has version 1.x
			continue
		}

		if highestSupportedVersion == nil || highestSupportedVersion.LessThan(currentHighestVer) {
			highestSupportedVersion = currentHighestVer
		}
	}

	if highestSupportedVersion == nil {
		return nil, fmt.Errorf(
			"could not find a highest supported version from versions (%v) reported by this plugin: %+v",
			versions, theErr)
	}

	if highestSupportedVersion.Major() != 1 {
		return nil, fmt.Errorf("highest supported version reported by plugin is %v, must be v1.x", highestSupportedVersion)
	}

	return highestSupportedVersion, nil
}

func (h *RegistrationHandler) validateVersions(
	callerName string,
	pluginName string,
	versions []string,
) (*utilversion.Version, error) {
	if len(versions) == 0 {
		return nil, errors.New(
			log(
				"%s for DRA plugin %q failed. Plugin returned an empty list for supported versions",
				callerName,
				pluginName,
			),
		)
	}

	// Validate version
	newPluginHighestVersion, err := highestSupportedVersion(versions)
	if err != nil {
		return nil, errors.New(
			log(
				"%s for DRA plugin %q failed. None of the versions specified %q are supported. err=%v",
				callerName,
				pluginName,
				versions,
				err,
			),
		)
	}

	existingPlugin := draPlugins.Get(pluginName)
	if existingPlugin != nil {
		if !existingPlugin.highestSupportedVersion.LessThan(newPluginHighestVersion) {
			return nil, errors.New(
				log(
					"%s for DRA plugin %q failed. Another plugin with the same name is already registered with a higher supported version: %q",
					callerName,
					pluginName,
					existingPlugin.highestSupportedVersion,
				),
			)
		}
	}

	return newPluginHighestVersion, nil
}

func unregisterPlugin(pluginName string) {
	draPlugins.Delete(pluginName)
}

// DeRegisterPlugin is called when a plugin has removed its socket,
// signaling it is no longer available.
func (h *RegistrationHandler) DeRegisterPlugin(pluginName string) {
	klog.InfoS("DeRegister DRA plugin", "name", pluginName)
	unregisterPlugin(pluginName)
}

// ValidatePlugin is called by kubelet's plugin watcher upon detection
// of a new registration socket opened by DRA plugin.
func (h *RegistrationHandler) ValidatePlugin(pluginName string, endpoint string, versions []string) error {
	klog.InfoS("Validate DRA plugin", "name", pluginName, "endpoint", endpoint, "versions", strings.Join(versions, ","))

	_, err := h.validateVersions("ValidatePlugin", pluginName, versions)
	if err != nil {
		return fmt.Errorf("validation failed for DRA plugin %s at endpoint %s: %+v", pluginName, endpoint, err)
	}

	return err
}

// log prepends log string with `kubernetes.io/dra`.
func log(msg string, parts ...interface{}) string {
	return fmt.Sprintf(fmt.Sprintf("%s: %s", DRAPluginName, msg), parts...)
}
