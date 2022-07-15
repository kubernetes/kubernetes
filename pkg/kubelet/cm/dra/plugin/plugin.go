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

package dra

import (
	"errors"
	"fmt"
	"strings"
	"time"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2"
)

const (
	// DRAPluginName is the name of the in-tree DRA Plugin
	DRAPluginName = "kubernetes.io/dra"

	resourceTimeout = 2 * time.Minute
)

// draPlugins map keep track of all registered DRA plugins on the node and their
// corresponding sockets
var draPlugins = &PluginsStore{}

// RegistrationHandler is the handler which is fed to the pluginwatcher API.
type RegistrationHandler struct {
}

var PluginHandler = &RegistrationHandler{}

// RegisterPlugin is called when a plugin can be registered
func (h *RegistrationHandler) RegisterPlugin(pluginName string, endpoint string, versions []string) error {
	klog.Infof(log("Register new plugin with name: %s at endpoint: %s", pluginName, endpoint))

	highestSupportedVersion, err := h.validateVersions("RegisterPlugin", pluginName, endpoint, versions)
	if err != nil {
		return err
	}

	// Storing endpoint of newly registered DRA Plugin into the map, where plugin name will be the key
	// all other DRA components will be able to get the actual socket of DRA plugins by its name.
	draPlugins.Set(pluginName, Plugin{
		endpoint:                endpoint,
		highestSupportedVersion: highestSupportedVersion,
	})

	return nil
}

// Return the highest supported version
func highestSupportedVersion(versions []string) (*utilversion.Version, error) {
	if len(versions) == 0 {
		return nil, errors.New(log("DRA driver reporting empty array for supported versions"))
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
			// CSI currently only has version 0.x and 1.x (see https://github.com/container-storage-interface/spec/releases).
			// Therefore any driver claiming version 2.x+ is ignored as an unsupported versions.
			// Future 1.x versions of CSI are supposed to be backwards compatible so this version of Kubernetes will work with any 1.x driver
			// (or 0.x), but it may not work with 2.x drivers (because 2.x does not have to be backwards compatible with 1.x).
			continue
		}
		if highestSupportedVersion == nil || highestSupportedVersion.LessThan(currentHighestVer) {
			highestSupportedVersion = currentHighestVer
		}
	}

	if highestSupportedVersion == nil {
		return nil, fmt.Errorf("could not find a highest supported version from versions (%v) reported by this driver: %v", versions, theErr)
	}

	if highestSupportedVersion.Major() != 1 {
		// CSI v0.x is no longer supported as of Kubernetes v1.17 in
		// accordance with deprecation policy set out in Kubernetes v1.13
		return nil, fmt.Errorf("highest supported version reported by driver is %v, must be v1.x", highestSupportedVersion)
	}
	return highestSupportedVersion, nil
}

func (h *RegistrationHandler) validateVersions(callerName, pluginName string, endpoint string, versions []string) (*utilversion.Version, error) {
	if len(versions) == 0 {
		return nil, errors.New(log("%s for DRA driver %q failed. Plugin returned an empty list for supported versions", callerName, pluginName))
	}

	// Validate version
	newDriverHighestVersion, err := highestSupportedVersion(versions)
	if err != nil {
		return nil, errors.New(log("%s for DRA driver %q failed. None of the versions specified %q are supported. err=%v", callerName, pluginName, versions, err))
	}

	existingDriver, driverExists := draPlugins.Get(pluginName)
	if driverExists {
		if !existingDriver.highestSupportedVersion.LessThan(newDriverHighestVersion) {
			return nil, errors.New(log("%s for DRA driver %q failed. Another driver with the same name is already registered with a higher supported version: %q", callerName, pluginName, existingDriver.highestSupportedVersion))
		}
	}

	return newDriverHighestVersion, nil
}

func unregisterPlugin(pluginName string) error {
	draPlugins.Delete(pluginName)
	return nil
}

// DeRegisterPlugin is called when a plugin removed its socket, signaling
// it is no longer available
func (h *RegistrationHandler) DeRegisterPlugin(pluginName string) {
	klog.Info(log("registrationHandler.DeRegisterPlugin request for plugin %s", pluginName))
	if err := unregisterPlugin(pluginName); err != nil {
		klog.Error(log("registrationHandler.DeRegisterPlugin failed: %v", err))
	}
}

// ValidatePlugin is called by kubelet's plugin watcher upon detection
// of a new registration socket opened by DRA plugin.
func (h *RegistrationHandler) ValidatePlugin(pluginName string, endpoint string, versions []string) error {
	klog.Infof(log("Trying to validate a new DRA plugin with name: %s endpoint: %s versions: %s",
		pluginName, endpoint, strings.Join(versions, ",")))

	_, err := h.validateVersions("ValidatePlugin", pluginName, endpoint, versions)
	if err != nil {
		return fmt.Errorf("validation failed for DRA plugin %s at endpoint %s: %v", pluginName, endpoint, err)
	}

	return err
}

// log prepends log string with `kubernetes.io/dra`
func log(msg string, parts ...interface{}) string {
	return fmt.Sprintf(fmt.Sprintf("%s: %s", DRAPluginName, msg), parts...)
}
