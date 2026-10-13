/*
Copyright 2019 The Kubernetes Authors.

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

package nodevolumelimits

import (
	storagev1 "k8s.io/api/storage/v1"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
)

// isCSIMigrationOn returns a boolean value indicating whether
// the CSI migration has been enabled for a particular storage plugin.
func isCSIMigrationOn(csiNode *storagev1.CSINode, pluginName string) bool {
	if csiNode == nil || len(pluginName) == 0 {
		return false
	}

	// In-tree storage to CSI driver migration feature should be enabled,
	// along with the plugin-specific one
	switch pluginName {
	case csilibplugins.AWSEBSInTreePluginName:
		return true
	case csilibplugins.PortworxVolumePluginName:
		return true
	case csilibplugins.GCEPDInTreePluginName:
		return true
	case csilibplugins.AzureDiskInTreePluginName:
		return true
	case csilibplugins.CinderInTreePluginName:
		return true
	default:
		return false
	}
}
