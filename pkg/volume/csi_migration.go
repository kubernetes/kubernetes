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

package volume

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/features"
)

func isFeatureFlagEnabled(key featuregate.Feature) bool {
	return utilfeature.DefaultFeatureGate.Enabled(key)
}

// IsCSIMigrationEnabledForPluginByName returns a boolean value indicating whether
// the CSI migration has been enabled for a particular storage plugin cluster-wide
func IsCSIMigrationEnabledForPluginByName(pluginName string) bool {
	// In-tree storage to CSI driver migration feature should be enabled,
	// along with the plugin-specific one
	if !isFeatureFlagEnabled(features.CSIMigration) {
		return false
	}

	switch pluginName {
	case csilibplugins.AWSEBSInTreePluginName:
		return isFeatureFlagEnabled(features.CSIMigrationAWS)
	case csilibplugins.GCEPDInTreePluginName:
		return isFeatureFlagEnabled(features.CSIMigrationGCE)
	case csilibplugins.AzureFileInTreePluginName:
		return isFeatureFlagEnabled(features.CSIMigrationAzureFile)
	case csilibplugins.AzureDiskInTreePluginName:
		return isFeatureFlagEnabled(features.CSIMigrationAzureDisk)
	case csilibplugins.CinderInTreePluginName:
		return isFeatureFlagEnabled(features.CSIMigrationOpenStack)
	default:
		return false
	}
}
