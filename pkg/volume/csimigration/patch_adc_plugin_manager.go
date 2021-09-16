package csimigration

import (
	"k8s.io/component-base/featuregate"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/features"
)

// NewADCPluginManager returns a new PluginManager instance for the Attach Detach controller which uses different
// featuregates in openshift to control enablement/disablement which *DO NOT MATCH* the featuregates for the rest of the
// cluster.
func NewADCPluginManager(m PluginNameMapper, featureGate featuregate.FeatureGate) PluginManager {
	ret := NewPluginManager(m, featureGate)
	ret.useADCPluginManagerFeatureGates = true
	return ret
}

// adcIsMigrationEnabledForPlugin indicates whether CSI migration has been enabled
// for a particular storage plugin in Attach/Detach controller.
func (pm PluginManager) adcIsMigrationEnabledForPlugin(pluginName string) bool {
	// CSIMigration feature should be enabled along with the plugin-specific one
	if !pm.featureGate.Enabled(features.CSIMigration) {
		return false
	}

	switch pluginName {
	case csilibplugins.AWSEBSInTreePluginName:
		return pm.featureGate.Enabled(features.ADCCSIMigrationAWS)
	case csilibplugins.AzureDiskInTreePluginName:
		return pm.featureGate.Enabled(features.ADCCSIMigrationAzureDisk)
	case csilibplugins.AzureFileInTreePluginName:
		return pm.featureGate.Enabled(features.ADCCSIMigrationAzureFile)
	case csilibplugins.CinderInTreePluginName:
		return pm.featureGate.Enabled(features.ADCCSIMigrationCinder)
	case csilibplugins.GCEPDInTreePluginName:
		return pm.featureGate.Enabled(features.ADCCSIMigrationGCEPD)
	case csilibplugins.VSphereInTreePluginName:
		return pm.featureGate.Enabled(features.ADCCSIMigrationVSphere)
	default:
		return pm.isMigrationEnabledForPlugin(pluginName)
	}
}

// IsMigrationEnabledForPlugin indicates whether CSI migration has been enabled
// for a particular storage plugin
func (pm PluginManager) IsMigrationEnabledForPlugin(pluginName string) bool {
	if pm.useADCPluginManagerFeatureGates {
		return pm.adcIsMigrationEnabledForPlugin(pluginName)
	}

	return pm.isMigrationEnabledForPlugin(pluginName)
}
