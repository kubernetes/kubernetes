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

package csimigration

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
)

func TestIsMigratable(t *testing.T) {
	testCases := []struct {
		name                 string
		pluginFeature        featuregate.Feature
		pluginFeatureEnabled bool
		csiMigrationEnabled  bool
		isMigratable         bool
		spec                 *volume.Spec
	}{
		{
			name:                 "RBD PV source with CSIMigrationGCE enabled",
			pluginFeature:        features.CSIMigrationRBD,
			pluginFeatureEnabled: true,
			isMigratable:         true,
			csiMigrationEnabled:  true,
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							RBD: &v1.RBDPersistentVolumeSource{
								RBDImage: "test-disk",
							},
						},
					},
				},
			},
		},
		{
			name:                 "RBD PD PV Source with CSIMigrationGCE disabled",
			pluginFeature:        features.CSIMigrationRBD,
			pluginFeatureEnabled: false,
			isMigratable:         false,
			csiMigrationEnabled:  true,
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							RBD: &v1.RBDPersistentVolumeSource{
								RBDImage: "test-disk",
							},
						},
					},
				},
			},
		},
	}
	csiTranslator := csitrans.New()
	for _, test := range testCases {
		pm := NewPluginManager(csiTranslator, utilfeature.DefaultFeatureGate)
		t.Run(fmt.Sprintf("Testing %v", test.name), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, test.pluginFeature, test.pluginFeatureEnabled)
			migratable, err := pm.IsMigratable(test.spec)
			if migratable != test.isMigratable {
				t.Errorf("Expected migratability of spec: %v does not match obtained migratability: %v", test.isMigratable, migratable)
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestMigrationFeatureFlagStatus(t *testing.T) {
	testCases := []struct {
		name                          string
		pluginName                    string
		csiMigrationEnabled           bool
		pluginFeature                 featuregate.Feature
		pluginFeatureEnabled          bool
		inTreePluginUnregister        featuregate.Feature
		inTreePluginUnregisterEnabled bool
		csiMigrationResult            bool
		csiMigrationCompleteResult    bool
	}{
		{
			name:                          "gce-pd migration flag enabled and migration-complete flag disabled with CSI migration flag",
			pluginName:                    "kubernetes.io/gce-pd",
			pluginFeatureEnabled:          true,
			csiMigrationEnabled:           true,
			inTreePluginUnregister:        features.InTreePluginGCEUnregister,
			inTreePluginUnregisterEnabled: false,
			csiMigrationResult:            true,
			csiMigrationCompleteResult:    false,
		},
		{
			name:                          "gce-pd migration flag enabled and migration-complete flag enabled with CSI migration flag",
			pluginName:                    "kubernetes.io/gce-pd",
			pluginFeatureEnabled:          true,
			csiMigrationEnabled:           true,
			inTreePluginUnregister:        features.InTreePluginGCEUnregister,
			inTreePluginUnregisterEnabled: true,
			csiMigrationResult:            true,
			csiMigrationCompleteResult:    true,
		},
	}
	csiTranslator := csitrans.New()
	for _, test := range testCases {
		pm := NewPluginManager(csiTranslator, utilfeature.DefaultFeatureGate)
		t.Run(fmt.Sprintf("Testing %v", test.name), func(t *testing.T) {
			// CSIMigrationGCE is locked to on, so it cannot be enabled or disabled. There are a couple
			// of test cases that check correct behavior when CSIMigrationGCE is enabled, but there are
			// no longer any tests cases for CSIMigrationGCE being disabled as that is not possible.
			if len(test.pluginFeature) > 0 {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, test.pluginFeature, test.pluginFeatureEnabled)
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, test.inTreePluginUnregister, test.inTreePluginUnregisterEnabled)

			csiMigrationResult := pm.IsMigrationEnabledForPlugin(test.pluginName)
			if csiMigrationResult != test.csiMigrationResult {
				t.Errorf("Expected migratability of plugin %v: %v does not match obtained migratability: %v", test.pluginName, test.csiMigrationResult, csiMigrationResult)
			}
			csiMigrationCompleteResult := pm.IsMigrationCompleteForPlugin(test.pluginName)
			if csiMigrationCompleteResult != test.csiMigrationCompleteResult {
				t.Errorf("Expected migration complete status of plugin: %v: %v does not match obtained migratability: %v", test.pluginName, test.csiMigrationCompleteResult, csiMigrationResult)
			}
		})
	}
}
