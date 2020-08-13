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

	"k8s.io/api/core/v1"
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
			name:                 "GCE PD PV source with CSIMigrationGCE enabled",
			pluginFeature:        features.CSIMigrationGCE,
			pluginFeatureEnabled: true,
			isMigratable:         true,
			csiMigrationEnabled:  true,
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
								PDName:    "test-disk",
								FSType:    "ext4",
								Partition: 0,
								ReadOnly:  false,
							},
						},
					},
				},
			},
		},
		{
			name:                 "GCE PD PV Source with CSIMigrationGCE disabled",
			pluginFeature:        features.CSIMigrationGCE,
			pluginFeatureEnabled: false,
			isMigratable:         false,
			csiMigrationEnabled:  true,
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
								PDName:    "test-disk",
								FSType:    "ext4",
								Partition: 0,
								ReadOnly:  false,
							},
						},
					},
				},
			},
		},
		{
			name:                 "AWS EBS PV with CSIMigrationAWS enabled",
			pluginFeature:        features.CSIMigrationAWS,
			pluginFeatureEnabled: true,
			isMigratable:         true,
			csiMigrationEnabled:  true,
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
								VolumeID:  "vol01",
								FSType:    "ext3",
								Partition: 1,
								ReadOnly:  true,
							},
						},
					},
				},
			},
		},
		{
			name:                 "AWS EBS PV with CSIMigration and CSIMigrationAWS disabled",
			pluginFeature:        features.CSIMigrationAWS,
			pluginFeatureEnabled: false,
			isMigratable:         false,
			csiMigrationEnabled:  false,
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
								VolumeID:  "vol01",
								FSType:    "ext3",
								Partition: 1,
								ReadOnly:  true,
							},
						},
					},
				},
			},
		},
	}
	csiTranslator := csitrans.New()
	for _, test := range testCases {
		pm := NewPluginManager(csiTranslator)
		t.Run(fmt.Sprintf("Testing %v", test.name), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, test.csiMigrationEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, test.pluginFeature, test.pluginFeatureEnabled)()
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

func TestCheckMigrationFeatureFlags(t *testing.T) {
	testCases := []struct {
		name                         string
		pluginFeature                featuregate.Feature
		pluginFeatureEnabled         bool
		csiMigrationEnabled          bool
		pluginFeatureComplete        featuregate.Feature
		pluginFeatureCompleteEnabled bool
		result                       bool
	}{
		{
			name:                         "plugin specific feature flag enabled with migration flag disabled",
			pluginFeature:                features.CSIMigrationGCE,
			pluginFeatureEnabled:         true,
			csiMigrationEnabled:          false,
			pluginFeatureComplete:        features.CSIMigrationGCEComplete,
			pluginFeatureCompleteEnabled: false,
			result:                       false,
		},
		{
			name:                         "plugin specific complete flag enabled but plugin specific feature flag disabled",
			pluginFeature:                features.CSIMigrationAWS,
			pluginFeatureEnabled:         false,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationAWSComplete,
			pluginFeatureCompleteEnabled: true,
			result:                       false,
		},
		{
			name:                         "plugin specific complete feature disabled but plugin specific migration feature and CSI migration flag enabled",
			pluginFeature:                features.CSIMigrationGCE,
			pluginFeatureEnabled:         true,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationGCEComplete,
			pluginFeatureCompleteEnabled: false,
			result:                       true,
		},
		{
			name:                         "all features enabled",
			pluginFeature:                features.CSIMigrationAWS,
			pluginFeatureEnabled:         true,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationAWSComplete,
			pluginFeatureCompleteEnabled: true,
			result:                       true,
		},
	}
	for _, test := range testCases {
		t.Run(fmt.Sprintf("Testing %v", test.name), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, test.csiMigrationEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, test.pluginFeature, test.pluginFeatureEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, test.pluginFeatureComplete, test.pluginFeatureCompleteEnabled)()
			err := CheckMigrationFeatureFlags(utilfeature.DefaultFeatureGate, test.pluginFeature, test.pluginFeatureComplete)
			if err != nil && test.result == true {
				t.Errorf("Unexpected error: %v", err)
			}
			if err == nil && test.result == false {
				t.Errorf("Unexpected validation pass")
			}
		})
	}
}

func TestMigrationFeatureFlagStatus(t *testing.T) {
	testCases := []struct {
		name                         string
		pluginName                   string
		csiMigrationEnabled          bool
		pluginFeature                featuregate.Feature
		pluginFeatureEnabled         bool
		pluginFeatureComplete        featuregate.Feature
		pluginFeatureCompleteEnabled bool
		csiMigrationResult           bool
		csiMigrationCompleteResult   bool
	}{
		{
			name:                         "gce-pd migration flag disabled and migration-complete flag disabled with CSI migration flag disabled",
			pluginName:                   "kubernetes.io/gce-pd",
			pluginFeature:                features.CSIMigrationGCE,
			pluginFeatureEnabled:         false,
			csiMigrationEnabled:          false,
			pluginFeatureComplete:        features.CSIMigrationGCEComplete,
			pluginFeatureCompleteEnabled: false,
			csiMigrationResult:           false,
			csiMigrationCompleteResult:   false,
		},
		{
			name:                         "gce-pd migration flag disabled and migration-complete flag disabled with CSI migration flag enabled",
			pluginName:                   "kubernetes.io/gce-pd",
			pluginFeature:                features.CSIMigrationGCE,
			pluginFeatureEnabled:         false,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationGCEComplete,
			pluginFeatureCompleteEnabled: false,
			csiMigrationResult:           false,
			csiMigrationCompleteResult:   false,
		},
		{
			name:                         "gce-pd migration flag enabled and migration-complete flag disabled with CSI migration flag enabled",
			pluginName:                   "kubernetes.io/gce-pd",
			pluginFeature:                features.CSIMigrationGCE,
			pluginFeatureEnabled:         true,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationGCEComplete,
			pluginFeatureCompleteEnabled: false,
			csiMigrationResult:           true,
			csiMigrationCompleteResult:   false,
		},
		{
			name:                         "gce-pd migration flag enabled and migration-complete flag enabled with CSI migration flag enabled",
			pluginName:                   "kubernetes.io/gce-pd",
			pluginFeature:                features.CSIMigrationGCE,
			pluginFeatureEnabled:         true,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationGCEComplete,
			pluginFeatureCompleteEnabled: true,
			csiMigrationResult:           true,
			csiMigrationCompleteResult:   true,
		},
		{
			name:                         "aws-ebs migration flag disabled and migration-complete flag disabled with CSI migration flag disabled",
			pluginName:                   "kubernetes.io/aws-ebs",
			pluginFeature:                features.CSIMigrationAWS,
			pluginFeatureEnabled:         false,
			csiMigrationEnabled:          false,
			pluginFeatureComplete:        features.CSIMigrationAWSComplete,
			pluginFeatureCompleteEnabled: false,
			csiMigrationResult:           false,
			csiMigrationCompleteResult:   false,
		},
		{
			name:                         "aws-ebs migration flag disabled and migration-complete flag disabled with CSI migration flag enabled",
			pluginName:                   "kubernetes.io/aws-ebs",
			pluginFeature:                features.CSIMigrationAWS,
			pluginFeatureEnabled:         false,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationAWSComplete,
			pluginFeatureCompleteEnabled: false,
			csiMigrationResult:           false,
			csiMigrationCompleteResult:   false,
		},
		{
			name:                         "aws-ebs migration flag enabled and migration-complete flag disabled with CSI migration flag enabled",
			pluginName:                   "kubernetes.io/aws-ebs",
			pluginFeature:                features.CSIMigrationAWS,
			pluginFeatureEnabled:         true,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationAWSComplete,
			pluginFeatureCompleteEnabled: false,
			csiMigrationResult:           true,
			csiMigrationCompleteResult:   false,
		},
		{
			name:                         "aws-ebs migration flag enabled and migration-complete flag enabled with CSI migration flag enabled",
			pluginName:                   "kubernetes.io/aws-ebs",
			pluginFeature:                features.CSIMigrationAWS,
			pluginFeatureEnabled:         true,
			csiMigrationEnabled:          true,
			pluginFeatureComplete:        features.CSIMigrationAWSComplete,
			pluginFeatureCompleteEnabled: true,
			csiMigrationResult:           true,
			csiMigrationCompleteResult:   true,
		},
	}
	csiTranslator := csitrans.New()
	for _, test := range testCases {
		pm := NewPluginManager(csiTranslator)
		t.Run(fmt.Sprintf("Testing %v", test.name), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, test.csiMigrationEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, test.pluginFeature, test.pluginFeatureEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, test.pluginFeatureComplete, test.pluginFeatureCompleteEnabled)()

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
