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

package csitranslation

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
)

func TestTranslationStability(t *testing.T) {
	testCases := []struct {
		name string
		pv   *v1.PersistentVolume
	}{

		{
			name: "GCE PD PV Source",
			pv: &v1.PersistentVolume{
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
		{
			name: "AWS EBS PV Source",
			pv: &v1.PersistentVolume{
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
	}
	for _, test := range testCases {
		t.Logf("Testing %v", test.name)
		csiSource, err := TranslateInTreePVToCSI(test.pv)
		if err != nil {
			t.Errorf("Error when translating to CSI: %v", err)
		}
		newPV, err := TranslateCSIPVToInTree(csiSource)
		if err != nil {
			t.Errorf("Error when translating CSI Source to in tree volume: %v", err)
		}
		if !reflect.DeepEqual(newPV, test.pv) {
			t.Errorf("Volumes after translation and back not equal:\n\nOriginal Volume: %#v\n\nRound-trip Volume: %#v", test.pv, newPV)
		}
	}
}

func TestPluginNameMappings(t *testing.T) {
	testCases := []struct {
		name             string
		inTreePluginName string
		csiPluginName    string
	}{
		{
			name:             "GCE PD plugin name",
			inTreePluginName: "kubernetes.io/gce-pd",
			csiPluginName:    "pd.csi.storage.gke.io",
		},
		{
			name:             "AWS EBS plugin name",
			inTreePluginName: "kubernetes.io/aws-ebs",
			csiPluginName:    "ebs.csi.aws.com",
		},
	}
	for _, test := range testCases {
		t.Logf("Testing %v", test.name)
		csiPluginName, err := GetCSINameFromInTreeName(test.inTreePluginName)
		if err != nil {
			t.Errorf("Error when mapping In-tree plugin name to CSI plugin name %s", err)
		}
		if !IsMigratedCSIDriverByName(csiPluginName) {
			t.Errorf("%s expected to supersede an In-tree plugin", csiPluginName)
		}
		inTreePluginName, err := GetInTreeNameFromCSIName(csiPluginName)
		if err != nil {
			t.Errorf("Error when mapping CSI plugin name to In-tree plugin name %s", err)
		}
		if !IsMigratableIntreePluginByName(inTreePluginName) {
			t.Errorf("%s expected to be migratable to a CSI name", inTreePluginName)
		}
		if inTreePluginName != test.inTreePluginName || csiPluginName != test.csiPluginName {
			t.Errorf("CSI plugin name and In-tree plugin name do not map to each other: [%s => %s], [%s => %s]", test.csiPluginName, inTreePluginName, test.inTreePluginName, csiPluginName)
		}
	}
}

// TODO: test for not modifying the original PV.
