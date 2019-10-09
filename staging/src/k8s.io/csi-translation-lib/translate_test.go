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

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
		t.Run(test.name, func(t *testing.T) {
			ctl := New()
			csiSource, err := ctl.TranslateInTreePVToCSI(test.pv)
			if err != nil {
				t.Errorf("Error when translating to CSI: %v", err)
			}
			newPV, err := ctl.TranslateCSIPVToInTree(csiSource)
			if err != nil {
				t.Errorf("Error when translating CSI Source to in tree volume: %v", err)
			}
			if !reflect.DeepEqual(newPV, test.pv) {
				t.Errorf("Volumes after translation and back not equal:\n\nOriginal Volume: %#v\n\nRound-trip Volume: %#v", test.pv, newPV)
			}
		})
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
		t.Run(test.name, func(t *testing.T) {
			ctl := New()
			csiPluginName, err := ctl.GetCSINameFromInTreeName(test.inTreePluginName)
			if err != nil {
				t.Errorf("Error when mapping In-tree plugin name to CSI plugin name %s", err)
			}
			if !ctl.IsMigratedCSIDriverByName(csiPluginName) {
				t.Errorf("got in-tree plugin, but %s expected to supersede ", csiPluginName)
			}
			inTreePluginName, err := ctl.GetInTreeNameFromCSIName(csiPluginName)
			if err != nil {
				t.Errorf("Error when mapping CSI plugin name to In-tree plugin name %s", err)
			}
			if !ctl.IsMigratableIntreePluginByName(inTreePluginName) {
				t.Errorf("%s expected to be migratable to a CSI name", inTreePluginName)
			}
			if inTreePluginName != test.inTreePluginName || csiPluginName != test.csiPluginName {
				t.Errorf("CSI plugin name and In-tree plugin name do not map to each other: [%s => %s], [%s => %s]", test.csiPluginName, inTreePluginName, test.inTreePluginName, csiPluginName)
			}
		})
	}
}

// TestNoInputModification tests that the inputs to each of the translation lib
// function is not modified. It's necessary because all the objects passed in
// could have nested pointer types that might get touched by translation
func TestNoInputModification(t *testing.T) {
	/*
		Remaining functions:

		TranslateInTreeInlineVolumeToCSI(volume *v1.Volume)
		TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)
		GetInTreePluginNameFromSpec(pv *v1.PersistentVolume, vol *v1.Volume) (string, error)
		IsPVMigratable(pv *v1.PersistentVolume) bool
		IsInlineMigratable(vol *v1.Volume) bool
	*/
	ctl := New()

	tests := []struct {
		name  string
		input interface{}
		copy  interface{}
		test  func(input interface{}) (interface{}, error)
	}{
		{
			name:  "TranslateInTreePVToCSI",
			input: inTreePV(),
			// TODO(dyzz) How do I get a copy of the input object that I can use
			// to reference to see if the input changed after calling the function
			copy: inTreePV(),
			test: func(input interface{}) (interface{}, error) {
				return ctl.TranslateInTreePVToCSI(input.(*v1.PersistentVolume))
			},
		},
		{
			name:  "TranslateInTreeStorageClassToCSI",
			input: inTreeSC(),
			copy:  inTreeSC(),
			test: func(input interface{}) (interface{}, error) {
				return ctl.TranslateInTreeStorageClassToCSI("kubernetes.io/gce-pd", input.(*storagev1.StorageClass))
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			output, err := tc.test(tc.input)
			if err != nil {
				t.Fatalf("Failed to Translate PV: %v", err)
			}

			if !reflect.DeepEqual(tc.copy, tc.input) {
				t.Errorf("Wanted original to not be modified, instead got new: %v", tc.input)
			}

			if reflect.DeepEqual(tc.copy, output) {
				t.Errorf("New is exactly the same as original, should have been changed")
			}
		})
	}

}

// TODO(dyzz) fully specify all these objects to make sure none of the nested
// pointer objects are being modified
func inTreeSC() *storagev1.StorageClass {
	rp := v1.PersistentVolumeReclaimDelete
	vm := storagev1.VolumeBindingImmediate
	return &storagev1.StorageClass{
		ReclaimPolicy:     &rp,
		MountOptions:      []string{"foo"},
		VolumeBindingMode: &vm,
	}
}

func inTreePV() *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
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
	}
}

// TODO: test for not modifying the original PV.
