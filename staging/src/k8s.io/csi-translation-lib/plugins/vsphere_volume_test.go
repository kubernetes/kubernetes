/*
Copyright 2020 The Kubernetes Authors.

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

package plugins

import (
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestTranslatevSphereInTreeStorageClassToCSI(t *testing.T) {
	translator := NewvSphereCSITranslator()
	topologySelectorTerm := v1.TopologySelectorTerm{[]v1.TopologySelectorLabelRequirement{
		{
			Key:    v1.LabelFailureDomainBetaZone,
			Values: []string{"zone-a"},
		},
		{
			Key:    v1.LabelFailureDomainBetaRegion,
			Values: []string{"region-a"},
		},
	}}
	cases := []struct {
		name   string
		sc     *storage.StorageClass
		expSc  *storage.StorageClass
		expErr bool
	}{
		{
			name:   "expect error when sc is nil",
			sc:     nil,
			expSc:  nil,
			expErr: true,
		},
		{
			name:  "translate with no parameter",
			sc:    NewStorageClass(map[string]string{}, nil),
			expSc: NewStorageClass(map[string]string{paramcsiMigration: "true"}, nil),
		},
		{
			name:  "translate with unknown parameter",
			sc:    NewStorageClass(map[string]string{"unknownparam": "value"}, nil),
			expSc: NewStorageClass(map[string]string{paramcsiMigration: "true"}, nil),
		},
		{
			name:  "translate with storagepolicyname and datastore",
			sc:    NewStorageClass(map[string]string{"storagepolicyname": "test-policy-name", "datastore": "vsanDatastore"}, nil),
			expSc: NewStorageClass(map[string]string{"storagepolicyname": "test-policy-name", "datastore-migrationparam": "vsanDatastore", paramcsiMigration: "true"}, nil),
		},
		{
			name:  "translate with fstype",
			sc:    NewStorageClass(map[string]string{"fstype": "ext4"}, nil),
			expSc: NewStorageClass(map[string]string{"csi.storage.k8s.io/fstype": "ext4", paramcsiMigration: "true"}, nil),
		},
		{
			name:  "translate with storagepolicyname and fstype",
			sc:    NewStorageClass(map[string]string{"storagepolicyname": "test-policy-name", "fstype": "ext4"}, nil),
			expSc: NewStorageClass(map[string]string{"csi.storage.k8s.io/fstype": "ext4", "storagepolicyname": "test-policy-name", paramcsiMigration: "true"}, nil),
		},
		{
			name:  "translate with no parameter and allowedTopology",
			sc:    NewStorageClass(map[string]string{}, []v1.TopologySelectorTerm{topologySelectorTerm}),
			expSc: NewStorageClass(map[string]string{paramcsiMigration: "true"}, []v1.TopologySelectorTerm{topologySelectorTerm}),
		},
		{
			name:  "translate with storagepolicyname and allowedTopology",
			sc:    NewStorageClass(map[string]string{"storagepolicyname": "test-policy-name"}, []v1.TopologySelectorTerm{topologySelectorTerm}),
			expSc: NewStorageClass(map[string]string{"storagepolicyname": "test-policy-name", paramcsiMigration: "true"}, []v1.TopologySelectorTerm{topologySelectorTerm}),
		},
		{
			name:  "translate with raw vSAN policy parameters, datastore and diskformat",
			sc:    NewStorageClass(map[string]string{"hostfailurestotolerate": "2", "datastore": "vsanDatastore", "diskformat": "thin"}, []v1.TopologySelectorTerm{topologySelectorTerm}),
			expSc: NewStorageClass(map[string]string{"hostfailurestotolerate-migrationparam": "2", "datastore-migrationparam": "vsanDatastore", "diskformat-migrationparam": "thin", paramcsiMigration: "true"}, []v1.TopologySelectorTerm{topologySelectorTerm}),
		},
		{
			name:  "translate with all parameters",
			sc:    NewStorageClass(map[string]string{"storagepolicyname": "test-policy-name", "datastore": "test-datastore-name", "fstype": "ext4", "diskformat": "thin", "hostfailurestotolerate": "1", "forceprovisioning": "yes", "cachereservation": "25", "diskstripes": "4", "objectspacereservation": "10", "iopslimit": "32"}, []v1.TopologySelectorTerm{topologySelectorTerm}),
			expSc: NewStorageClass(map[string]string{"storagepolicyname": "test-policy-name", "datastore-migrationparam": "test-datastore-name", "csi.storage.k8s.io/fstype": "ext4", "diskformat-migrationparam": "thin", "hostfailurestotolerate-migrationparam": "1", "forceprovisioning-migrationparam": "yes", "cachereservation-migrationparam": "25", "diskstripes-migrationparam": "4", "objectspacereservation-migrationparam": "10", "iopslimit-migrationparam": "32", paramcsiMigration: "true"}, []v1.TopologySelectorTerm{topologySelectorTerm}),
		},
	}
	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateInTreeStorageClassToCSI(tc.sc)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect error but got: %v", err)
		}

		if err == nil && tc.expErr {
			t.Errorf("Expected error, but did not get one.")
		}
		if !reflect.DeepEqual(got, tc.expSc) {
			t.Errorf("Got parameters: %v, expected :%v", got, tc.expSc)
		}
	}
}

func TestTranslateVSphereCSIPVToInTree(t *testing.T) {
	translator := NewvSphereCSITranslator()
	cases := []struct {
		name     string
		csiPV    *v1.PersistentVolume
		intreePV *v1.PersistentVolume
		expErr   bool
	}{
		{
			name:     "expect error when pv is nil",
			csiPV:    nil,
			intreePV: nil,
			expErr:   true,
		},
		{
			name: "expect error when pv.Spec.CSI is nil",
			csiPV: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pvc-d8b4475f-2c47-486e-9b57-43ae006f9b59",
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: nil,
					},
					AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
				},
			},
			intreePV: nil,
			expErr:   true,
		},
		{
			name: "translate valid vSphere CSI PV to vSphere in-tree PV",
			csiPV: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pvc-d8b4475f-2c47-486e-9b57-43ae006f9b59",
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:       VSphereDriverName,
							VolumeHandle: "e4073a6d-642e-4dff-8f4a-b4e3a47c4bbd",
							FSType:       "ext4",
							VolumeAttributes: map[string]string{
								paramStoragePolicyName:         "vSAN Default Storage Policy",
								AttributeInitialVolumeFilepath: "[vsanDatastore] 6785a85e-268e-6352-a2e8-02008b7afadd/kubernetes-dynamic-pvc-68734c9f-a679-42e6-a694-39632c51e31f.vmdk",
							},
						},
					},
					AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
				},
			},
			intreePV: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pvc-d8b4475f-2c47-486e-9b57-43ae006f9b59",
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
							VolumePath: "[vsanDatastore] 6785a85e-268e-6352-a2e8-02008b7afadd/kubernetes-dynamic-pvc-68734c9f-a679-42e6-a694-39632c51e31f.vmdk",
							FSType:     "ext4",
						},
					},
					AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
				},
			},
			expErr: false,
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateCSIPVToInTree(tc.csiPV)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect error but got: %v", err)
		}

		if err == nil && tc.expErr {
			t.Errorf("Expected error, but did not get one.")
		}

		if !reflect.DeepEqual(got, tc.intreePV) {
			t.Errorf("Got PV: %v, expected :%v", got, tc.intreePV)
		}
	}
}

func TestTranslateVSphereInTreePVToCSI(t *testing.T) {
	translator := NewvSphereCSITranslator()
	cases := []struct {
		name     string
		intreePV *v1.PersistentVolume
		csiPV    *v1.PersistentVolume
		expErr   bool
	}{
		{
			name:     "expect error when in-tree vsphere PV is nil",
			intreePV: &v1.PersistentVolume{},
			csiPV:    nil,
			expErr:   true,
		},
		{
			name: "translate valid vSphere in-tree PV to vSphere CSI PV",
			intreePV: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pvc-d8b4475f-2c47-486e-9b57-43ae006f9b59",
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
							VolumePath:        "[vsanDatastore] 6785a85e-268e-6352-a2e8-02008b7afadd/kubernetes-dynamic-pvc-68734c9f-a679-42e6-a694-39632c51e31f.vmdk",
							FSType:            "ext4",
							StoragePolicyName: "vSAN Default Storage Policy",
						},
					},
					AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
				},
			},
			csiPV: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pvc-d8b4475f-2c47-486e-9b57-43ae006f9b59",
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:       VSphereDriverName,
							VolumeHandle: "[vsanDatastore] 6785a85e-268e-6352-a2e8-02008b7afadd/kubernetes-dynamic-pvc-68734c9f-a679-42e6-a694-39632c51e31f.vmdk",
							FSType:       "ext4",
							VolumeAttributes: map[string]string{
								paramStoragePolicyName: "vSAN Default Storage Policy",
							},
						},
					},
					AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
				},
			},
			expErr: false,
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateInTreePVToCSI(tc.intreePV)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect error but got: %v", err)
		}

		if err == nil && tc.expErr {
			t.Errorf("Expected error, but did not get one.")
		}

		if !reflect.DeepEqual(got, tc.csiPV) {
			t.Errorf("Got PV: %v, expected :%v", got, tc.csiPV)
		}
	}
}

func TestTranslatevSphereInTreeInlineVolumeToCSI(t *testing.T) {
	translator := NewvSphereCSITranslator()
	cases := []struct {
		name         string
		inlinevolume *v1.Volume
		csiPV        *v1.PersistentVolume
		expErr       bool
	}{
		{
			name:         "expect error when inline vsphere volume is nil",
			inlinevolume: &v1.Volume{},
			csiPV:        nil,
			expErr:       true,
		},
		{
			name: "translate valid in-tree vsphere volume to vSphere CSI PV",
			inlinevolume: &v1.Volume{
				Name: "inlinevolume",
				VolumeSource: v1.VolumeSource{
					VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
						VolumePath: "[vsanDatastore] volume/inlinevolume.vmdk",
						FSType:     "ext4",
					},
				}},
			csiPV: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: fmt.Sprintf("%s-%s", VSphereDriverName, "[vsanDatastore] volume/inlinevolume.vmdk"),
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:           VSphereDriverName,
							VolumeHandle:     "[vsanDatastore] volume/inlinevolume.vmdk",
							FSType:           "ext4",
							VolumeAttributes: make(map[string]string),
						},
					},
					AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
				},
			},
			expErr: false,
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateInTreeInlineVolumeToCSI(tc.inlinevolume)
		if err == nil && tc.expErr {
			t.Errorf("Expected error, but did not get one.")
			continue
		}
		if err != nil {
			if tc.expErr {
				t.Logf("expected error received")
				continue
			} else {
				t.Errorf("Did not expect error but got: %v", err)
				continue
			}
		}
		if !reflect.DeepEqual(got, tc.csiPV) {
			t.Errorf("Got PV: %v, expected :%v", got, tc.csiPV)
		}
	}
}
