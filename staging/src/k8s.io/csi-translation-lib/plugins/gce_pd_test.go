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

package plugins

import (
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init"
)

func NewStorageClass(params map[string]string, allowedTopologies []v1.TopologySelectorTerm) *storage.StorageClass {
	return &storage.StorageClass{
		Parameters:        params,
		AllowedTopologies: allowedTopologies,
	}
}

func TestTranslatePDInTreeStorageClassToCSI(t *testing.T) {
	g := NewGCEPersistentDiskCSITranslator()
	logger, _ := ktesting.NewTestContext(t)

	tcs := []struct {
		name       string
		options    *storage.StorageClass
		expOptions *storage.StorageClass
		expErr     bool
	}{
		{
			name:       "nothing special",
			options:    NewStorageClass(map[string]string{"foo": "bar"}, nil),
			expOptions: NewStorageClass(map[string]string{"foo": "bar"}, nil),
		},
		{
			name:       "fstype",
			options:    NewStorageClass(map[string]string{"fstype": "myfs"}, nil),
			expOptions: NewStorageClass(map[string]string{"csi.storage.k8s.io/fstype": "myfs"}, nil),
		},
		{
			name:       "empty params",
			options:    NewStorageClass(map[string]string{}, nil),
			expOptions: NewStorageClass(map[string]string{}, nil),
		},
		{
			name:       "zone",
			options:    NewStorageClass(map[string]string{"zone": "foo"}, nil),
			expOptions: NewStorageClass(map[string]string{}, generateToplogySelectors(GCEPDTopologyKey, []string{"foo"})),
		},
		{
			name:       "zones",
			options:    NewStorageClass(map[string]string{"zones": "foo,bar,baz"}, nil),
			expOptions: NewStorageClass(map[string]string{}, generateToplogySelectors(GCEPDTopologyKey, []string{"foo", "bar", "baz"})),
		},
		{
			name:       "some normal topology",
			options:    NewStorageClass(map[string]string{}, generateToplogySelectors(GCEPDTopologyKey, []string{"foo"})),
			expOptions: NewStorageClass(map[string]string{}, generateToplogySelectors(GCEPDTopologyKey, []string{"foo"})),
		},
		{
			name:       "some translated topology",
			options:    NewStorageClass(map[string]string{}, generateToplogySelectors(v1.LabelFailureDomainBetaZone, []string{"foo"})),
			expOptions: NewStorageClass(map[string]string{}, generateToplogySelectors(GCEPDTopologyKey, []string{"foo"})),
		},
		{
			name:    "zone and topology",
			options: NewStorageClass(map[string]string{"zone": "foo"}, generateToplogySelectors(GCEPDTopologyKey, []string{"foo"})),
			expErr:  true,
		},
	}

	for _, tc := range tcs {
		t.Logf("Testing %v", tc.name)
		gotOptions, err := g.TranslateInTreeStorageClassToCSI(logger, tc.options)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect error but got: %v", err)
		}
		if err == nil && tc.expErr {
			t.Errorf("Expected error, but did not get one.")
		}
		if !reflect.DeepEqual(gotOptions, tc.expOptions) {
			t.Errorf("Got parameters: %v, expected :%v", gotOptions, tc.expOptions)
		}
	}
}

func TestRepairVolumeHandle(t *testing.T) {
	testCases := []struct {
		name                 string
		volumeHandle         string
		nodeID               string
		expectedVolumeHandle string
		expectedErr          bool
	}{
		{
			name:                 "fully specified",
			volumeHandle:         fmt.Sprintf(volIDZonalFmt, "foo", "bar", "baz"),
			nodeID:               fmt.Sprintf(nodeIDFmt, "bing", "bada", "boom"),
			expectedVolumeHandle: fmt.Sprintf(volIDZonalFmt, "foo", "bar", "baz"),
		},
		{
			name:                 "fully specified (regional)",
			volumeHandle:         fmt.Sprintf(volIDRegionalFmt, "foo", "us-central1-c", "baz"),
			nodeID:               fmt.Sprintf(nodeIDFmt, "bing", "bada", "boom"),
			expectedVolumeHandle: fmt.Sprintf(volIDRegionalFmt, "foo", "us-central1-c", "baz"),
		},
		{
			name:                 "no project",
			volumeHandle:         fmt.Sprintf(volIDZonalFmt, UnspecifiedValue, "bar", "baz"),
			nodeID:               fmt.Sprintf(nodeIDFmt, "bing", "bada", "boom"),
			expectedVolumeHandle: fmt.Sprintf(volIDZonalFmt, "bing", "bar", "baz"),
		},
		{
			name:                 "no project or zone",
			volumeHandle:         fmt.Sprintf(volIDZonalFmt, UnspecifiedValue, UnspecifiedValue, "baz"),
			nodeID:               fmt.Sprintf(nodeIDFmt, "bing", "bada", "boom"),
			expectedVolumeHandle: fmt.Sprintf(volIDZonalFmt, "bing", "bada", "baz"),
		},
		{
			name:                 "no project or region",
			volumeHandle:         fmt.Sprintf(volIDRegionalFmt, UnspecifiedValue, UnspecifiedValue, "baz"),
			nodeID:               fmt.Sprintf(nodeIDFmt, "bing", "us-central1-c", "boom"),
			expectedVolumeHandle: fmt.Sprintf(volIDRegionalFmt, "bing", "us-central1", "baz"),
		},
		{
			name:                 "no project (regional)",
			volumeHandle:         fmt.Sprintf(volIDRegionalFmt, UnspecifiedValue, "us-west1", "baz"),
			nodeID:               fmt.Sprintf(nodeIDFmt, "bing", "us-central1-c", "boom"),
			expectedVolumeHandle: fmt.Sprintf(volIDRegionalFmt, "bing", "us-west1", "baz"),
		},
		{
			name:         "invalid handle",
			volumeHandle: "foo",
			nodeID:       fmt.Sprintf(nodeIDFmt, "bing", "us-central1-c", "boom"),
			expectedErr:  true,
		},
		{
			name:         "invalid node ID",
			volumeHandle: fmt.Sprintf(volIDRegionalFmt, UnspecifiedValue, "us-west1", "baz"),
			nodeID:       "foo",
			expectedErr:  true,
		},
	}
	g := NewGCEPersistentDiskCSITranslator()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			gotVolumeHandle, err := g.RepairVolumeHandle(tc.volumeHandle, tc.nodeID)
			if err != nil && !tc.expectedErr {
				if !tc.expectedErr {
					t.Fatalf("Got error: %v, but expected none", err)
				}
				return
			}
			if err == nil && tc.expectedErr {
				t.Fatal("Got no error, but expected one")
			}

			if gotVolumeHandle != tc.expectedVolumeHandle {
				t.Fatalf("Got volume handle %s, but expected %s", gotVolumeHandle, tc.expectedVolumeHandle)
			}
		})
	}
}

func TestBackwardCompatibleAccessModes(t *testing.T) {
	testCases := []struct {
		name           string
		accessModes    []v1.PersistentVolumeAccessMode
		expAccessModes []v1.PersistentVolumeAccessMode
	}{
		{
			name: "ROX",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadOnlyMany,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadOnlyMany,
			},
		},
		{
			name: "RWO",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
		},
		{
			name: "RWX",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteMany,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
		},
		{
			name: "RWO, ROX",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadOnlyMany,
				v1.ReadWriteOnce,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
		},
		{
			name: "RWO, RWX",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadWriteMany,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
		},
		{
			name: "RWX, ROX",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteMany,
				v1.ReadOnlyMany,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
		},
		{
			name: "RWX, ROX, RWO",
			accessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteMany,
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			expAccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
		},
	}

	for _, tc := range testCases {
		t.Logf("running test: %v", tc.name)

		got := backwardCompatibleAccessModes(tc.accessModes)

		if !reflect.DeepEqual(tc.expAccessModes, got) {
			t.Fatalf("Expected access modes: %v, instead got: %v", tc.expAccessModes, got)
		}
	}
}

func TestInlineReadOnly(t *testing.T) {
	g := NewGCEPersistentDiskCSITranslator()
	logger, _ := ktesting.NewTestContext(t)
	pv, err := g.TranslateInTreeInlineVolumeToCSI(logger, &v1.Volume{
		VolumeSource: v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName:   "foo",
				ReadOnly: true,
			},
		},
	}, "")
	if err != nil {
		t.Fatalf("Failed to translate in tree inline volume to CSI: %v", err)
	}

	if pv == nil || pv.Spec.PersistentVolumeSource.CSI == nil {
		t.Fatal("PV or volume source unexpectedly nil")
	}

	if !pv.Spec.PersistentVolumeSource.CSI.ReadOnly {
		t.Error("PV readonly value not true")
	}

	ams := pv.Spec.AccessModes
	if len(ams) != 1 {
		t.Errorf("got am %v, expected length of 1", ams)
	}

	if ams[0] != v1.ReadOnlyMany {
		t.Errorf("got am %v, expected access mode of ReadOnlyMany", ams[0])
	}
}

func TestTranslateInTreePVToCSIVolIDFmt(t *testing.T) {
	g := NewGCEPersistentDiskCSITranslator()
	logger, _ := ktesting.NewTestContext(t)
	pdName := "pd-name"
	tests := []struct {
		desc               string
		topologyLabelKey   string
		topologyLabelValue string
		wantVolId          string
	}{
		{
			desc:               "beta topology key zonal",
			topologyLabelKey:   v1.LabelFailureDomainBetaZone,
			topologyLabelValue: "us-east1-a",
			wantVolId:          "projects/UNSPECIFIED/zones/us-east1-a/disks/pd-name",
		},
		{
			desc:               "v1 topology key zonal",
			topologyLabelKey:   v1.LabelTopologyZone,
			topologyLabelValue: "us-east1-a",
			wantVolId:          "projects/UNSPECIFIED/zones/us-east1-a/disks/pd-name",
		},
		{
			desc:               "beta topology key regional",
			topologyLabelKey:   v1.LabelFailureDomainBetaZone,
			topologyLabelValue: "us-central1-a__us-central1-c",
			wantVolId:          "projects/UNSPECIFIED/regions/us-central1/disks/pd-name",
		},
		{
			desc:               "v1 topology key regional",
			topologyLabelKey:   v1.LabelTopologyZone,
			topologyLabelValue: "us-central1-a__us-central1-c",
			wantVolId:          "projects/UNSPECIFIED/regions/us-central1/disks/pd-name",
		},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			translatedPV, err := g.TranslateInTreePVToCSI(logger, &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{tc.topologyLabelKey: tc.topologyLabelValue},
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: pdName,
						},
					},
				},
			})
			if err != nil {
				t.Errorf("got error translating in-tree PV to CSI: %v", err)
			}
			if got := translatedPV.Spec.PersistentVolumeSource.CSI.VolumeHandle; got != tc.wantVolId {
				t.Errorf("got translated volume handle: %q, want %q", got, tc.wantVolId)
			}
		})
	}
}
