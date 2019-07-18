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
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
)

func NewStorageClass(params map[string]string, allowedTopologies []v1.TopologySelectorTerm) *storage.StorageClass {
	return &storage.StorageClass{
		Parameters:        params,
		AllowedTopologies: allowedTopologies,
	}
}

func TestTranslatePDInTreeStorageClassToCSI(t *testing.T) {
	g := NewGCEPersistentDiskCSITranslator()

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
			options:    NewStorageClass(map[string]string{}, generateToplogySelectors(v1.LabelZoneFailureDomain, []string{"foo"})),
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
		gotOptions, err := g.TranslateInTreeStorageClassToCSI(tc.options)
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

func TestTranslateAllowedTopologies(t *testing.T) {
	testCases := []struct {
		name            string
		topology        []v1.TopologySelectorTerm
		expectedToplogy []v1.TopologySelectorTerm
		expErr          bool
	}{
		{
			name:     "no translation",
			topology: generateToplogySelectors(GCEPDTopologyKey, []string{"foo", "bar"}),
			expectedToplogy: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    GCEPDTopologyKey,
							Values: []string{"foo", "bar"},
						},
					},
				},
			},
		},
		{
			name: "translate",
			topology: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    "failure-domain.beta.kubernetes.io/zone",
							Values: []string{"foo", "bar"},
						},
					},
				},
			},
			expectedToplogy: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    GCEPDTopologyKey,
							Values: []string{"foo", "bar"},
						},
					},
				},
			},
		},
		{
			name: "combo",
			topology: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    "failure-domain.beta.kubernetes.io/zone",
							Values: []string{"foo", "bar"},
						},
						{
							Key:    GCEPDTopologyKey,
							Values: []string{"boo", "baz"},
						},
					},
				},
			},
			expectedToplogy: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    GCEPDTopologyKey,
							Values: []string{"foo", "bar"},
						},
						{
							Key:    GCEPDTopologyKey,
							Values: []string{"boo", "baz"},
						},
					},
				},
			},
		},
		{
			name: "some other key",
			topology: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    "test",
							Values: []string{"foo", "bar"},
						},
					},
				},
			},
			expErr: true,
		},
	}

	for _, tc := range testCases {
		t.Logf("Running test: %v", tc.name)
		gotTop, err := translateAllowedTopologies(tc.topology)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect an error, got: %v", err)
		}
		if err == nil && tc.expErr {
			t.Errorf("Expected an error but did not get one")
		}

		if !reflect.DeepEqual(gotTop, tc.expectedToplogy) {
			t.Errorf("Expected topology: %v, but got: %v", tc.expectedToplogy, gotTop)
		}
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
