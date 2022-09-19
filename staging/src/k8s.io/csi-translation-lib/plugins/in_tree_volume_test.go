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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	useast1aGALabels = map[string]string{
		v1.LabelTopologyZone:   "us-east1-a",
		v1.LabelTopologyRegion: "us-east1",
	}
	useast1aGANodeSelectorTermZoneFirst = []v1.NodeSelectorTerm{
		{
			MatchExpressions: []v1.NodeSelectorRequirement{
				{
					Key:      v1.LabelTopologyZone,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"us-east1-a"},
				},
				{
					Key:      v1.LabelTopologyRegion,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"us-east1"},
				},
			},
		},
	}

	useast1aGANodeSelectorTermRegionFirst = []v1.NodeSelectorTerm{
		{
			MatchExpressions: []v1.NodeSelectorRequirement{
				{
					Key:      v1.LabelTopologyRegion,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"us-east1"},
				},
				{
					Key:      v1.LabelTopologyZone,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"us-east1-a"},
				},
			},
		},
	}

	uswest2bBetaLabels = map[string]string{
		v1.LabelFailureDomainBetaZone:   "us-west2-b",
		v1.LabelFailureDomainBetaRegion: "us-west2",
	}

	uswest2bBetaNodeSelectorTermZoneFirst = []v1.NodeSelectorTerm{
		{
			MatchExpressions: []v1.NodeSelectorRequirement{
				{
					Key:      v1.LabelFailureDomainBetaZone,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"us-west2-b"},
				},
				{
					Key:      v1.LabelFailureDomainBetaRegion,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"us-west2"},
				},
			},
		},
	}

	uswest2bBetaNodeSelectorTermRegionFirst = []v1.NodeSelectorTerm{
		{
			MatchExpressions: []v1.NodeSelectorRequirement{
				{
					Key:      v1.LabelFailureDomainBetaRegion,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"us-west2"},
				},
				{
					Key:      v1.LabelFailureDomainBetaZone,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{"us-west2-b"},
				},
			},
		},
	}
)

func TestTranslateTopologyFromCSIToInTree(t *testing.T) {
	testCases := []struct {
		name                      string
		key                       string
		expErr                    bool
		regionParser              regionParserFn
		pv                        *v1.PersistentVolume
		expectedNodeSelectorTerms []v1.NodeSelectorTerm
		expectedLabels            map[string]string
	}{
		{
			name:         "Remove CSI Topology Key and do not change existing GA Kubernetes topology",
			key:          GCEPDTopologyKey,
			expErr:       false,
			regionParser: gceGetRegionFromZones,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
					Labels: useast1aGALabels,
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      v1.LabelTopologyRegion,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-east1"},
										},
										{
											Key:      GCEPDTopologyKey,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-east1-a"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedNodeSelectorTerms: useast1aGANodeSelectorTermRegionFirst,
			expectedLabels:            useast1aGALabels,
		},
		{
			name:         "Remove CSI Topology Key and do not change existing Beta Kubernetes topology",
			key:          GCEPDTopologyKey,
			expErr:       false,
			regionParser: gceGetRegionFromZones,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
					Labels: uswest2bBetaLabels,
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      v1.LabelFailureDomainBetaRegion,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-west2"},
										},
										{
											Key:      GCEPDTopologyKey,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-west2-b"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedNodeSelectorTerms: uswest2bBetaNodeSelectorTermRegionFirst,
			expectedLabels:            uswest2bBetaLabels,
		},
		{
			name:         "Remove CSI Topology Key and add Kubernetes topology from NodeAffinity, ignore labels",
			key:          GCEPDTopologyKey,
			expErr:       false,
			regionParser: gceGetRegionFromZones,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
					Labels: map[string]string{
						v1.LabelTopologyZone:   "existingZone",
						v1.LabelTopologyRegion: "existingRegion",
					},
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      GCEPDTopologyKey,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-east1-a"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedNodeSelectorTerms: useast1aGANodeSelectorTermZoneFirst,
			expectedLabels: map[string]string{
				v1.LabelTopologyRegion: "existingRegion",
				v1.LabelTopologyZone:   "existingZone",
			},
		},
		{
			name:         "No CSI topology label exists and no change to the NodeAffinity",
			key:          GCEPDTopologyKey,
			expErr:       false,
			regionParser: gceGetRegionFromZones,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
					Labels: map[string]string{
						v1.LabelTopologyZone:   "existingZone",
						v1.LabelTopologyRegion: "existingRegion",
					},
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{},
						},
					},
				},
			},
			expectedNodeSelectorTerms: []v1.NodeSelectorTerm{},
			expectedLabels: map[string]string{
				v1.LabelTopologyZone:   "existingZone",
				v1.LabelTopologyRegion: "existingRegion",
			},
		},
		{
			name:         "Generate GA labels and kubernetes topology only from CSI topology",
			key:          GCEPDTopologyKey,
			expErr:       false,
			regionParser: gceGetRegionFromZones,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      GCEPDTopologyKey,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-east1-a"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedNodeSelectorTerms: useast1aGANodeSelectorTermZoneFirst,
			expectedLabels:            useast1aGALabels,
		},
		{
			name:         "Generate Beta labels and kubernetes topology from Beta NodeAffinity",
			key:          GCEPDTopologyKey,
			expErr:       false,
			regionParser: gceGetRegionFromZones,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      v1.LabelFailureDomainBetaZone,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-west2-b"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedNodeSelectorTerms: uswest2bBetaNodeSelectorTermZoneFirst,
			expectedLabels:            uswest2bBetaLabels,
		},
		{
			name:   "regionParser is missing and only zone labels get generated",
			key:    GCEPDTopologyKey,
			expErr: false,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      GCEPDTopologyKey,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-east1-a"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedNodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      v1.LabelTopologyZone,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-a"},
						},
					},
				},
			},
			expectedLabels: map[string]string{
				v1.LabelTopologyZone: "us-east1-a",
			},
		},
		{
			name:         "Replace multi-term CSI Topology Key and add Region Kubernetes topology for both",
			key:          GCEPDTopologyKey,
			expErr:       false,
			regionParser: gceGetRegionFromZones,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      GCEPDTopologyKey,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-east1-a"},
										},
									},
								},
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      GCEPDTopologyKey,
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"us-east1-c"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedNodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      v1.LabelTopologyZone,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-a"},
						},
						{
							Key:      v1.LabelTopologyRegion,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      v1.LabelTopologyZone,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-c"},
						},
						{
							Key:      v1.LabelTopologyRegion,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1"},
						},
					},
				},
			},
			expectedLabels: map[string]string{
				v1.LabelTopologyZone:   "us-east1-a__us-east1-c",
				v1.LabelTopologyRegion: "us-east1",
			},
		},
	}

	for _, tc := range testCases {
		t.Logf("Running test: %v", tc.name)
		err := translateTopologyFromCSIToInTree(tc.pv, tc.key, tc.regionParser)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect an error, got: %v", err)
		}
		if err == nil && tc.expErr {
			t.Errorf("Expected an error but did not get one")
		}

		if !reflect.DeepEqual(tc.pv.Spec.NodeAffinity.Required.NodeSelectorTerms, tc.expectedNodeSelectorTerms) {
			t.Errorf("Expected topology: %v, but got: %v", tc.expectedNodeSelectorTerms, tc.pv.Spec.NodeAffinity.Required.NodeSelectorTerms)
		}
		if !reflect.DeepEqual(tc.pv.Labels, tc.expectedLabels) {
			t.Errorf("Expected labels: %v, but got: %v", tc.expectedLabels, tc.pv.Labels)
		}
	}
}

func TestTranslateTopologyFromInTreeToCSI(t *testing.T) {
	testCases := []struct {
		name                      string
		key                       string
		expErr                    bool
		pv                        *v1.PersistentVolume
		expectedNodeSelectorTerms []v1.NodeSelectorTerm
	}{
		{
			name:   "Replace GA Kubernetes Zone Topology to GCE CSI Topology",
			key:    GCEPDTopologyKey,
			expErr: false,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
					Labels: useast1aGALabels,
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: useast1aGANodeSelectorTermZoneFirst,
						},
					},
				},
			},
			expectedNodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      GCEPDTopologyKey,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-a"},
						},
						{
							Key:      v1.LabelTopologyRegion,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1"},
						},
					},
				},
			},
		},
		{
			name:   "Replace Beta Kubernetes Topology to GCE CSI Topology and upgrade region label",
			key:    GCEPDTopologyKey,
			expErr: false,
			pv: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gcepd", Namespace: "myns",
					Labels: useast1aGALabels,
				},
				Spec: v1.PersistentVolumeSpec{
					NodeAffinity: &v1.VolumeNodeAffinity{
						Required: &v1.NodeSelector{
							NodeSelectorTerms: uswest2bBetaNodeSelectorTermZoneFirst,
						},
					},
				},
			},
			expectedNodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      GCEPDTopologyKey,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-west2-b"},
						},
						{
							Key:      v1.LabelTopologyRegion,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-west2"},
						},
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Logf("Running test: %v", tc.name)
		err := translateTopologyFromInTreeToCSI(tc.pv, tc.key)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect an error, got: %v", err)
		}
		if err == nil && tc.expErr {
			t.Errorf("Expected an error but did not get one")
		}

		if !reflect.DeepEqual(tc.pv.Spec.NodeAffinity.Required.NodeSelectorTerms, tc.expectedNodeSelectorTerms) {
			t.Errorf("Expected topology: %v, but got: %v", tc.expectedNodeSelectorTerms, tc.pv.Spec.NodeAffinity.Required.NodeSelectorTerms)
		}
	}
}

func TestTranslateAllowedTopologies(t *testing.T) {
	testCases := []struct {
		name            string
		topology        []v1.TopologySelectorTerm
		expectedToplogy []v1.TopologySelectorTerm
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
			expectedToplogy: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    "test",
							Values: []string{"foo", "bar"},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Logf("Running test: %v", tc.name)
		gotTop, err := translateAllowedTopologies(tc.topology, GCEPDTopologyKey)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		if !reflect.DeepEqual(gotTop, tc.expectedToplogy) {
			t.Errorf("Expected topology: %v, but got: %v", tc.expectedToplogy, gotTop)
		}
	}
}

func TestAddTopology(t *testing.T) {
	testCases := []struct {
		name             string
		topologyKey      string
		zones            []string
		expErr           bool
		expectedAffinity *v1.VolumeNodeAffinity
	}{
		{
			name:        "empty zones",
			topologyKey: GCEPDTopologyKey,
			zones:       nil,
			expErr:      true,
		},
		{
			name:        "only whitespace-named zones",
			topologyKey: GCEPDTopologyKey,
			zones:       []string{" ", "\n", "\t", "  "},
			expErr:      true,
		},
		{
			name:        "including whitespace-named zones",
			topologyKey: GCEPDTopologyKey,
			zones:       []string{" ", "us-central1-a"},
			expErr:      false,
			expectedAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      GCEPDTopologyKey,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"us-central1-a"},
								},
							},
						},
					},
				},
			},
		},
		{
			name:        "unsorted zones",
			topologyKey: GCEPDTopologyKey,
			zones:       []string{"us-central1-f", "us-central1-a", "us-central1-c", "us-central1-b"},
			expErr:      false,
			expectedAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      GCEPDTopologyKey,
									Operator: v1.NodeSelectorOpIn,
									// Values are expected to be ordered
									Values: []string{"us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Logf("Running test: %v", tc.name)
		pv := &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{},
		}
		err := addTopology(pv, tc.topologyKey, tc.zones)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect an error, got: %v", err)
		}
		if err == nil && tc.expErr {
			t.Errorf("Expected an error but did not get one")
		}
		if err == nil && !reflect.DeepEqual(pv.Spec.NodeAffinity, tc.expectedAffinity) {
			t.Errorf("Expected affinity: %v, but got: %v", tc.expectedAffinity, pv.Spec.NodeAffinity)
		}
	}
}

func TestReplaceTopology(t *testing.T) {
	testCases := []struct {
		name             string
		oldKey           string
		newKey           string
		pv               *v1.PersistentVolume
		expOk            bool
		expectedAffinity *v1.VolumeNodeAffinity
	}{
		{
			name:   "Replace single csi topology from PV",
			oldKey: GCEPDTopologyKey,
			newKey: v1.LabelTopologyZone,
			pv: makePVWithNodeSelectorTerms([]v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      GCEPDTopologyKey,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-a"},
						},
					},
				},
			}),
			expOk: true,
			expectedAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      v1.LabelTopologyZone,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"us-east1-a"},
								},
							},
						},
					},
				},
			},
		},
		{
			name:   "Not found the topology key so do nothing",
			oldKey: GCEPDTopologyKey,
			newKey: v1.LabelTopologyZone,
			pv: makePVWithNodeSelectorTerms([]v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      v1.LabelTopologyZone,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-a"},
						},
					},
				},
			}),
			expOk: false,
			expectedAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      v1.LabelTopologyZone,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"us-east1-a"},
								},
							},
						},
					},
				},
			},
		},
		{
			name:   "Replace the topology key from multiple terms",
			oldKey: GCEPDTopologyKey,
			newKey: v1.LabelTopologyZone,
			pv: makePVWithNodeSelectorTerms([]v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      GCEPDTopologyKey,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-a"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      GCEPDTopologyKey,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-c"},
						},
					},
				},
			}),
			expOk: true,
			expectedAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      v1.LabelTopologyZone,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"us-east1-a"},
								},
							},
						},
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      v1.LabelTopologyZone,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"us-east1-c"},
								},
							},
						},
					},
				},
			},
		},
		{
			name:   "Replace the topology key from single term and not combine topology key",
			oldKey: GCEPDTopologyKey,
			newKey: v1.LabelTopologyZone,
			pv: makePVWithNodeSelectorTerms([]v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      GCEPDTopologyKey,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-a"},
						},
						{
							Key:      v1.LabelTopologyZone,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"us-east1-c"},
						},
					},
				},
			}),
			expOk: true,
			expectedAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      v1.LabelTopologyZone,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"us-east1-a"},
								},
								{
									Key:      v1.LabelTopologyZone,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"us-east1-c"},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Logf("Running test: %v", tc.name)
		err := replaceTopology(tc.pv, tc.oldKey, tc.newKey)
		if err != nil && tc.expOk {
			t.Errorf("Expected no err: %v, but got err: %v", tc.expOk, err)
		}
		if !reflect.DeepEqual(tc.pv.Spec.NodeAffinity, tc.expectedAffinity) {
			t.Errorf("Expected affinity: %v, but got: %v", tc.expectedAffinity, tc.pv.Spec.NodeAffinity)
		}
	}
}

func makePVWithNodeSelectorTerms(nodeSelectorTerms []v1.NodeSelectorTerm) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: nodeSelectorTerms,
				},
			},
		},
	}

}
