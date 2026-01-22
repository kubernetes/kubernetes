/*
Copyright 2021 The Kubernetes Authors.

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
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

var nodeLabels = map[string]string{
	"test-key1": "test-value1",
	"test-key2": "test-value2",
}

func TestCheckVolumeNodeAffinity(t *testing.T) {
	type affinityTest struct {
		name          string
		expectSuccess bool
		pv            *v1.PersistentVolume
	}

	cases := []affinityTest{
		{
			name:          "valid-nil",
			expectSuccess: true,
			pv:            testVolumeWithNodeAffinity(t, nil),
		},
		{
			name:          "valid-no-constraints",
			expectSuccess: true,
			pv:            testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{}),
		},
		{
			name:          "select-nothing",
			expectSuccess: false,
			pv:            testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{Required: &v1.NodeSelector{}}),
		},
		{
			name:          "select-nothing-empty-terms",
			expectSuccess: false,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{},
						},
					},
				},
			}),
		},
		{
			name:          "valid-multiple-terms",
			expectSuccess: true,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key3",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
							},
						},
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "valid-multiple-match-expressions",
			expectSuccess: true,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "invalid-multiple-match-expressions-key",
			expectSuccess: false,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
								{
									Key:      "test-key3",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "invalid-multiple-match-expressions-values",
			expectSuccess: false,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key1",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value3", "test-value4"},
								},
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value2"},
								},
							},
						},
					},
				},
			}),
		},
		{
			name:          "invalid-multiple-terms",
			expectSuccess: false,
			pv: testVolumeWithNodeAffinity(t, &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key3",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value1", "test-value3"},
								},
							},
						},
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "test-key2",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"test-value0", "test-value1"},
								},
							},
						},
					},
				},
			}),
		},
	}

	for _, c := range cases {
		err := CheckNodeAffinity(c.pv, nodeLabels)

		if err != nil && c.expectSuccess {
			t.Errorf("CheckTopology %v returned error: %v", c.name, err)
		}
		if err == nil && !c.expectSuccess {
			t.Errorf("CheckTopology %v returned success, expected error", c.name)
		}
	}
}

func testVolumeWithNodeAffinity(t *testing.T, affinity *v1.VolumeNodeAffinity) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "test-constraints"},
		Spec: v1.PersistentVolumeSpec{
			NodeAffinity: affinity,
		},
	}
}

func TestPersistentVolumeClaimHasClass(t *testing.T) {
	testCases := []struct {
		name string
		pvc  *v1.PersistentVolumeClaim
		want bool
	}{
		{
			name: "no storage class",
			pvc:  &v1.PersistentVolumeClaim{},
			want: false,
		},
		{
			name: "storage class set on annotation",
			pvc: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.BetaStorageClassAnnotation: "",
					},
				},
			},
			want: true,
		},
		{
			name: "storage class set on spec",
			pvc: &v1.PersistentVolumeClaim{
				Spec: v1.PersistentVolumeClaimSpec{
					StorageClassName: ptr.To(""),
				},
			},
			want: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := PersistentVolumeClaimHasClass(tc.pvc)
			if got != tc.want {
				t.Errorf("PersistentVolumeClaimHasClass() = %v, want %v", got, tc.want)
			}
		})
	}
}
