/*
Copyright 2017 The Kubernetes Authors.

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

package schedulercache

import (
	"fmt"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/v1"
)

// TODO: remove when alpha support for affinity is removed
func TestReconcileAffinity(t *testing.T) {
	baseAffinity := &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      "foo",
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{"bar", "value2"},
							},
						},
					},
				},
			},
		},
		PodAffinity: &v1.PodAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "security",
								Operator: metav1.LabelSelectorOpDoesNotExist,
								Values:   []string{"securityscan"},
							},
						},
					},
					TopologyKey: "topologyKey1",
				},
			},
		},
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "service",
								Operator: metav1.LabelSelectorOpIn,
								Values:   []string{"S1", "value2"},
							},
						},
					},
					TopologyKey: "topologyKey2",
					Namespaces:  []string{"ns1"},
				},
			},
		},
	}

	nodeAffinityAnnotation := map[string]string{
		v1.AffinityAnnotationKey: `
		{"nodeAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": [
			{
				"weight": 2,
				"preference": {"matchExpressions": [
					{
						"key": "foo",
						"operator": "In", "values": ["bar"]
					}
				]}
			}
		]}}`,
	}

	testCases := []struct {
		pod                *v1.Pod
		expected           *v1.Affinity
		annotationsEnabled bool
	}{
		{
			// affinity is set in both PodSpec and annotations; take from PodSpec.
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: nodeAffinityAnnotation,
				},
				Spec: v1.PodSpec{
					Affinity: baseAffinity,
				},
			},
			expected:           baseAffinity,
			annotationsEnabled: true,
		},
		{
			// affinity is only set in annotation; take from annotation.
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: nodeAffinityAnnotation,
				},
			},
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
						{
							Weight: 2,
							Preference: v1.NodeSelectorTerm{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "foo",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"bar"},
									},
								},
							},
						},
					},
				},
			},
			annotationsEnabled: true,
		},
	}

	for i, tc := range testCases {
		utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("AffinityInAnnotations=%t", tc.annotationsEnabled))
		affinity := ReconcileAffinity(tc.pod)
		if !reflect.DeepEqual(affinity, tc.expected) {
			t.Errorf("[%v] Did not get expected affinity:\n\n%v\n\n. got:\n\n %v", i, tc.expected, affinity)
		}
	}
}
