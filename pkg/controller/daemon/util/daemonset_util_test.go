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

package util

import (
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	utilpointer "k8s.io/utils/pointer"
)

func newPod(podName string, nodeName string, label map[string]string) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Labels:    label,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			NodeName: nodeName,
			Containers: []v1.Container{
				{
					Image: "foo/bar",
				},
			},
		},
	}
	pod.Name = podName
	return pod
}

func TestIsPodUpdated(t *testing.T) {
	templateGeneration := utilpointer.Int64Ptr(12345)
	badGeneration := utilpointer.Int64Ptr(12345)
	hash := "55555"
	labels := map[string]string{extensions.DaemonSetTemplateGenerationKey: fmt.Sprint(templateGeneration), extensions.DefaultDaemonSetUniqueLabelKey: hash}
	labelsNoHash := map[string]string{extensions.DaemonSetTemplateGenerationKey: fmt.Sprint(templateGeneration)}
	tests := []struct {
		test               string
		templateGeneration *int64
		pod                *v1.Pod
		hash               string
		isUpdated          bool
	}{
		{
			"templateGeneration and hash both match",
			templateGeneration,
			newPod("pod1", "node1", labels),
			hash,
			true,
		},
		{
			"templateGeneration matches, hash doesn't",
			templateGeneration,
			newPod("pod1", "node1", labels),
			hash + "123",
			true,
		},
		{
			"templateGeneration matches, no hash label, has hash",
			templateGeneration,
			newPod("pod1", "node1", labelsNoHash),
			hash,
			true,
		},
		{
			"templateGeneration matches, no hash label, no hash",
			templateGeneration,
			newPod("pod1", "node1", labelsNoHash),
			"",
			true,
		},
		{
			"templateGeneration matches, has hash label, no hash",
			templateGeneration,
			newPod("pod1", "node1", labels),
			"",
			true,
		},
		{
			"templateGeneration doesn't match, hash does",
			badGeneration,
			newPod("pod1", "node1", labels),
			hash,
			true,
		},
		{
			"templateGeneration and hash don't match",
			badGeneration,
			newPod("pod1", "node1", labels),
			hash + "123",
			false,
		},
		{
			"empty labels, no hash",
			templateGeneration,
			newPod("pod1", "node1", map[string]string{}),
			"",
			false,
		},
		{
			"empty labels",
			templateGeneration,
			newPod("pod1", "node1", map[string]string{}),
			hash,
			false,
		},
		{
			"no labels",
			templateGeneration,
			newPod("pod1", "node1", nil),
			hash,
			false,
		},
	}
	for _, test := range tests {
		updated := IsPodUpdated(test.pod, test.hash, test.templateGeneration)
		if updated != test.isUpdated {
			t.Errorf("%s: IsPodUpdated returned wrong value. Expected %t, got %t", test.test, test.isUpdated, updated)
		}
	}
}

func TestCreatePodTemplate(t *testing.T) {
	tests := []struct {
		templateGeneration *int64
		hash               string
		expectUniqueLabel  bool
	}{
		{utilpointer.Int64Ptr(1), "", false},
		{utilpointer.Int64Ptr(2), "3242341807", true},
	}
	for _, test := range tests {
		podTemplateSpec := v1.PodTemplateSpec{}
		newPodTemplate := CreatePodTemplate(podTemplateSpec, test.templateGeneration, test.hash)
		val, exists := newPodTemplate.ObjectMeta.Labels[extensions.DaemonSetTemplateGenerationKey]
		if !exists || val != fmt.Sprint(*test.templateGeneration) {
			t.Errorf("Expected podTemplateSpec to have generation label value: %d, got: %s", *test.templateGeneration, val)
		}
		val, exists = newPodTemplate.ObjectMeta.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
		if test.expectUniqueLabel && (!exists || val != test.hash) {
			t.Errorf("Expected podTemplateSpec to have hash label value: %s, got: %s", test.hash, val)
		}
		if !test.expectUniqueLabel && exists {
			t.Errorf("Expected podTemplateSpec to have no hash label, got: %s", val)
		}
	}
}

func TestReplaceDaemonSetPodNodeNameNodeAffinity(t *testing.T) {
	tests := []struct {
		affinity *v1.Affinity
		hostname string
		expected *v1.Affinity
	}{
		{
			affinity: nil,
			hostname: "host_1",
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      v1.LabelHostname,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
			hostname: "host_1",
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
						{
							Preference: v1.NodeSelectorTerm{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      v1.LabelHostname,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
			hostname: "host_1",
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
						{
							Preference: v1.NodeSelectorTerm{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      v1.LabelHostname,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1", "host_2"},
									},
								},
							},
						},
					},
				},
			},
			hostname: "host_1",
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			affinity: nil,
			hostname: "host_1",
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "hostname",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_2"},
									},
								},
							},
						},
					},
				},
			},
			hostname: "host_1",
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpNotIn,
										Values:   []string{"host_2"},
									},
								},
							},
						},
					},
				},
			},
			hostname: "host_1",
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										// NOTE: Only `metadata.name` is valid key in `MatchFields` in 1.11;
										//       added this case for compatibility: the feature works as normal
										//       when new Keys introduced.
										Key:      "metadata.foo",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"bar"},
									},
								},
							},
						},
					},
				},
			},
			hostname: "host_1",
			expected: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      api.ObjectNameField,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"host_1"},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for i, test := range tests {
		got := ReplaceDaemonSetPodNodeNameNodeAffinity(test.affinity, test.hostname)
		if !reflect.DeepEqual(test.expected, got) {
			t.Errorf("Failed to append NodeAffinity in case %d, got: %v, expected: %v",
				i, got, test.expected)
		}
	}
}

func forEachFeatureGate(t *testing.T, tf func(t *testing.T), gates ...featuregate.Feature) {
	for _, fg := range gates {
		for _, f := range []bool{true, false} {
			func() {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, fg, f)()
				t.Run(fmt.Sprintf("%v (%t)", fg, f), tf)
			}()
		}
	}
}

func TestGetTargetNodeName(t *testing.T) {
	testFun := func(t *testing.T) {
		tests := []struct {
			pod         *v1.Pod
			nodeName    string
			expectedErr bool
		}{
			{
				pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod1",
						Namespace: "default",
					},
					Spec: v1.PodSpec{
						NodeName: "node-1",
					},
				},
				nodeName: "node-1",
			},
			{
				pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod2",
						Namespace: "default",
					},
					Spec: v1.PodSpec{
						Affinity: &v1.Affinity{
							NodeAffinity: &v1.NodeAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
									NodeSelectorTerms: []v1.NodeSelectorTerm{
										{
											MatchFields: []v1.NodeSelectorRequirement{
												{
													Key:      api.ObjectNameField,
													Operator: v1.NodeSelectorOpIn,
													Values:   []string{"node-1"},
												},
											},
										},
									},
								},
							},
						},
					},
				},
				nodeName: "node-1",
			},
			{
				pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod3",
						Namespace: "default",
					},
					Spec: v1.PodSpec{
						Affinity: &v1.Affinity{
							NodeAffinity: &v1.NodeAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
									NodeSelectorTerms: []v1.NodeSelectorTerm{
										{
											MatchFields: []v1.NodeSelectorRequirement{
												{
													Key:      api.ObjectNameField,
													Operator: v1.NodeSelectorOpIn,
													Values:   []string{"node-1", "node-2"},
												},
											},
										},
									},
								},
							},
						},
					},
				},
				expectedErr: true,
			},
			{
				pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod4",
						Namespace: "default",
					},
					Spec: v1.PodSpec{},
				},
				expectedErr: true,
			},
		}

		for _, test := range tests {
			got, err := GetTargetNodeName(test.pod)
			if test.expectedErr != (err != nil) {
				t.Errorf("Unexpected error, expectedErr: %v, err: %v", test.expectedErr, err)
			} else if !test.expectedErr {
				if test.nodeName != got {
					t.Errorf("Failed to get target node name, got: %v, expected: %v", got, test.nodeName)
				}
			}
		}
	}

	forEachFeatureGate(t, testFun)
}
