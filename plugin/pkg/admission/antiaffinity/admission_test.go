/*
Copyright 2016 The Kubernetes Authors.

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

package antiaffinity

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

// ensures the hard PodAntiAffinity is denied if it defines TopologyKey other than kubernetes.io/hostname.
func TestInterPodAffinityAdmission(t *testing.T) {
	handler := NewInterPodAntiAffinity()
	pod := api.Pod{
		Spec: api.PodSpec{},
	}
	tests := []struct {
		affinity      *api.Affinity
		errorExpected bool
	}{
		// empty affinity its success.
		{
			affinity:      &api.Affinity{},
			errorExpected: false,
		},
		// what ever topologyKey in preferredDuringSchedulingIgnoredDuringExecution, the admission should success.
		{
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{
							Weight: 5,
							PodAffinityTerm: api.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "security",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"S2"},
										},
									},
								},
								TopologyKey: "az",
							},
						},
					},
				},
			},
			errorExpected: false,
		},
		// valid topologyKey in requiredDuringSchedulingIgnoredDuringExecution,
		// plus any topologyKey in preferredDuringSchedulingIgnoredDuringExecution, then admission success.
		{
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{
							Weight: 5,
							PodAffinityTerm: api.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "security",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"S2"},
										},
									},
								},
								TopologyKey: "az",
							},
						},
					},
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S2"},
									},
								},
							},
							TopologyKey: metav1.LabelHostname,
						},
					},
				},
			},
			errorExpected: false,
		},
		// valid topologyKey in requiredDuringSchedulingIgnoredDuringExecution then admission success.
		{
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S2"},
									},
								},
							},
							TopologyKey: metav1.LabelHostname,
						},
					},
				},
			},
			errorExpected: false,
		},
		// invalid topologyKey in requiredDuringSchedulingIgnoredDuringExecution then admission fails.
		{
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S2"},
									},
								},
							},
							TopologyKey: " zone ",
						},
					},
				},
			},
			errorExpected: true,
		},
		// invalid topologyKey in requiredDuringSchedulingRequiredDuringExecution then admission fails.
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		// {
		//         affinity: map[string]string{
		//			api.AffinityAnnotationKey: `
		//				{"podAntiAffinity": {
		//					"requiredDuringSchedulingRequiredDuringExecution": [{
		//						"labelSelector": {
		//							"matchExpressions": [{
		//								"key": "security",
		//								"operator": "In",
		//								"values":["S2"]
		//							}]
		//						},
		//						"namespaces":[],
		//						"topologyKey": " zone "
		//					}]
		//				}}`,
		//			},
		//			errorExpected: true,
		//  }
		// list of requiredDuringSchedulingIgnoredDuringExecution middle element topologyKey is not valid.
		{
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S2"},
									},
								},
							},
							TopologyKey: metav1.LabelHostname,
						}, {
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S2"},
									},
								},
							},
							TopologyKey: " zone ",
						}, {
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S2"},
									},
								},
							},
							TopologyKey: metav1.LabelHostname,
						},
					},
				},
			},
			errorExpected: true,
		},
	}
	for _, test := range tests {
		pod.Spec.Affinity = test.affinity
		err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), "foo", "name", api.Resource("pods").WithVersion("version"), "", "ignored", nil))

		if test.errorExpected && err == nil {
			t.Errorf("Expected error for Anti Affinity %+v but did not get an error", test.affinity)
		}

		if !test.errorExpected && err != nil {
			t.Errorf("Unexpected error %v for AntiAffinity %+v", err, test.affinity)
		}
	}
}
func TestHandles(t *testing.T) {
	handler := NewInterPodAntiAffinity()
	tests := map[admission.Operation]bool{
		admission.Update:  true,
		admission.Create:  true,
		admission.Delete:  false,
		admission.Connect: false,
	}
	for op, expected := range tests {
		result := handler.Handles(op)
		if result != expected {
			t.Errorf("Unexpected result for operation %s: %v\n", op, result)
		}
	}
}

// TestOtherResources ensures that this admission controller is a no-op for other resources,
// subresources, and non-pods.
func TestOtherResources(t *testing.T) {
	namespace := "testnamespace"
	name := "testname"
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	tests := []struct {
		name        string
		kind        string
		resource    string
		subresource string
		object      runtime.Object
		expectError bool
	}{
		{
			name:     "non-pod resource",
			kind:     "Foo",
			resource: "foos",
			object:   pod,
		},
		{
			name:        "pod subresource",
			kind:        "Pod",
			resource:    "pods",
			subresource: "eviction",
			object:      pod,
		},
		{
			name:        "non-pod object",
			kind:        "Pod",
			resource:    "pods",
			object:      &api.Service{},
			expectError: true,
		},
	}

	for _, tc := range tests {
		handler := &plugin{}

		err := handler.Admit(admission.NewAttributesRecord(tc.object, nil, api.Kind(tc.kind).WithVersion("version"), namespace, name, api.Resource(tc.resource).WithVersion("version"), tc.subresource, admission.Create, nil))

		if tc.expectError {
			if err == nil {
				t.Errorf("%s: unexpected nil error", tc.name)
			}
			continue
		}

		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}
	}
}
