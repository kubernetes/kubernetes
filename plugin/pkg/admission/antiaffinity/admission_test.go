/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"testing"
)

// ensures the hard PodAntiAffinity is denied if it defines TopologyKey other than kubernetes.io/hostname.
func TestInterPodAffinityAdmission(t *testing.T) {
	handler := NewInterPodAntiAffinity(nil)
	pod := api.Pod{
		Spec: api.PodSpec{},
	}
	tests := []struct {
		affinity      map[string]string
		errorExpected bool
	}{
		// empty affinity its success.
		{
			affinity:      map[string]string{},
			errorExpected: false,
		},
		// what ever topologyKey in preferredDuringSchedulingIgnoredDuringExecution, the admission should success.
		{
			affinity: map[string]string{
				api.AffinityAnnotationKey: `
					{"podAntiAffinity": {
						"preferredDuringSchedulingIgnoredDuringExecution": [{
							"weight": 5,
							"podAffinityTerm": {
								"labelSelector": {
									"matchExpressions": [{
										"key": "security",
										"operator": "In",
										"values":["S2"]
										}]
									},
								"namespaces": [],
								"topologyKey": "az"
							}
						}]
					}}`,
			},
			errorExpected: false,
		},
		// valid topologyKey in requiredDuringSchedulingIgnoredDuringExecution,
		// plus any topologyKey in preferredDuringSchedulingIgnoredDuringExecution, then admission success.
		{
			affinity: map[string]string{
				api.AffinityAnnotationKey: `
					{"podAntiAffinity": {
						"preferredDuringSchedulingIgnoredDuringExecution": [{
							"weight": 5,
							"podAffinityTerm": {
								"labelSelector": {
									"matchExpressions": [{
										"key": "security",
										"operator": "In",
										"values":["S2"]
									}]
								},
								"namespaces": [],
								"topologyKey": "az"
							}
						}],
						"requiredDuringSchedulingIgnoredDuringExecution": [{
							"labelSelector": {
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values":["S2"]
								}]
							},
							"namespaces": [],
							"topologyKey": "` + unversioned.LabelHostname + `"
						}]
					}}`,
			},
			errorExpected: false,
		},
		// valid topologyKey in requiredDuringSchedulingIgnoredDuringExecution then admission success.
		{
			affinity: map[string]string{
				api.AffinityAnnotationKey: `
					{"podAntiAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": [{
							"labelSelector": {
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values":["S2"]
								}]
							},
							"namespaces":[],
							"topologyKey": "` + unversioned.LabelHostname + `"
						}]
					}}`,
			},
			errorExpected: false,
		},
		// invalid topologyKey in requiredDuringSchedulingIgnoredDuringExecution then admission fails.
		{
			affinity: map[string]string{
				api.AffinityAnnotationKey: `
					{"podAntiAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": [{
							"labelSelector": {
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values":["S2"]
								}]
							},
							"namespaces":[],
							"topologyKey": " zone "
						}]
					}}`,
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
			affinity: map[string]string{
				api.AffinityAnnotationKey: `
					{"podAntiAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": [{
							"labelSelector": {
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values":["S2"]
								}]
							},
							"namespaces":[],
							"topologyKey": "` + unversioned.LabelHostname + `"
						},
						{
							"labelSelector": {
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values":["S2"]
								}]
							},
							"namespaces":[],
							"topologyKey": " zone "
						},
						{
							"labelSelector": {
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values":["S2"]
								}]
							},
							"namespaces": [],
							"topologyKey": "` + unversioned.LabelHostname + `"
						}]
					}}`,
			},
			errorExpected: true,
		},
	}
	for _, test := range tests {
		pod.ObjectMeta.Annotations = test.affinity
		err := handler.Admit(admission.NewAttributesRecord(&pod, api.Kind("Pod").WithVersion("version"), "foo", "name", api.Resource("pods").WithVersion("version"), "", "ignored", nil))

		if test.errorExpected && err == nil {
			t.Errorf("Expected error for Anti Affinity %+v but did not get an error", test.affinity)
		}

		if !test.errorExpected && err != nil {
			t.Errorf("Unexpected error %v for AntiAffinity %+v", err, test.affinity)
		}
	}
}
func TestHandles(t *testing.T) {
	handler := NewInterPodAntiAffinity(nil)
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
