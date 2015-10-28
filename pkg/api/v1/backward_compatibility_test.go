/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testing/compat"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

func TestCompatibility_v1_PodSecurityContext(t *testing.T) {
	cases := []struct {
		name         string
		input        string
		expectedKeys map[string]string
		absentKeys   []string
	}{
		{
			name: "reseting defaults for pre-v1.1 mirror pods",
			input: `
{
	"kind":"Pod",
	"apiVersion":"v1",
	"metadata":{
		"name":"my-pod-name",
		"namespace":"my-pod-namespace",
		"annotations": {
			"kubernetes.io/config.mirror": "mirror"
		}
	},
	"spec": {
		"containers":[{
			"name":"a",
			"image":"my-container-image",
			"resources": {
				"limits": {
					"cpu": "100m"
				}
			}
		}]
	}
}
`,
			absentKeys: []string{
				"spec.terminationGracePeriodSeconds",
				"spec.containers[0].resources.requests",
			},
		},
		{
			name: "preserving defaults for v1.1+ mirror pods",
			input: `
		{
			"kind":"Pod",
			"apiVersion":"v1",
			"metadata":{
				"name":"my-pod-name",
				"namespace":"my-pod-namespace",
				"annotations": {
					"kubernetes.io/config.mirror": "cbe924f710c7e26f7693d6a341bcfad0"
				}
			},
			"spec": {
				"containers":[{
					"name":"a",
					"image":"my-container-image",
					"resources": {
						"limits": {
							"cpu": "100m"
						}
					}
				}]
			}
		}
		`,
			expectedKeys: map[string]string{
				"spec.terminationGracePeriodSeconds":    "30",
				"spec.containers[0].resources.requests": "map[cpu:100m]",
			},
		},
	}

	validator := func(obj runtime.Object) fielderrors.ValidationErrorList {
		return validation.ValidatePodSpec(&(obj.(*api.Pod).Spec))
	}

	for _, tc := range cases {
		t.Logf("Testing 1.0.0 backward compatibility for %v", tc.name)
		compat.TestCompatibility(t, "v1", []byte(tc.input), validator, tc.expectedKeys, tc.absentKeys)
	}
}
