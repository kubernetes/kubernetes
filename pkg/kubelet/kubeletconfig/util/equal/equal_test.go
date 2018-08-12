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

package equal

import (
	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"testing"
)

func TestConfigSourceEq(t *testing.T) {
	nodeConfigSrc := &apiv1.NodeConfigSource{
		ConfigMapRef: &apiv1.ObjectReference{
			UID:       types.UID("UID-1111"),
			Name:      "ref1",
			Namespace: "default",
		},
	}
	testcases := []struct {
		name      string
		inputObjA *apiv1.NodeConfigSource
		inputObjB *apiv1.NodeConfigSource
		expected  bool
	}{
		{
			name:      "same pointer",
			inputObjA: nodeConfigSrc,
			inputObjB: nodeConfigSrc,
			expected:  true,
		},
		{
			name:      "nil pointer",
			inputObjA: nil,
			inputObjB: nodeConfigSrc,
			expected:  false,
		},
		{
			name: "not eqaul UID",
			inputObjA: &apiv1.NodeConfigSource{
				ConfigMapRef: &apiv1.ObjectReference{
					UID:       types.UID("UIDA-2222"),
					Name:      "ref2",
					Namespace: "default",
				},
			},
			inputObjB: &apiv1.NodeConfigSource{
				ConfigMapRef: &apiv1.ObjectReference{
					UID:       types.UID("UIDB-2222"),
					Name:      "ref2",
					Namespace: "default",
				},
			},
			expected: false,
		},
		{
			name: "nil ConfigMapRef",
			inputObjA: &apiv1.NodeConfigSource{
				ConfigMapRef: nil,
			},
			inputObjB: &apiv1.NodeConfigSource{
				ConfigMapRef: &apiv1.ObjectReference{
					UID:       types.UID("UIDB-2222"),
					Name:      "ref2",
					Namespace: "default",
				},
			},
			expected: false,
		},
		{
			name: "all equal",
			inputObjA: &apiv1.NodeConfigSource{
				ConfigMapRef: &apiv1.ObjectReference{
					UID:       types.UID("UID-equal"),
					Name:      "ref2",
					Namespace: "default",
				},
			},
			inputObjB: &apiv1.NodeConfigSource{
				ConfigMapRef: &apiv1.ObjectReference{
					UID:       types.UID("UID-equal"),
					Name:      "ref2",
					Namespace: "default",
				},
			},
			expected: true,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			result := ConfigSourceEq(testcase.inputObjA, testcase.inputObjB)
			if testcase.expected != result {
				t.Errorf("unexpected result, expected: %v, actual: %v", testcase.expected, result)
			}
		})
	}
}

func TestConfigOKEq(t *testing.T) {
	nodeCondition := &apiv1.NodeCondition{
		Status:  apiv1.ConditionFalse,
		Reason:  "timeout updating",
		Message: "timeout updating node status to apiserver",
	}

	testcases := []struct {
		name      string
		inputObjA *apiv1.NodeCondition
		inputObjB *apiv1.NodeCondition
		expected  bool
	}{
		{
			name:      "same pointer",
			inputObjA: nodeCondition,
			inputObjB: nodeCondition,
			expected:  true,
		},
		{
			name: "not eqaul Status",
			inputObjA: &apiv1.NodeCondition{
				Status:  apiv1.ConditionFalse,
				Reason:  "timeout updating",
				Message: "timeout updating node status to apiserver",
			},
			inputObjB: &apiv1.NodeCondition{
				Status:  apiv1.ConditionTrue,
				Reason:  "timeout updating",
				Message: "timeout updating node status to apiserver",
			},
		},
		{
			name: "not eqaul Reason",
			inputObjA: &apiv1.NodeCondition{
				Status:  apiv1.ConditionUnknown,
				Reason:  "timeout updating",
				Message: "timeout updating node status to apiserver",
			},
			inputObjB: &apiv1.NodeCondition{
				Status:  apiv1.ConditionUnknown,
				Reason:  "timeout connnecting to apiserver",
				Message: "timeout updating node status to apiserver",
			},
		},
		{
			name: "not eqaul Message",
			inputObjA: &apiv1.NodeCondition{
				Status:  apiv1.ConditionFalse,
				Reason:  "timeout updating",
				Message: "timeout updating node status to apiserver",
			},
			inputObjB: &apiv1.NodeCondition{
				Status:  apiv1.ConditionFalse,
				Reason:  "timeout updating",
				Message: "timeout connecting to apiserver",
			},
		},
		{
			name: "all equal",
			inputObjA: &apiv1.NodeCondition{
				Status:  apiv1.ConditionTrue,
				Reason:  "successfully update status",
				Message: "successfully update status to apiserver",
			},
			inputObjB: &apiv1.NodeCondition{
				Status:  apiv1.ConditionTrue,
				Reason:  "successfully update status",
				Message: "successfully update status to apiserver",
			},
			expected: true,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			result := ConfigOKEq(testcase.inputObjA, testcase.inputObjB)
			if testcase.expected != result {
				t.Errorf("unexpected result, expected: %v, actual: %v", testcase.expected, result)
			}
		})
	}
}
