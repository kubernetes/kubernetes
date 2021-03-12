/*
Copyright 2020 The Kubernetes Authors.

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

package resourcequota

import (
	"context"
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	v1 "k8s.io/apiserver/pkg/admission/plugin/resourcequota/apis/resourcequota/v1"
)

func TestPrettyPrint(t *testing.T) {
	toResourceList := func(resources map[corev1.ResourceName]string) corev1.ResourceList {
		resourceList := corev1.ResourceList{}
		for key, value := range resources {
			resourceList[key] = resource.MustParse(value)
		}
		return resourceList
	}
	testCases := []struct {
		input    corev1.ResourceList
		expected string
	}{
		{
			input: toResourceList(map[corev1.ResourceName]string{
				corev1.ResourceCPU: "100m",
			}),
			expected: "cpu=100m",
		},
		{
			input: toResourceList(map[corev1.ResourceName]string{
				corev1.ResourcePods:                   "10",
				corev1.ResourceServices:               "10",
				corev1.ResourceReplicationControllers: "10",
				corev1.ResourceServicesNodePorts:      "10",
				corev1.ResourceRequestsCPU:            "100m",
				corev1.ResourceRequestsMemory:         "100Mi",
				corev1.ResourceLimitsCPU:              "100m",
				corev1.ResourceLimitsMemory:           "100Mi",
			}),
			expected: "limits.cpu=100m,limits.memory=100Mi,pods=10,replicationcontrollers=10,requests.cpu=100m,requests.memory=100Mi,services=10,services.nodeports=10",
		},
	}
	for i, testCase := range testCases {
		result := prettyPrint(testCase.input)
		if result != testCase.expected {
			t.Errorf("Pretty print did not give stable sorted output[%d], expected %v, but got %v", i, testCase.expected, result)
		}
	}
}

func TestHasUsageStats(t *testing.T) {
	testCases := map[string]struct {
		a        corev1.ResourceQuota
		relevant []corev1.ResourceName
		expected bool
	}{
		"empty": {
			a:        corev1.ResourceQuota{Status: corev1.ResourceQuotaStatus{Hard: corev1.ResourceList{}}},
			relevant: []corev1.ResourceName{corev1.ResourceMemory},
			expected: true,
		},
		"hard-only": {
			a: corev1.ResourceQuota{
				Status: corev1.ResourceQuotaStatus{
					Hard: corev1.ResourceList{
						corev1.ResourceMemory: resource.MustParse("1Gi"),
					},
					Used: corev1.ResourceList{},
				},
			},
			relevant: []corev1.ResourceName{corev1.ResourceMemory},
			expected: false,
		},
		"hard-used": {
			a: corev1.ResourceQuota{
				Status: corev1.ResourceQuotaStatus{
					Hard: corev1.ResourceList{
						corev1.ResourceMemory: resource.MustParse("1Gi"),
					},
					Used: corev1.ResourceList{
						corev1.ResourceMemory: resource.MustParse("500Mi"),
					},
				},
			},
			relevant: []corev1.ResourceName{corev1.ResourceMemory},
			expected: true,
		},
		"hard-used-relevant": {
			a: corev1.ResourceQuota{
				Status: corev1.ResourceQuotaStatus{
					Hard: corev1.ResourceList{
						corev1.ResourceMemory: resource.MustParse("1Gi"),
						corev1.ResourcePods:   resource.MustParse("1"),
					},
					Used: corev1.ResourceList{
						corev1.ResourceMemory: resource.MustParse("500Mi"),
					},
				},
			},
			relevant: []corev1.ResourceName{corev1.ResourceMemory},
			expected: true,
		},
	}
	for testName, testCase := range testCases {
		if result := hasUsageStats(&testCase.a, testCase.relevant); result != testCase.expected {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, result)
		}
	}
}

type fakeEvaluator struct{}

func (fakeEvaluator) Evaluate(a admission.Attributes) error {
	return errors.New("should not be called")
}

func TestExcludedOperations(t *testing.T) {
	a := &QuotaAdmission{
		evaluator: fakeEvaluator{},
	}
	testCases := []struct {
		desc string
		attr admission.Attributes
	}{
		{
			"subresource",
			admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "namespace", "name", schema.GroupVersionResource{}, "subresource", admission.Create, nil, false, nil),
		},
		{
			"non-namespaced resource",
			admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "", "namespace", schema.GroupVersionResource{}, "", admission.Create, nil, false, nil),
		},
		{
			"namespace creation",
			admission.NewAttributesRecord(nil, nil, v1.SchemeGroupVersion.WithKind("Namespace"), "namespace", "namespace", schema.GroupVersionResource{}, "", admission.Create, nil, false, nil),
		},
	}
	for _, test := range testCases {
		if err := a.Validate(context.TODO(), test.attr, nil); err != nil {
			t.Errorf("Test case: %q. Expected no error but got: %v", test.desc, err)
		}
	}
}
