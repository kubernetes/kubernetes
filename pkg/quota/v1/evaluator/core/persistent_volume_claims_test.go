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

package core

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	api "k8s.io/kubernetes/pkg/apis/core"
	quota "k8s.io/kubernetes/pkg/quota/v1"
	"k8s.io/kubernetes/pkg/quota/v1/generic"
)

func testVolumeClaim(name string, namespace string, spec api.PersistentVolumeClaimSpec) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
	}
}

func TestPersistentVolumeClaimEvaluatorUsage(t *testing.T) {
	classGold := "gold"
	validClaim := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		Selector: &metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      "key2",
					Operator: "Exists",
				},
			},
		},
		AccessModes: []api.PersistentVolumeAccessMode{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
		},
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10Gi"),
			},
		},
	})
	validClaimByStorageClass := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		Selector: &metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      "key2",
					Operator: "Exists",
				},
			},
		},
		AccessModes: []api.PersistentVolumeAccessMode{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
		},
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10Gi"),
			},
		},
		StorageClassName: &classGold,
	})

	evaluator := NewPersistentVolumeClaimEvaluator(nil)
	testCases := map[string]struct {
		pvc   *api.PersistentVolumeClaim
		usage corev1.ResourceList
	}{
		"pvc-usage": {
			pvc: validClaim,
			usage: corev1.ResourceList{
				corev1.ResourceRequestsStorage:        resource.MustParse("10Gi"),
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "persistentvolumeclaims"}): resource.MustParse("1"),
			},
		},
		"pvc-usage-by-class": {
			pvc: validClaimByStorageClass,
			usage: corev1.ResourceList{
				corev1.ResourceRequestsStorage:                                                                    resource.MustParse("10Gi"),
				corev1.ResourcePersistentVolumeClaims:                                                             resource.MustParse("1"),
				V1ResourceByStorageClass(classGold, corev1.ResourceRequestsStorage):                               resource.MustParse("10Gi"),
				V1ResourceByStorageClass(classGold, corev1.ResourcePersistentVolumeClaims):                        resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "persistentvolumeclaims"}): resource.MustParse("1"),
			},
		},
	}
	for testName, testCase := range testCases {
		actual, err := evaluator.Usage(testCase.pvc)
		if err != nil {
			t.Errorf("%s unexpected error: %v", testName, err)
		}
		if !quota.Equals(testCase.usage, actual) {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.usage, actual)
		}
	}
}
