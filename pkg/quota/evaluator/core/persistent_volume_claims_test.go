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

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/quota"
)

func testVolumeClaim(name string, namespace string, spec v1.PersistentVolumeClaimSpec) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: v1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
	}
}

func TestPersistentVolumeClaimsConstraintsFunc(t *testing.T) {
	validClaim := testVolumeClaim("foo", "ns", v1.PersistentVolumeClaimSpec{
		Selector: &unversioned.LabelSelector{
			MatchExpressions: []unversioned.LabelSelectorRequirement{
				{
					Key:      "key2",
					Operator: "Exists",
				},
			},
		},
		AccessModes: []v1.PersistentVolumeAccessMode{
			v1.ReadWriteOnce,
			v1.ReadOnlyMany,
		},
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})
	missingStorage := testVolumeClaim("foo", "ns", v1.PersistentVolumeClaimSpec{
		Selector: &unversioned.LabelSelector{
			MatchExpressions: []unversioned.LabelSelectorRequirement{
				{
					Key:      "key2",
					Operator: "Exists",
				},
			},
		},
		AccessModes: []v1.PersistentVolumeAccessMode{
			v1.ReadWriteOnce,
			v1.ReadOnlyMany,
		},
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{},
		},
	})

	testCases := map[string]struct {
		pvc      *v1.PersistentVolumeClaim
		required []v1.ResourceName
		err      string
	}{
		"missing storage": {
			pvc:      missingStorage,
			required: []v1.ResourceName{v1.ResourceRequestsStorage},
			err:      `must specify requests.storage`,
		},
		"valid-claim-quota-storage": {
			pvc:      validClaim,
			required: []v1.ResourceName{v1.ResourceRequestsStorage},
		},
		"valid-claim-quota-pvc": {
			pvc:      validClaim,
			required: []v1.ResourceName{v1.ResourcePersistentVolumeClaims},
		},
		"valid-claim-quota-storage-and-pvc": {
			pvc:      validClaim,
			required: []v1.ResourceName{v1.ResourceRequestsStorage, v1.ResourcePersistentVolumeClaims},
		},
	}
	for testName, test := range testCases {
		err := PersistentVolumeClaimConstraintsFunc(test.required, test.pvc)
		switch {
		case err != nil && len(test.err) == 0,
			err == nil && len(test.err) != 0,
			err != nil && test.err != err.Error():
			t.Errorf("%s unexpected error: %v", testName, err)
		}
	}
}

func TestPersistentVolumeClaimEvaluatorUsage(t *testing.T) {
	validClaim := testVolumeClaim("foo", "ns", v1.PersistentVolumeClaimSpec{
		Selector: &unversioned.LabelSelector{
			MatchExpressions: []unversioned.LabelSelectorRequirement{
				{
					Key:      "key2",
					Operator: "Exists",
				},
			},
		},
		AccessModes: []v1.PersistentVolumeAccessMode{
			v1.ReadWriteOnce,
			v1.ReadOnlyMany,
		},
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10Gi"),
			},
		},
	})

	kubeClient := fake.NewSimpleClientset()
	evaluator := NewPersistentVolumeClaimEvaluator(kubeClient)
	testCases := map[string]struct {
		pvc   *v1.PersistentVolumeClaim
		usage v1.ResourceList
	}{
		"pvc-usage": {
			pvc: validClaim,
			usage: v1.ResourceList{
				v1.ResourceRequestsStorage:        resource.MustParse("10Gi"),
				v1.ResourcePersistentVolumeClaims: resource.MustParse("1"),
			},
		},
	}
	for testName, testCase := range testCases {
		actual := evaluator.Usage(testCase.pvc)
		if !quota.Equals(testCase.usage, actual) {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.usage, actual)
		}
	}
}
