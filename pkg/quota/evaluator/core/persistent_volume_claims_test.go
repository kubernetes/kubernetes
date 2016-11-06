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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/quota"
)

func testVolumeClaim(name string, namespace string, spec api.PersistentVolumeClaimSpec) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
	}
}

func TestPersistentVolumeClaimsConstraintsFunc(t *testing.T) {
	validClaim := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		Selector: &unversioned.LabelSelector{
			MatchExpressions: []unversioned.LabelSelectorRequirement{
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
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})
	missingStorage := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		Selector: &unversioned.LabelSelector{
			MatchExpressions: []unversioned.LabelSelectorRequirement{
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
			Requests: api.ResourceList{},
		},
	})

	testCases := map[string]struct {
		pvc      *api.PersistentVolumeClaim
		required []api.ResourceName
		err      string
	}{
		"missing storage": {
			pvc:      missingStorage,
			required: []api.ResourceName{api.ResourceRequestsStorage},
			err:      `must specify requests.storage`,
		},
		"valid-claim-quota-storage": {
			pvc:      validClaim,
			required: []api.ResourceName{api.ResourceRequestsStorage},
		},
		"valid-claim-quota-pvc": {
			pvc:      validClaim,
			required: []api.ResourceName{api.ResourcePersistentVolumeClaims},
		},
		"valid-claim-quota-storage-and-pvc": {
			pvc:      validClaim,
			required: []api.ResourceName{api.ResourceRequestsStorage, api.ResourcePersistentVolumeClaims},
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
	validClaim := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		Selector: &unversioned.LabelSelector{
			MatchExpressions: []unversioned.LabelSelectorRequirement{
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

	kubeClient := fake.NewSimpleClientset()
	evaluator := NewPersistentVolumeClaimEvaluator(kubeClient, nil)
	testCases := map[string]struct {
		pvc   *api.PersistentVolumeClaim
		usage api.ResourceList
	}{
		"pvc-usage": {
			pvc: validClaim,
			usage: api.ResourceList{
				api.ResourceRequestsStorage:        resource.MustParse("10Gi"),
				api.ResourcePersistentVolumeClaims: resource.MustParse("1"),
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
