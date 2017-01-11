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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/storage/util"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
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
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})
	validClaimGoldStorageClass := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
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
	validClaimGoldStorageClass.Annotations = map[string]string{
		util.StorageClassAnnotation: "gold",
	}

	validClaimBronzeStorageClass := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
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
	validClaimBronzeStorageClass.Annotations = map[string]string{
		util.StorageClassAnnotation: "bronze",
	}

	missingStorage := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
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
			Requests: api.ResourceList{},
		},
	})

	missingGoldStorage := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
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
			Requests: api.ResourceList{},
		},
	})
	missingGoldStorage.Annotations = map[string]string{
		util.StorageClassAnnotation: "gold",
	}

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
		"missing gold storage": {
			pvc:      missingGoldStorage,
			required: []api.ResourceName{ResourceByStorageClass("gold", api.ResourceRequestsStorage)},
			err:      `must specify gold.storageclass.storage.k8s.io/requests.storage`,
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
		"valid-claim-gold-quota-gold": {
			pvc: validClaimGoldStorageClass,
			required: []api.ResourceName{
				api.ResourceRequestsStorage,
				api.ResourcePersistentVolumeClaims,
				ResourceByStorageClass("gold", api.ResourceRequestsStorage),
				ResourceByStorageClass("gold", api.ResourcePersistentVolumeClaims),
			},
		},
		"valid-claim-bronze-with-quota-gold": {
			pvc: validClaimBronzeStorageClass,
			required: []api.ResourceName{
				api.ResourceRequestsStorage,
				api.ResourcePersistentVolumeClaims,
				ResourceByStorageClass("gold", api.ResourceRequestsStorage),
				ResourceByStorageClass("gold", api.ResourcePersistentVolumeClaims),
			},
		},
	}

	kubeClient := fake.NewSimpleClientset()
	evaluator := NewPersistentVolumeClaimEvaluator(kubeClient, nil)
	for testName, test := range testCases {
		err := evaluator.Constraints(test.required, test.pvc)
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
	})
	storageClassName := "gold"
	validClaimByStorageClass.Annotations = map[string]string{
		util.StorageClassAnnotation: storageClassName,
	}

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
		"pvc-usage-by-class": {
			pvc: validClaimByStorageClass,
			usage: api.ResourceList{
				api.ResourceRequestsStorage:                                                  resource.MustParse("10Gi"),
				api.ResourcePersistentVolumeClaims:                                           resource.MustParse("1"),
				ResourceByStorageClass(storageClassName, api.ResourceRequestsStorage):        resource.MustParse("10Gi"),
				ResourceByStorageClass(storageClassName, api.ResourcePersistentVolumeClaims): resource.MustParse("1"),
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
