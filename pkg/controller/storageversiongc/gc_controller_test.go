/*
Copyright 2022 The Kubernetes Authors.

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

package storageversiongc

import (
	"context"
	"reflect"
	"testing"
	"time"

	apiserverinternalv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	coordinationv1 "k8s.io/api/coordination/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	utilpointer "k8s.io/utils/pointer"
)

func setupController(clientset kubernetes.Interface) {
	informerFactory := informers.NewSharedInformerFactory(clientset, 100*time.Millisecond)
	leaseInformer := informerFactory.Coordination().V1().Leases()
	storageVersionInformer := informerFactory.Internal().V1alpha1().StorageVersions()

	controller := NewStorageVersionGC(clientset, leaseInformer, storageVersionInformer)
	go controller.Run(context.Background())
	informerFactory.Start(nil)
}

func Test_StorageVersionUpdatedWithAllEncodingVersionsEqualOnLeaseDeletion(t *testing.T) {
	lease1 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-1",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-1"),
		},
	}
	lease2 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-2",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-2"),
		},
	}
	lease3 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-3",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-3"),
		},
	}

	storageVersion := &apiserverinternalv1alpha1.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name: "k8s.test.resources",
		},
		Status: apiserverinternalv1alpha1.StorageVersionStatus{
			StorageVersions: []apiserverinternalv1alpha1.ServerStorageVersion{
				{
					APIServerID:       "kube-apiserver-1",
					EncodingVersion:   "v1",
					DecodableVersions: []string{"v1"},
				},
				{
					APIServerID:       "kube-apiserver-2",
					EncodingVersion:   "v2",
					DecodableVersions: []string{"v2"},
				},
				{
					APIServerID:       "kube-apiserver-3",
					EncodingVersion:   "v2",
					DecodableVersions: []string{"v2"},
				},
			},
			CommonEncodingVersion: utilpointer.String("v1"),
		},
	}

	clientset := fake.NewSimpleClientset(lease1, lease2, lease3, storageVersion)
	setupController(clientset)

	// Delete the lease object and verify that storage version status is updated
	if err := clientset.CoordinationV1().Leases(metav1.NamespaceSystem).Delete(context.Background(), "kube-apiserver-1", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("error deleting lease object: %v", err)
	}

	// add a delay to ensure controller had a chance to reconcile
	time.Sleep(2 * time.Second)

	storageVersion, err := clientset.InternalV1alpha1().StorageVersions().Get(context.Background(), "k8s.test.resources", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting StorageVersion: %v", err)
	}

	expectedServerStorageVersions := []apiserverinternalv1alpha1.ServerStorageVersion{
		{
			APIServerID:       "kube-apiserver-2",
			EncodingVersion:   "v2",
			DecodableVersions: []string{"v2"},
		},
		{
			APIServerID:       "kube-apiserver-3",
			EncodingVersion:   "v2",
			DecodableVersions: []string{"v2"},
		},
	}

	if !reflect.DeepEqual(storageVersion.Status.StorageVersions, expectedServerStorageVersions) {
		t.Error("unexpected storage version object")
		t.Logf("got: %+v", storageVersion)
		t.Logf("expected: %+v", expectedServerStorageVersions)
	}

	if *storageVersion.Status.CommonEncodingVersion != "v2" {
		t.Errorf("unexpected common encoding version")
		t.Logf("got: %q", *storageVersion.Status.CommonEncodingVersion)
		t.Logf("expected: %q", "v2")
	}

	if len(storageVersion.Status.Conditions) != 1 {
		t.Errorf("expected 1 condition, got: %d", len(storageVersion.Status.Conditions))
	}

	if storageVersion.Status.Conditions[0].Type != apiserverinternalv1alpha1.AllEncodingVersionsEqual {
		t.Errorf("expected condition type 'AllEncodingVersionsEqual', got: %q", storageVersion.Status.Conditions[0].Type)
	}

	if storageVersion.Status.Conditions[0].Status != apiserverinternalv1alpha1.ConditionTrue {
		t.Errorf("expected condition status 'True', got: %q", storageVersion.Status.Conditions[0].Status)
	}
}

func Test_StorageVersionUpdatedWithDifferentEncodingVersionsOnLeaseDeletion(t *testing.T) {
	lease1 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-1",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-1"),
		},
	}
	lease2 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-2",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-2"),
		},
	}
	lease3 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-3",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-3"),
		},
	}

	storageVersion := &apiserverinternalv1alpha1.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name: "k8s.test.resources",
		},
		Status: apiserverinternalv1alpha1.StorageVersionStatus{
			StorageVersions: []apiserverinternalv1alpha1.ServerStorageVersion{
				{
					APIServerID:       "kube-apiserver-1",
					EncodingVersion:   "v1",
					DecodableVersions: []string{"v1"},
				},
				{
					APIServerID:       "kube-apiserver-3",
					EncodingVersion:   "v2",
					DecodableVersions: []string{"v2"},
				},
			},
			CommonEncodingVersion: utilpointer.String("v1"),
		},
	}

	clientset := fake.NewSimpleClientset(lease1, lease2, lease3, storageVersion)
	setupController(clientset)

	// Delete the lease object and verify that storage version status is updated
	if err := clientset.CoordinationV1().Leases(metav1.NamespaceSystem).Delete(context.Background(), "kube-apiserver-2", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("error deleting lease object: %v", err)
	}

	// add a delay to ensure controller had a chance to reconcile
	time.Sleep(2 * time.Second)

	storageVersion, err := clientset.InternalV1alpha1().StorageVersions().Get(context.Background(), "k8s.test.resources", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting StorageVersion: %v", err)
	}

	expectedServerStorageVersions := []apiserverinternalv1alpha1.ServerStorageVersion{
		{
			APIServerID:       "kube-apiserver-1",
			EncodingVersion:   "v1",
			DecodableVersions: []string{"v1"},
		},
		{
			APIServerID:       "kube-apiserver-3",
			EncodingVersion:   "v2",
			DecodableVersions: []string{"v2"},
		},
	}

	if !reflect.DeepEqual(storageVersion.Status.StorageVersions, expectedServerStorageVersions) {
		t.Error("unexpected storage version object")
		t.Logf("got: %+v", storageVersion)
		t.Logf("expected: %+v", expectedServerStorageVersions)
	}

	if *storageVersion.Status.CommonEncodingVersion != "v1" {
		t.Errorf("unexpected common encoding version")
		t.Logf("got: %q", *storageVersion.Status.CommonEncodingVersion)
		t.Logf("expected: %q", "v1")
	}
}

func Test_StorageVersionContainsInvalidLeaseID(t *testing.T) {
	lease1 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-1",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-1"),
		},
	}
	lease2 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-2",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-2"),
		},
	}
	lease3 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-3",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-3"),
		},
	}

	storageVersion := &apiserverinternalv1alpha1.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name: "k8s.test.resources",
		},
		Status: apiserverinternalv1alpha1.StorageVersionStatus{
			StorageVersions: []apiserverinternalv1alpha1.ServerStorageVersion{
				{
					APIServerID:       "kube-apiserver-1",
					EncodingVersion:   "v1",
					DecodableVersions: []string{"v1"},
				},
				{
					APIServerID:       "kube-apiserver-2",
					EncodingVersion:   "v2",
					DecodableVersions: []string{"v2"},
				},
				{
					APIServerID:       "kube-apiserver-3",
					EncodingVersion:   "v2",
					DecodableVersions: []string{"v2"},
				},
				{
					APIServerID:       "kube-apiserver-4", // doesn't exist
					EncodingVersion:   "v2",
					DecodableVersions: []string{"v1"},
				},
			},
			CommonEncodingVersion: utilpointer.String("v1"),
		},
	}

	clientset := fake.NewSimpleClientset(lease1, lease2, lease3, storageVersion)
	setupController(clientset)

	// add a delay to ensure controller had a chance to reconcile
	time.Sleep(2 * time.Second)

	storageVersion, err := clientset.InternalV1alpha1().StorageVersions().Get(context.Background(), "k8s.test.resources", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting StorageVersion: %v", err)
	}

	expectedServerStorageVersions := []apiserverinternalv1alpha1.ServerStorageVersion{
		{
			APIServerID:       "kube-apiserver-1",
			EncodingVersion:   "v1",
			DecodableVersions: []string{"v1"},
		},
		{
			APIServerID:       "kube-apiserver-2",
			EncodingVersion:   "v2",
			DecodableVersions: []string{"v2"},
		},
		{
			APIServerID:       "kube-apiserver-3",
			EncodingVersion:   "v2",
			DecodableVersions: []string{"v2"},
		},
	}

	if !reflect.DeepEqual(storageVersion.Status.StorageVersions, expectedServerStorageVersions) {
		t.Error("unexpected storage version object")
		t.Logf("got: %+v", storageVersion)
		t.Logf("expected: %+v", expectedServerStorageVersions)
	}

	if len(storageVersion.Status.Conditions) != 1 {
		t.Errorf("expected 1 condition, got: %d", len(storageVersion.Status.Conditions))
	}

	if storageVersion.Status.Conditions[0].Type != apiserverinternalv1alpha1.AllEncodingVersionsEqual {
		t.Errorf("expected condition type 'AllEncodingVersionsEqual', got: %q", storageVersion.Status.Conditions[0].Type)
	}

	if storageVersion.Status.Conditions[0].Status != apiserverinternalv1alpha1.ConditionFalse {
		t.Errorf("expected condition status 'True', got: %q", storageVersion.Status.Conditions[0].Status)
	}
}

func Test_StorageVersionDeletedOnLeaseDeletion(t *testing.T) {
	lease1 := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver-1",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s.io/component": "kube-apiserver",
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity: utilpointer.StringPtr("kube-apiserver-1"),
		},
	}

	storageVersion := &apiserverinternalv1alpha1.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name: "k8s.test.resources",
		},
		Status: apiserverinternalv1alpha1.StorageVersionStatus{
			StorageVersions: []apiserverinternalv1alpha1.ServerStorageVersion{
				{
					APIServerID:       "kube-apiserver-1",
					EncodingVersion:   "v1",
					DecodableVersions: []string{"v1"},
				},
			},
		},
	}

	clientset := fake.NewSimpleClientset(lease1, storageVersion)
	setupController(clientset)

	// Delete the lease object and verify that storage version status is updated
	if err := clientset.CoordinationV1().Leases(metav1.NamespaceSystem).Delete(context.Background(), "kube-apiserver-1", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("error deleting lease object: %v", err)
	}

	// add a delay to ensure controller had a chance to reconcile
	time.Sleep(2 * time.Second)

	// expect deleted
	_, err := clientset.InternalV1alpha1().StorageVersions().Get(context.Background(), "k8s.test.resources", metav1.GetOptions{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected IsNotFound error, got: %v", err)
	}
}
