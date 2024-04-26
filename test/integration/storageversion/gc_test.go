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

package storageversion

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	apiserverinternalv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	coordinationv1 "k8s.io/api/coordination/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/storageversiongc"
	"k8s.io/kubernetes/pkg/controlplane"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

const (
	svName     = "storageversion.integration.test.foos"
	idA        = "id-1"
	idB        = "id-2"
	idNonExist = "id-non-exist"
)

func TestStorageVersionGarbageCollection(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer result.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	informers := informers.NewSharedInformerFactory(kubeclient, time.Second)
	leaseInformer := informers.Coordination().V1().Leases()
	storageVersionInformer := informers.Internal().V1alpha1().StorageVersions()

	_, ctx := ktesting.NewTestContext(t)
	controller := storageversiongc.NewStorageVersionGC(ctx, kubeclient, leaseInformer, storageVersionInformer)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go leaseInformer.Informer().Run(ctx.Done())
	go storageVersionInformer.Informer().Run(ctx.Done())
	go controller.Run(ctx)

	createTestAPIServerIdentityLease(t, kubeclient, idA)
	createTestAPIServerIdentityLease(t, kubeclient, idB)

	t.Run("storage version with non-existing id should be GC'ed", func(t *testing.T) {
		createTestStorageVersion(t, kubeclient, idNonExist)
		assertStorageVersionDeleted(t, kubeclient)
	})

	t.Run("storage version with valid id should not be GC'ed", func(t *testing.T) {
		createTestStorageVersion(t, kubeclient, idA)
		time.Sleep(10 * time.Second)
		sv, err := kubeclient.InternalV1alpha1().StorageVersions().Get(
			context.TODO(), svName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to retrieve valid storage version: %v", err)
		}
		if len(sv.Status.StorageVersions) != 1 {
			t.Errorf("unexpected number of storage version entries, expected 1, got: %v",
				sv.Status.StorageVersions)
		}
		expectedID := idA
		if sv.Status.StorageVersions[0].APIServerID != expectedID {
			t.Errorf("unexpected storage version entry id, expected %v, got: %v",
				expectedID, sv.Status.StorageVersions[0].APIServerID)
		}
		assertCommonEncodingVersion(t, kubeclient, pointer.String(idToVersion(t, idA)))
		if err := kubeclient.InternalV1alpha1().StorageVersions().Delete(
			context.TODO(), svName, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("failed to cleanup valid storage version: %v", err)
		}
	})

	t.Run("deleting an id should delete a storage version entry that it owns", func(t *testing.T) {
		createTestStorageVersion(t, kubeclient, idA, idB)
		assertStorageVersionEntries(t, kubeclient, 2, idA)
		assertCommonEncodingVersion(t, kubeclient, nil)
		deleteTestAPIServerIdentityLease(t, kubeclient, idA)
		assertStorageVersionEntries(t, kubeclient, 1, idB)
		assertCommonEncodingVersion(t, kubeclient, pointer.String(idToVersion(t, idB)))
	})

	t.Run("deleting an id should delete a storage version object that it owns entirely", func(t *testing.T) {
		deleteTestAPIServerIdentityLease(t, kubeclient, idB)
		assertStorageVersionDeleted(t, kubeclient)
	})
}

func createTestStorageVersion(t *testing.T, client kubernetes.Interface, ids ...string) {
	sv := &apiserverinternalv1alpha1.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name: svName,
		},
	}
	for _, id := range ids {
		version := idToVersion(t, id)
		v := apiserverinternalv1alpha1.ServerStorageVersion{
			APIServerID:       id,
			EncodingVersion:   version,
			DecodableVersions: []string{version},
		}
		sv.Status.StorageVersions = append(sv.Status.StorageVersions, v)
	}
	// every id is unique and creates a different version. We know we have a common encoding
	// version when there is only one id. Pick it
	if len(ids) == 1 {
		sv.Status.CommonEncodingVersion = pointer.String(sv.Status.StorageVersions[0].EncodingVersion)
	}

	createdSV, err := client.InternalV1alpha1().StorageVersions().Create(context.TODO(), sv, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create storage version %s: %v", svName, err)
	}
	// update the created sv with intended status
	createdSV.Status = sv.Status
	if _, err := client.InternalV1alpha1().StorageVersions().UpdateStatus(
		context.TODO(), createdSV, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("failed to update store version status: %v", err)
	}
}

func assertStorageVersionDeleted(t *testing.T, client kubernetes.Interface) {
	if err := wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		_, err := client.InternalV1alpha1().StorageVersions().Get(
			context.TODO(), svName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if err != nil {
			return false, err
		}
		return false, nil
	}); err != nil {
		t.Fatalf("failed to wait for storageversion garbage collection: %v", err)
	}
}

func createTestAPIServerIdentityLease(t *testing.T, client kubernetes.Interface, name string) {
	lease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				controlplaneapiserver.IdentityLeaseComponentLabelKey: controlplane.KubeAPIServer,
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       pointer.String(name),
			LeaseDurationSeconds: pointer.Int32(3600),
			// create fresh leases
			AcquireTime: &metav1.MicroTime{Time: time.Now()},
			RenewTime:   &metav1.MicroTime{Time: time.Now()},
		},
	}
	if _, err := client.CoordinationV1().Leases(metav1.NamespaceSystem).Create(
		context.TODO(), lease, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create apiserver identity lease %s: %v", name, err)
	}
}

func deleteTestAPIServerIdentityLease(t *testing.T, client kubernetes.Interface, name string) {
	if err := client.CoordinationV1().Leases(metav1.NamespaceSystem).Delete(
		context.TODO(), name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("failed to delete apiserver identity lease %s: %v", name, err)
	}
}

func assertStorageVersionEntries(t *testing.T, client kubernetes.Interface,
	numEntries int, firstID string) {
	var lastErr error
	if err := wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		sv, err := client.InternalV1alpha1().StorageVersions().Get(
			context.TODO(), svName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if len(sv.Status.StorageVersions) != numEntries {
			lastErr = fmt.Errorf("unexpected number of storage version entries, expected %v, got: %v",
				numEntries, len(sv.Status.StorageVersions))
			return false, nil
		}
		if sv.Status.StorageVersions[0].APIServerID != firstID {
			lastErr = fmt.Errorf("unexpected first storage version entry id, expected %v, got: %v",
				firstID, sv.Status.StorageVersions[0].APIServerID)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("failed to get expected storage verion entries: %v, last error: %v", err, lastErr)
	}
}

func assertCommonEncodingVersion(t *testing.T, client kubernetes.Interface, e *string) {
	sv, err := client.InternalV1alpha1().StorageVersions().Get(
		context.TODO(), svName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to retrieve storage version: %v", err)
	}
	if e == nil {
		if sv.Status.CommonEncodingVersion != nil {
			t.Errorf("unexpected non-nil common encoding version: %v", sv.Status.CommonEncodingVersion)
		}
		return
	}
	if sv.Status.CommonEncodingVersion == nil || *sv.Status.CommonEncodingVersion != *e {
		t.Errorf("unexpected common encoding version, expected: %v, got %v", e, sv.Status.CommonEncodingVersion)
	}
}

func idToVersion(t *testing.T, id string) string {
	// TODO(roycaihw): rewrite the test, use a id-version table
	if !strings.HasPrefix(id, "id-") {
		t.Fatalf("should not happen: test using id without id- prefix: %s", id)
	}
	return fmt.Sprintf("v%s", strings.TrimPrefix(id, "id-"))
}
