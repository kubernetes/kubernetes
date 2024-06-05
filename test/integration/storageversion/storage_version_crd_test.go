/*
Copyright 2024 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"testing"
	"time"

	apiserverinternalv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"

	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
)

func TestStorageVersionCustomResource(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	etcdConfig := framework.SharedEtcd()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		[]string{
			"--runtime-config=internal.apiserver.k8s.io/v1alpha1=true",
		},
		etcdConfig)
	crdClient := apiextensionsclientset.NewForConfigOrDie(server.ClientConfig)
	crd := etcd.GetCustomResourceDefinitionData()[0]
	etcd.CreateTestCRDs(t, crdClient, false, crd)
	defer server.TearDownFn()

	kubeclient := kubernetes.NewForConfigOrDie(server.ClientConfig)
	gr := crd.Spec.Group + "." + crd.Spec.Names.Plural
	createdCRD, err := crdClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), crd.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get crd: %v", err)
	}

	var lastErr error
	var storageVersion apiserverinternalv1alpha1.ServerStorageVersion

	if err := wait.PollUntilContextTimeout(context.TODO(), 100*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		sv, err := kubeclient.InternalV1alpha1().StorageVersions().Get(context.TODO(), gr, metav1.GetOptions{})
		if err != nil {
			lastErr = fmt.Errorf("failed to get storage version: %w", err)
			return false, nil
		}
		ref := metav1.OwnerReference{
			APIVersion: "apiextensions.k8s.io/v1",
			Kind:       "CustomResourceDefinition",
			Name:       createdCRD.Name,
			UID:        createdCRD.UID,
		}
		if len(sv.OwnerReferences) == 0 || sv.OwnerReferences[0] != ref {
			return false, fmt.Errorf("apiserver failed to set crd owner references, expect: %v, got: %v", ref, sv.OwnerReferences)
		}
		if len(sv.Status.StorageVersions) != 1 {
			lastErr = fmt.Errorf("apiserver failed to set one storage version record in the status, got: %v", sv.Status.StorageVersions)
			return false, nil
		}
		storageVersion = sv.Status.StorageVersions[0]
		return true, nil
	}); err != nil {
		t.Fatalf("failed to wait for storage version creation: %v, last error: %v", err, lastErr)
	}

	if !strings.HasPrefix(storageVersion.APIServerID, "apiserver-") {
		t.Errorf("apiserver ID doesn't contain apiserver- prefix, has: %v", storageVersion.APIServerID)
	}
	expectedVersion := crd.Spec.Group + "/" + crd.Spec.Versions[0].Name
	if storageVersion.EncodingVersion != expectedVersion {
		t.Errorf("unexpected encoding version, expected: %v, got: %v", expectedVersion, storageVersion.EncodingVersion)
	}
	if len(storageVersion.DecodableVersions) != 1 {
		t.Errorf("unexpected number of decodable versions, expected 1 version, got: %v", storageVersion.DecodableVersions)
	}
	if storageVersion.DecodableVersions[0] != expectedVersion {
		t.Errorf("unexpected decodable version, expected %v, got: %v", expectedVersion, storageVersion.DecodableVersions[0])
	}

	// add a new version v2 to the CRD and make it the storage version
	newVersion := createdCRD.Spec.Versions[0]
	newVersion.Name = "v2"
	createdCRD.Spec.Versions[0].Storage = false
	createdCRD.Spec.Versions = append(createdCRD.Spec.Versions, newVersion)
	if _, err := crdClient.ApiextensionsV1().CustomResourceDefinitions().Update(context.TODO(), createdCRD, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("failed to update crd: %v", err)
	}

	expectedNewVersion := crd.Spec.Group + "/" + newVersion.Name
	expectedDecodableVersions := []string{expectedVersion, expectedNewVersion}
	lastErr = nil
	if err := wait.PollUntilContextTimeout(context.TODO(), 100*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		sv, err := kubeclient.InternalV1alpha1().StorageVersions().Get(context.TODO(), gr, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		storageVersion := sv.Status.StorageVersions[0]
		if len(storageVersion.DecodableVersions) != 2 {
			lastErr = fmt.Errorf("unexpected number of decodable versions, expected 2 versions, got: %v", len(sv.Status.StorageVersions))
			return false, nil
		}
		if !reflect.DeepEqual(storageVersion.DecodableVersions, expectedDecodableVersions) {
			lastErr = fmt.Errorf("unexpected decodable versions, expected %v, got: %v", expectedDecodableVersions, storageVersion.DecodableVersions)
			return false, nil
		}
		if storageVersion.EncodingVersion != expectedNewVersion {
			lastErr = fmt.Errorf("unexpected encoding version, expected: %v, got: %v", expectedNewVersion, storageVersion.EncodingVersion)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("failed to wait for storage version update: %v, last error: %v", err, lastErr)
	}

	// cleanup
	if err := fixtures.DeleteV1CustomResourceDefinition(crd, crdClient); err != nil {
		t.Fatal(err)
	}
	if err := kubeclient.InternalV1alpha1().StorageVersions().Delete(context.TODO(), gr, metav1.DeleteOptions{}); err != nil {
		t.Errorf("failed to delete storage version: %v", err)
	}
}

func TestStorageVersionMultipleCRDs(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	etcdConfig := framework.SharedEtcd()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		[]string{
			"--runtime-config=internal.apiserver.k8s.io/v1alpha1=true",
		},
		etcdConfig)
	crdClient := apiextensionsclientset.NewForConfigOrDie(server.ClientConfig)
	crd := etcd.GetCustomResourceDefinitionData()[0]
	etcd.CreateTestCRDs(t, crdClient, false, crd)
	defer server.TearDownFn()
	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error creating dynamic client: %v", err)
	}
	gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural}
	errCh := make(chan error)
	// keep flipping the storage version of the CRD and create new CRs
	go func() {
		for i := 0; i < 10; i++ {
			createdCRD, err := crdClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), crd.Name, metav1.GetOptions{})
			if err != nil {
				errCh <- fmt.Errorf("failed to get crd: %w", err)
				return
			}

			// add a new version to the CRD and set it as the storage version
			newVersion := createdCRD.Spec.Versions[i]
			newVersion.Name = fmt.Sprintf("v%d", i+2)
			createdCRD.Spec.Versions[i].Storage = false
			createdCRD.Spec.Versions = append(createdCRD.Spec.Versions, newVersion)
			if _, err := crdClient.ApiextensionsV1().CustomResourceDefinitions().Update(context.TODO(), createdCRD, metav1.UpdateOptions{}); err != nil {
				errCh <- fmt.Errorf("failed to update crd: %w", err)
				return
			}
			cr := &unstructured.Unstructured{}
			cr.SetName(newVersion.Name)
			cr.SetAPIVersion(fmt.Sprintf("%s/v1", crd.Spec.Group))
			cr.SetKind(crd.Spec.Names.Kind)
			if _, err = dynamicClient.Resource(gvr).Namespace("default").Create(context.TODO(), cr, metav1.CreateOptions{}); err != nil {
				fmt.Printf("\nerror creating cr for version %v: %v\n", newVersion.Name, err)
				// errCh <- fmt.Errorf("error creating cr for version %v: %v", newVersion.Name, err)
				// return
			}
		}
	}()

	// verify new CRD creation is not blocked by the watch event.
	newCRD := etcd.GetCustomResourceDefinitionData()[1]
	etcd.CreateTestCRDs(t, crdClient, false, newCRD)

	// verify that the storage version of the first CRD eventually stablize to the expected version
	kubeclient := kubernetes.NewForConfigOrDie(server.ClientConfig)
	gr := crd.Spec.Group + "." + crd.Spec.Names.Plural
	var lastErr error
	if err := wait.PollUntilContextTimeout(context.TODO(), 100*time.Millisecond, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		select {
		case err := <-errCh:
			return false, err
		default:
		}

		sv, err := kubeclient.InternalV1alpha1().StorageVersions().Get(context.TODO(), gr, metav1.GetOptions{})
		if err != nil {
			lastErr = fmt.Errorf("failed to get storage version: %w", err)
			return false, nil
		}
		if len(sv.Status.StorageVersions) != 1 {
			lastErr = fmt.Errorf("apiserver failed to set one storage version record in the status, got: %v", sv.Status.StorageVersions)
			return false, nil
		}
		storageVersion := sv.Status.StorageVersions[0]
		if len(storageVersion.DecodableVersions) != 11 {
			lastErr = fmt.Errorf("unexpected number of decodable versions, expected 11 versions, got: %v", len(sv.Status.StorageVersions))
			return false, nil
		}
		expectedNewVersion := fmt.Sprintf("%s/v11", crd.Spec.Group)
		if storageVersion.EncodingVersion != expectedNewVersion {
			lastErr = fmt.Errorf("unexpected encoding version, expected: %v, got: %v", expectedNewVersion, storageVersion.EncodingVersion)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Errorf("failed to wait for storage version update: %v, last error: %v", err, lastErr)
	}

	// cleanup
	if err := fixtures.DeleteV1CustomResourceDefinition(crd, crdClient); err != nil {
		t.Fatal(err)
	}
	if err := kubeclient.InternalV1alpha1().StorageVersions().Delete(context.TODO(), gr, metav1.DeleteOptions{}); err != nil {
		t.Errorf("failed to delete storage version: %v", err)
	}
	if err := fixtures.DeleteV1CustomResourceDefinition(newCRD, crdClient); err != nil {
		t.Fatal(err)
	}
	gr = newCRD.Spec.Group + "." + newCRD.Spec.Names.Plural
	if err := kubeclient.InternalV1alpha1().StorageVersions().Delete(context.TODO(), gr, metav1.DeleteOptions{}); err != nil {
		t.Errorf("failed to delete storage version: %v", err)
	}

}

func TestWatchAndMutateStorageVersionCRDs(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	etcdConfig := framework.SharedEtcd()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		[]string{
			"--runtime-config=internal.apiserver.k8s.io/v1alpha1=true",
		},
		etcdConfig)
	defer server.TearDownFn()

	crdClient := apiextensionsclientset.NewForConfigOrDie(server.ClientConfig)

	// create a watch on CRDs
	w, err := crdClient.ApiextensionsV1().CustomResourceDefinitions().Watch(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Stop()

	select {
	case <-w.ResultChan():
		t.Fatal("Watch closed before mutation requests were made")
	default:
	}

	// verify new CRD creation is not blocked by watch event
	crd := etcd.GetCustomResourceDefinitionData()[0]
	etcd.CreateTestCRDs(t, crdClient, false, crd)

	errCh := make(chan error)

	kubeclient := kubernetes.NewForConfigOrDie(server.ClientConfig)
	gr := crd.Spec.Group + "." + crd.Spec.Names.Plural
	var lastErr error
	if err := wait.PollUntilContextTimeout(context.TODO(), 100*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		select {
		case err := <-errCh:
			return false, err
		default:
		}

		sv, err := kubeclient.InternalV1alpha1().StorageVersions().Get(context.TODO(), gr, metav1.GetOptions{})
		if err != nil {
			lastErr = fmt.Errorf("failed to get storage version: %w", err)
			return false, nil
		}
		storageVersion := sv.Status.StorageVersions[0]
		expectedVersion := crd.Spec.Group + "/" + crd.Spec.Versions[0].Name
		if storageVersion.EncodingVersion != expectedVersion {
			t.Errorf("unexpected encoding version, expected: %v, got: %v", expectedVersion, storageVersion.EncodingVersion)
		}
		if len(storageVersion.DecodableVersions) != 1 {
			t.Errorf("unexpected number of decodable versions, expected 1 version, got: %v", storageVersion.DecodableVersions)
		}
		if storageVersion.DecodableVersions[0] != expectedVersion {
			t.Errorf("unexpected decodable version, expected %v, got: %v", expectedVersion, storageVersion.DecodableVersions[0])
		}
		return true, nil
	}); err != nil {
		t.Errorf("failed to wait for storage version creation: %v, last error: %v", err, lastErr)
	}

	// cleanup
	if err := fixtures.DeleteV1CustomResourceDefinition(crd, crdClient); err != nil {
		t.Fatal(err)
	}
	if err := kubeclient.InternalV1alpha1().StorageVersions().Delete(context.TODO(), gr, metav1.DeleteOptions{}); err != nil {
		t.Errorf("failed to delete storage version: %v", err)
	}
}
