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
	"testing"
	"time"

	"k8s.io/api/apiserverinternal/v1alpha1"
	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storageversion"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	aggregatorclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

type wrappedStorageVersionManager struct {
	storageversion.Manager
	startUpdateSV  <-chan struct{}
	updateFinished chan<- struct{}
	finishUpdateSV <-chan struct{}
	completed      <-chan struct{}
}

func (w *wrappedStorageVersionManager) UpdateStorageVersions(loopbackClientConfig *rest.Config, serverID string) {
	<-w.startUpdateSV
	w.Manager.UpdateStorageVersions(loopbackClientConfig, serverID)
	close(w.updateFinished)
	<-w.finishUpdateSV
}

func (w *wrappedStorageVersionManager) Completed() bool {
	select {
	case <-w.completed:
		return true
	default:
		return false
	}
}

func assertBlocking(name string, t *testing.T, err error, shouldBlock bool) {
	if shouldBlock {
		if err == nil || !errors.IsServiceUnavailable(err) {
			t.Fatalf("%q should be rejected with service unavailable error, got %v", name, err)
		}
	} else {
		if err != nil {
			t.Fatalf("%q should be allowed, got %v", name, err)
		}
	}
}

func testBuiltinResourceWrite(t *testing.T, cfg *rest.Config, shouldBlock bool) {
	client := clientset.NewForConfigOrDie(cfg)
	_, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "test"}}, metav1.CreateOptions{})
	assertBlocking("writes to built in resources", t, err, shouldBlock)
}

func testCRDWrite(t *testing.T, cfg *rest.Config, shouldBlock bool) {
	crdClient := apiextensionsclientset.NewForConfigOrDie(cfg)
	_, err := crdClient.ApiextensionsV1().CustomResourceDefinitions().Create(context.TODO(), etcd.GetCustomResourceDefinitionData()[1], metav1.CreateOptions{})
	assertBlocking("writes to CRD", t, err, shouldBlock)
}

func testAPIServiceWrite(t *testing.T, cfg *rest.Config, shouldBlock bool) {
	aggregatorClient := aggregatorclient.NewForConfigOrDie(cfg)
	_, err := aggregatorClient.ApiregistrationV1().APIServices().Create(context.TODO(), &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1alpha1.wardle.example.com"},
		Spec: apiregistrationv1.APIServiceSpec{
			Service: &apiregistrationv1.ServiceReference{
				Namespace: "kube-wardle",
				Name:      "api",
			},
			Group:                "wardle.example.com",
			Version:              "v1alpha1",
			GroupPriorityMinimum: 200,
			VersionPriority:      200,
		},
	}, metav1.CreateOptions{})
	assertBlocking("writes to APIServices", t, err, shouldBlock)
}

func testCRWrite(t *testing.T, cfg *rest.Config, shouldBlock bool) {
	dynamicClient := dynamic.NewForConfigOrDie(cfg)
	crclient := dynamicClient.Resource(schema.GroupVersionResource{Group: "cr.bar.com", Version: "v1", Resource: "foos"}).Namespace("default")
	_, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"generateName": "test-"}}}, metav1.CreateOptions{})
	assertBlocking("writes to CR", t, err, shouldBlock)
}

func testStorageVersionWrite(t *testing.T, cfg *rest.Config, shouldBlock bool) {
	apiserverClient := clientset.NewForConfigOrDie(cfg)
	_, err := apiserverClient.InternalV1alpha1().StorageVersions().Create(context.TODO(), &v1alpha1.StorageVersion{ObjectMeta: metav1.ObjectMeta{GenerateName: "test.resource"}}, metav1.CreateOptions{})
	assertBlocking("writes to Storage Version", t, err, shouldBlock)
}

func testNonPersistedWrite(t *testing.T, cfg *rest.Config, shouldBlock bool) {
	client := clientset.NewForConfigOrDie(cfg)
	_, err := client.AuthenticationV1().TokenReviews().Create(context.TODO(), &authenticationv1.TokenReview{
		Spec: authenticationv1.TokenReviewSpec{
			Token: "some token",
		},
	}, metav1.CreateOptions{})
	assertBlocking("non-persisted write", t, err, shouldBlock)
}

func testBuiltinResourceRead(t *testing.T, cfg *rest.Config, shouldBlock bool) {
	client := clientset.NewForConfigOrDie(cfg)
	_, err := client.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
	assertBlocking("reads of built-in resources", t, err, shouldBlock)
}

// TestStorageVersionBootstrap ensures that before the StorageVersions are
// updated, only the following the request are accepted by the apiserver:
// 1. read requests
// 2. non-persisting write requests
// 3. write requests to the storageversion API
// 4. requests to CR or aggregated API
func TestStorageVersionBootstrap(t *testing.T) {
	// Start server and create CRD
	etcdConfig := framework.SharedEtcd()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, etcdConfig)
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(server.ClientConfig), false, etcd.GetCustomResourceDefinitionData()[0])
	server.TearDownFn()

	startUpdateSV := make(chan struct{})
	finishUpdateSV := make(chan struct{})
	updateFinished := make(chan struct{})
	completed := make(chan struct{})
	wrapperFunc := func(delegate storageversion.Manager) storageversion.Manager {
		return &wrappedStorageVersionManager{
			startUpdateSV:  startUpdateSV,
			finishUpdateSV: finishUpdateSV,
			updateFinished: updateFinished,
			completed:      completed,
			Manager:        delegate,
		}
	}
	// Restart api server, enable the storage version API and the feature gates.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	server = kubeapiservertesting.StartTestServerOrDie(t,
		&kubeapiservertesting.TestServerInstanceOptions{
			StorageVersionWrapFunc: wrapperFunc,
		},
		[]string{
			// force enable all resources to ensure that the storage updates can handle cross group resources.
			// TODO: drop these once we stop allowing them to be served.
			"--runtime-config=api/all=true,extensions/v1beta1/deployments=true,extensions/v1beta1/daemonsets=true,extensions/v1beta1/replicasets=true,extensions/v1beta1/podsecuritypolicies=true,extensions/v1beta1/networkpolicies=true,internal.apiserver.k8s.io/v1alpha1=true",
		},
		etcdConfig)
	defer server.TearDownFn()

	cfg := rest.CopyConfig(server.ClientConfig)

	t.Run("before storage version update", func(t *testing.T) {
		t.Run("write to k8s native API object should be blocked", func(t *testing.T) {
			testBuiltinResourceWrite(t, cfg, true)
		})
		t.Run("write to CRD should be blocked", func(t *testing.T) {
			testCRDWrite(t, cfg, true)
		})
		t.Run("write to APIService should be blocked", func(t *testing.T) {
			testAPIServiceWrite(t, cfg, true)
		})
		t.Run("write to CR should pass", func(t *testing.T) {
			testCRWrite(t, cfg, false)
		})
		t.Run("write to the storage version API should pass", func(t *testing.T) {
			testStorageVersionWrite(t, cfg, false)
		})
		t.Run("write to non-persisted API should pass", func(t *testing.T) {
			testNonPersistedWrite(t, cfg, false)
		})
		t.Run("read of k8s native API should pass", func(t *testing.T) {
			testBuiltinResourceRead(t, cfg, false)
		})
		// TODO: Write to aggregated API should pass.
	})

	// After the storage versions are updated, even though the
	// StorageVersionManager.Complete() still returns false, the filter
	// should not block any request.
	close(startUpdateSV)
	<-updateFinished
	t.Run("after storage version update", func(t *testing.T) {
		t.Run("write to k8s native API object should pass", func(t *testing.T) {
			testBuiltinResourceWrite(t, cfg, false)
		})
		t.Run("write to CRD should pass", func(t *testing.T) {
			testCRDWrite(t, cfg, false)
		})
		t.Run("write to APIService should pass", func(t *testing.T) {
			testAPIServiceWrite(t, cfg, false)
		})
		t.Run("write to the storage version API should pass", func(t *testing.T) {
			testStorageVersionWrite(t, cfg, false)
		})
		t.Run("write to non-persisted API should pass", func(t *testing.T) {
			testNonPersistedWrite(t, cfg, false)
		})
		t.Run("read of k8s native API should pass", func(t *testing.T) {
			testBuiltinResourceRead(t, cfg, false)
		})
	})

	// After the StorageVersionManager.Complete() returns true, the server should become healthy.
	close(completed)
	close(finishUpdateSV)
	t.Run("after storage version manager complete", func(t *testing.T) {
		// wait until healthz endpoint returns ok
		client := clientset.NewForConfigOrDie(cfg)
		err := wait.Poll(100*time.Millisecond, 10*time.Second, func() (bool, error) {
			result := client.CoreV1().RESTClient().Get().AbsPath("/healthz").Do(context.TODO())
			status := 0
			result.StatusCode(&status)
			if status == 200 {
				return true, nil
			}
			return false, nil
		})
		if err != nil {
			t.Errorf("failed to wait for /healthz to return ok: %v", err)
		}
	})
}
