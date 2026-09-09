/*
Copyright 2025 The Kubernetes Authors.

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

package staleness

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestPodListerRV(t *testing.T) {
	// Start the server with default storage setup
	server := apiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	ctx := ktesting.Init(t)

	clientSet, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("error in create clientset: %v", err)
	}

	testpod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container1",
					Image: "image1",
				},
			},
		},
	}

	// Create pods
	createdPod, err := clientSet.CoreV1().Pods("default").Create(ctx, testpod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error in create pod: %v", err)
	}

	factory := informers.NewSharedInformerFactoryWithOptions(
		clientSet,
		0,
	)
	store := factory.Core().V1().Pods().Informer().GetStore()
	factory.Start(ctx.Done())
	factory.WaitForCacheSync(ctx.Done())

	rv := store.LastStoreSyncResourceVersion()
	if rv == "" {
		t.Fatalf("Expected rv to be set: %v", err)
	}
	// Update the pod labels to increment the resource version.
	createdPod.Labels = map[string]string{"foo": "bar"}
	_, err = clientSet.CoreV1().Pods("default").Update(ctx, createdPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("error in update pod: %v", err)
	}
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 5*time.Second, true, rvGreatherThan(rv, store))
	if err != nil {
		t.Errorf("RV did not increase after update: %v", err)
	}
	// Test that delete also increments the RV
	rv = store.LastStoreSyncResourceVersion()
	if rv == "" {
		t.Error("Expected rv to be set, but it was not")
	}
	err = clientSet.CoreV1().Pods("default").Delete(ctx, createdPod.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("error in delete pod: %v", err)
	}
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 5*time.Second, true, rvGreatherThan(rv, store))
	if err != nil {
		t.Errorf("RV did not increase after delete: %v", err)
	}
}

func rvGreatherThan(rv string, store cache.Store) func(ctx context.Context) (bool, error) {
	return func(ctx context.Context) (bool, error) {
		curRv := store.LastStoreSyncResourceVersion()
		cmp, err := resourceversion.CompareResourceVersion(rv, curRv)
		if err != nil {
			return false, err
		}
		if cmp >= 0 {
			return false, nil
		}
		return true, nil
	}
}
