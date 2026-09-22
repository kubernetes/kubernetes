/*
Copyright The Kubernetes Authors.

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

package apiserver

import (
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestManagedFieldsOptOut(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManagedFieldsOptOut, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsAllowCBOR, true)

	ctx, client, config, tearDownFn := setup(t)
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "managedfields-opt-out", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	sa := &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "default"}}
	if _, err := client.CoreV1().ServiceAccounts(ns.Name).Create(ctx, sa, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create default service account: %v", err)
	}

	for _, tc := range []struct {
		name      string
		mediaType string
	}{
		{
			name:      "json",
			mediaType: "application/json",
		},
		{
			name:      "protobuf",
			mediaType: "application/vnd.kubernetes.protobuf",
		},
		{
			name:      "cbor",
			mediaType: "application/cbor",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Both clients accept the same media type, so drop is the only
			// difference between their watch encodings.
			fullConfig := restclient.CopyConfig(config)
			fullConfig.AcceptContentTypes = tc.mediaType
			fullPods := clientset.NewForConfigOrDie(fullConfig).CoreV1().Pods(ns.Name)
			dropConfig := restclient.CopyConfig(config)
			dropConfig.AcceptContentTypes = tc.mediaType + ";drop=metadata.managedFields"
			dropClient := clientset.NewForConfigOrDie(dropConfig)
			dropPods := dropClient.CoreV1().Pods(ns.Name)

			// Get initial resource version.
			initialList, err := fullPods.List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("failed to list pods: %v", err)
			}
			watchOpts := metav1.ListOptions{ResourceVersion: initialList.ResourceVersion}

			// Watch with and without drop at once, so the watch cache encodes
			// each event both ways.
			fullWatch, err := fullPods.Watch(ctx, watchOpts)
			if err != nil {
				t.Fatalf("failed to start watch: %v", err)
			}
			defer fullWatch.Stop()
			dropWatch, err := dropPods.Watch(ctx, watchOpts)
			if err != nil {
				t.Fatalf("failed to start watch with drop: %v", err)
			}
			defer dropWatch.Stop()

			pod, err := dropPods.Create(ctx, newPod("pod-1"), metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create pod: %v", err)
			}
			expectManagedFields(t, "create", pod, false)

			pod, err = dropPods.Get(ctx, "pod-1", metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to get pod: %v", err)
			}
			expectManagedFields(t, "get", pod, false)

			list, err := dropPods.List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("failed to list pods: %v", err)
			}
			if len(list.Items) != 1 {
				t.Fatalf("expected 1 pod, got %d", len(list.Items))
			}
			expectManagedFields(t, "list", &list.Items[0], false)

			// Writing back an object read without managedFields must keep the
			// stored ones, which the watch without drop checks.
			pod.Labels = map[string]string{"updated": "true"}
			pod, err = dropPods.Update(ctx, pod, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("failed to update pod: %v", err)
			}
			expectManagedFields(t, "update", pod, false)

			pod, err = dropPods.Patch(ctx, "pod-1", types.MergePatchType, []byte(`{"metadata":{"labels":{"patched":"true"}}}`), metav1.PatchOptions{})
			if err != nil {
				t.Fatalf("failed to patch pod: %v", err)
			}
			expectManagedFields(t, "patch", pod, false)

			// An unsupported drop value makes its clause unacceptable, so the
			// next clause is used.
			fallback := &v1.Pod{}
			if err := dropClient.CoreV1().RESTClient().Get().Namespace(ns.Name).Resource("pods").Name("pod-1").
				SetHeader("Accept", tc.mediaType+";drop=spec, "+tc.mediaType).
				Do(ctx).Into(fallback); err != nil {
				t.Fatalf("failed to get pod with an unsupported drop value: %v", err)
			}
			expectManagedFields(t, "get with an unsupported drop value", fallback, true)

			deleted := &v1.Pod{}
			if err := dropClient.CoreV1().RESTClient().Delete().Namespace(ns.Name).Resource("pods").Name("pod-1").Do(ctx).Into(deleted); err != nil {
				t.Fatalf("failed to delete pod: %v", err)
			}
			expectManagedFields(t, "delete", deleted, false)

			if _, err := dropPods.Create(ctx, newPod("pod-2"), metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create pod: %v", err)
			}
			deletedList := &v1.PodList{}
			if err := dropClient.CoreV1().RESTClient().Delete().Namespace(ns.Name).Resource("pods").Do(ctx).Into(deletedList); err != nil {
				t.Fatalf("failed to delete pods: %v", err)
			}
			if len(deletedList.Items) != 1 {
				t.Fatalf("expected 1 deleted pod, got %d", len(deletedList.Items))
			}
			expectManagedFields(t, "deletecollection", &deletedList.Items[0], false)

			expectManagedFieldsInEvents(t, fullWatch, "pod-2", true)
			expectManagedFieldsInEvents(t, dropWatch, "pod-2", false)
		})
	}
}

func TestManagedFieldsOptOutFeatureGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManagedFieldsOptOut, false)

	ctx, client, config, tearDownFn := setup(t)
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "managedfields-opt-out-disabled", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// Create with drop — when the gate is off, drop should be ignored and
	// managedFields returned.
	dropConfig := restclient.CopyConfig(config)
	dropConfig.AcceptContentTypes = "application/json;drop=metadata.managedFields"
	cm, err := clientset.NewForConfigOrDie(dropConfig).CoreV1().ConfigMaps(ns.Name).Create(ctx, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
		Data:       map[string]string{"key": "value"},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create configmap: %v", err)
	}
	expectManagedFields(t, "create", cm, true)
}

// expectManagedFieldsInEvents checks every event up to the deletion of the pod named last.
func expectManagedFieldsInEvents(t *testing.T, w watch.Interface, last string, want bool) {
	t.Helper()
	for {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatalf("watch closed before the deletion of %s", last)
			}
			pod, ok := event.Object.(*v1.Pod)
			if !ok {
				t.Fatalf("unexpected %s event: %#v", event.Type, event.Object)
			}
			expectManagedFields(t, fmt.Sprintf("%s event for %s", event.Type, pod.Name), pod, want)
			if event.Type == watch.Deleted && pod.Name == last {
				return
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatalf("timed out waiting for the deletion of %s", last)
		}
	}
}

func expectManagedFields(t *testing.T, what string, obj metav1.Object, want bool) {
	t.Helper()
	if got := len(obj.GetManagedFields()) > 0; got != want {
		t.Errorf("%s: managedFields present = %t, want %t", what, got, want)
	}
}

func newPod(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "c", Image: "nginx"}}},
	}
}
