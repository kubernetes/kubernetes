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
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestManagedFieldsOptOut(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManagedFieldsOptOut, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsAllowCBOR, true)

	// Unlike setup, this also serves CRDs.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	ctx, config := t.Context(), server.ClientConfig
	client := clientset.NewForConfigOrDie(config)

	ns := framework.CreateNamespaceOrDie(client, "managedfields-opt-out", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

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

	crd, err := fixtures.CreateNewV1CustomResourceDefinition(
		fixtures.NewRandomNameV1CustomResourceDefinition(apiextensionsv1.ClusterScoped),
		apiextensionsclient.NewForConfigOrDie(config),
		dynamic.NewForConfigOrDie(config))
	if err != nil {
		t.Fatalf("failed to create CRD: %v", err)
	}
	gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural}

	// Custom resources aren't served as protobuf.
	for _, tc := range []struct {
		name      string
		mediaType string
	}{
		{
			name:      "json",
			mediaType: "application/json",
		},
		{
			name:      "cbor",
			mediaType: "application/cbor",
		},
	} {
		t.Run("custom resource/"+tc.name, func(t *testing.T) {
			fullCRs := newDynamicClient(t, config, tc.mediaType).Resource(gvr)
			dropCRs := newDynamicClient(t, config, tc.mediaType+";drop=metadata.managedFields").Resource(gvr)

			initialList, err := fullCRs.List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("failed to list custom resources: %v", err)
			}
			watchOpts := metav1.ListOptions{ResourceVersion: initialList.GetResourceVersion()}

			fullWatch, err := fullCRs.Watch(ctx, watchOpts)
			if err != nil {
				t.Fatalf("failed to start watch: %v", err)
			}
			defer fullWatch.Stop()
			dropWatch, err := dropCRs.Watch(ctx, watchOpts)
			if err != nil {
				t.Fatalf("failed to start watch with drop: %v", err)
			}
			defer dropWatch.Stop()

			name := "cr-" + tc.name
			// managedFields only records the fields set, so it needs a spec.
			cr := &unstructured.Unstructured{Object: map[string]interface{}{"spec": map[string]interface{}{"a": "b"}}}
			cr.SetAPIVersion(gvr.GroupVersion().String())
			cr.SetKind(crd.Spec.Names.Kind)
			cr.SetName(name)
			if _, err := fullCRs.Create(ctx, cr, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create custom resource: %v", err)
			}

			got, err := dropCRs.Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to get custom resource: %v", err)
			}
			expectManagedFields(t, "get", got, false)

			list, err := dropCRs.List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("failed to list custom resources: %v", err)
			}
			if len(list.Items) != 1 {
				t.Fatalf("expected 1 custom resource, got %d", len(list.Items))
			}
			expectManagedFields(t, "list", &list.Items[0], false)

			if err := fullCRs.Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
				t.Fatalf("failed to delete custom resource: %v", err)
			}
			expectManagedFieldsInEvents(t, fullWatch, name, true)
			expectManagedFieldsInEvents(t, dropWatch, name, false)
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

// newDynamicClient returns a dynamic client that accepts the given media
// types, which dynamic.NewForConfig would override.
func newDynamicClient(t *testing.T, config *restclient.Config, accept string) *dynamic.DynamicClient {
	t.Helper()
	config = dynamic.ConfigFor(config)
	config.AcceptContentTypes = accept
	client, err := restclient.UnversionedRESTClientFor(config)
	if err != nil {
		t.Fatalf("failed to create REST client: %v", err)
	}
	return dynamic.New(client)
}

// expectManagedFieldsInEvents checks every event up to the deletion of the object named last.
func expectManagedFieldsInEvents(t *testing.T, w watch.Interface, last string, want bool) {
	t.Helper()
	for {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatalf("watch closed before the deletion of %s", last)
			}
			obj, err := meta.Accessor(event.Object)
			if err != nil {
				t.Fatalf("unexpected %s event: %#v", event.Type, event.Object)
			}
			expectManagedFields(t, fmt.Sprintf("%s event for %s", event.Type, obj.GetName()), obj, want)
			if event.Type == watch.Deleted && obj.GetName() == last {
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
