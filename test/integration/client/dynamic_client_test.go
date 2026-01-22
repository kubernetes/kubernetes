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

package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	clientset "k8s.io/client-go/kubernetes"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestDynamicClient(t *testing.T) {
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer result.TearDownFn()

	client := clientset.NewForConfigOrDie(result.ClientConfig)
	dynamicClient, err := dynamic.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error creating dynamic client: %v", err)
	}

	resource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}

	// Create a Pod with the normal client
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "test",
					Image: "test-image",
				},
			},
		},
	}

	actual, err := client.CoreV1().Pods("default").Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error when creating pod: %v", err)
	}

	// check dynamic list
	unstructuredList, err := dynamicClient.Resource(resource).Namespace("default").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error when listing pods: %v", err)
	}

	if len(unstructuredList.Items) != 1 {
		t.Fatalf("expected one pod, got %d", len(unstructuredList.Items))
	}

	got, err := unstructuredToPod(&unstructuredList.Items[0])
	if err != nil {
		t.Fatalf("unexpected error converting Unstructured to v1.Pod: %v", err)
	}

	if !reflect.DeepEqual(actual, got) {
		t.Fatalf("unexpected pod in list. wanted %#v, got %#v", actual, got)
	}

	// check dynamic get
	unstruct, err := dynamicClient.Resource(resource).Namespace("default").Get(context.TODO(), actual.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error when getting pod %q: %v", actual.Name, err)
	}

	got, err = unstructuredToPod(unstruct)
	if err != nil {
		t.Fatalf("unexpected error converting Unstructured to v1.Pod: %v", err)
	}

	if !reflect.DeepEqual(actual, got) {
		t.Fatalf("unexpected pod in list. wanted %#v, got %#v", actual, got)
	}

	// delete the pod dynamically
	err = dynamicClient.Resource(resource).Namespace("default").Delete(context.TODO(), actual.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("unexpected error when deleting pod: %v", err)
	}

	list, err := client.CoreV1().Pods("default").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error when listing pods: %v", err)
	}

	if len(list.Items) != 0 {
		t.Fatalf("expected zero pods, got %d", len(list.Items))
	}
}

func TestDynamicClientWatch(t *testing.T) {
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer result.TearDownFn()

	client := clientset.NewForConfigOrDie(result.ClientConfig)
	dynamicClient, err := dynamic.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error creating dynamic client: %v", err)
	}

	testDynamicClientWatch(t, client, dynamicClient)
}

func TestDynamicClientWatchWithCBOR(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsAllowCBOR, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsPreferCBOR, true)

	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer result.TearDownFn()

	client := clientset.NewForConfigOrDie(result.ClientConfig)
	dynamicClientConfig := rest.CopyConfig(result.ClientConfig)
	dynamicClientConfig.Wrap(framework.AssertRequestResponseAsCBOR(t))
	dynamicClientConfig.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			response, rterr := rt.RoundTrip(request)
			if rterr != nil {
				return response, rterr
			}

			// We can't synchronously inspect streaming responses, so tee to a buffer
			// and inspect it at the end of the test.
			var buf bytes.Buffer
			response.Body = struct {
				io.Reader
				io.Closer
			}{
				Reader: io.TeeReader(response.Body, &buf),
				Closer: response.Body,
			}
			t.Cleanup(func() {
				var event metav1.WatchEvent
				if err := cbor.Unmarshal(buf.Bytes(), &event); err != nil {
					t.Errorf("non-cbor event: 0x%x", buf.Bytes())
					return
				}
				if err := cbor.Unmarshal(event.Object.Raw, new(interface{})); err != nil {
					t.Errorf("non-cbor event object: 0x%x", buf.Bytes())
				}
			})

			return response, rterr
		})
	})
	dynamicClient, err := dynamic.NewForConfig(dynamicClientConfig)
	if err != nil {
		t.Fatalf("unexpected error creating dynamic client: %v", err)
	}

	testDynamicClientWatch(t, client, dynamicClient)
}

func testDynamicClientWatch(t *testing.T, client clientset.Interface, dynamicClient dynamic.Interface) {
	resource := corev1.SchemeGroupVersion.WithResource("events")

	mkEvent := func(i int) *corev1.Event {
		name := fmt.Sprintf("event-%v", i)
		return &corev1.Event{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "default",
				Name:      name,
			},
			InvolvedObject: corev1.ObjectReference{
				Namespace: "default",
				Name:      name,
			},
			Reason: fmt.Sprintf("event %v", i),
		}
	}

	rv1 := ""
	for i := 0; i < 10; i++ {
		event := mkEvent(i)
		got, err := client.CoreV1().Events("default").Create(context.TODO(), event, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed creating event %#q: %v", event, err)
		}
		if rv1 == "" {
			rv1 = got.ResourceVersion
			if rv1 == "" {
				t.Fatal("did not get a resource version.")
			}
		}
		t.Logf("Created event %#v", got.ObjectMeta)
	}

	w, err := dynamicClient.Resource(resource).Namespace("default").Watch(context.TODO(), metav1.ListOptions{
		ResourceVersion: rv1,
		Watch:           true,
		FieldSelector:   fields.OneTermEqualSelector("metadata.name", "event-9").String(),
	})

	if err != nil {
		t.Fatalf("Failed watch: %v", err)
	}
	defer w.Stop()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("watch took longer than %s", wait.ForeverTestTimeout.String())
	case got, ok := <-w.ResultChan():
		if !ok {
			t.Fatal("Watch channel closed unexpectedly.")
		}

		// We expect to see an ADD of event-9 and only event-9. (This
		// catches a bug where all the events would have been sent down
		// the channel.)
		if e, a := watch.Added, got.Type; e != a {
			t.Errorf("Wanted %v, got %v", e, a)
		}

		unstructured, ok := got.Object.(*unstructured.Unstructured)
		if !ok {
			t.Fatalf("Unexpected watch event containing object %#q", got.Object)
		}
		event, err := unstructuredToEvent(unstructured)
		if err != nil {
			t.Fatalf("unexpected error converting Unstructured to v1.Event: %v", err)
		}
		if e, a := "event-9", event.Name; e != a {
			t.Errorf("Wanted %v, got %v", e, a)
		}
	}
}

func TestUnstructuredExtract(t *testing.T) {
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer result.TearDownFn()

	dynamicClient, err := dynamic.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error creating dynamic client: %v", err)
	}

	resource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}

	// Apply an unstructured with the dynamic client
	name := "test-pod"
	pod := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata": map[string]interface{}{
				"name": name,
				// namespace will always get set by extract,
				// so we add it here (even though it's optional)
				// to ensure what we apply equals what we extract.
				"namespace": "default",
			},
			"spec": map[string]interface{}{
				"containers": []interface{}{
					map[string]interface{}{
						"name":  "test",
						"image": "test-image",
					},
				},
			},
		},
	}
	mgr := "testManager"
	podData, err := json.Marshal(pod)
	if err != nil {
		t.Fatalf("failed to marshal pod into bytes: %v", err)
	}

	// apply the unstructured object to the cluster
	actual, err := dynamicClient.Resource(resource).Namespace("default").Patch(
		context.TODO(),
		name,
		types.ApplyPatchType,
		podData,
		metav1.PatchOptions{FieldManager: mgr})
	if err != nil {
		t.Fatalf("unexpected error when creating pod: %v", err)
	}

	// extract the object
	discoveryClient := discovery.NewDiscoveryClientForConfigOrDie(result.ClientConfig)
	extractor, err := metav1ac.NewUnstructuredExtractor(discoveryClient)
	if err != nil {
		t.Fatalf("unexpected error when constructing extrator: %v", err)
	}
	extracted, err := extractor.Extract(actual, mgr)
	if err != nil {
		t.Fatalf("unexpected error when extracting: %v", err)
	}

	// confirm that the extracted object equals the applied object
	if !reflect.DeepEqual(pod, extracted) {
		t.Fatalf("extracted pod doesn't equal applied pod. wanted:\n %v\n, got:\n %v\n", pod, extracted)
	}

}

func unstructuredToPod(obj *unstructured.Unstructured) (*corev1.Pod, error) {
	json, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}
	pod := new(corev1.Pod)
	err = runtime.DecodeInto(clientscheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion), json, pod)
	pod.Kind = ""
	pod.APIVersion = ""
	return pod, err
}

func unstructuredToEvent(obj *unstructured.Unstructured) (*corev1.Event, error) {
	json, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}
	event := new(corev1.Event)
	err = runtime.DecodeInto(clientscheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion), json, event)
	return event, err
}

func TestDynamicClientCBOREnablement(t *testing.T) {
	DoCreate := func(t *testing.T, config *rest.Config) error {
		client, err := dynamic.NewForConfig(config)
		if err != nil {
			t.Fatal(err)
		}

		_, err = client.Resource(corev1.SchemeGroupVersion.WithResource("namespaces")).Create(
			context.TODO(),
			&unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"name": "test-dynamic-client-cbor-enablement",
					},
				},
			},
			metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}},
		)
		return err
	}

	DoApply := func(t *testing.T, config *rest.Config) error {
		client, err := dynamic.NewForConfig(config)
		if err != nil {
			t.Fatal(err)
		}

		name := "test-dynamic-client-cbor-enablement"
		_, err = client.Resource(corev1.SchemeGroupVersion.WithResource("namespaces")).Apply(
			context.TODO(),
			name,
			&unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Namespace",
					"metadata": map[string]interface{}{
						"name": name,
					},
				},
			},
			metav1.ApplyOptions{
				FieldManager: "foo-bar",
				DryRun:       []string{metav1.DryRunAll},
			},
		)
		return err
	}

	DoWatch := func(t *testing.T, config *rest.Config) error {
		client, err := dynamic.NewForConfig(config)
		if err != nil {
			t.Fatal(err)
		}

		w, err := client.Resource(corev1.SchemeGroupVersion.WithResource("namespaces")).Watch(context.TODO(), metav1.ListOptions{LabelSelector: "a,!a"})
		if err != nil {
			return err
		}
		w.Stop()
		return nil
	}

	testCases := []struct {
		name                    string
		serving                 bool
		allowed                 bool
		preferred               bool
		wantRequestContentType  string
		wantRequestAccept       string
		wantResponseContentType string
		wantResponseStatus      int
		wantStatusError         bool
		doRequest               func(t *testing.T, config *rest.Config) error
	}{
		{
			name:                    "sends cbor accepts both gets cbor",
			serving:                 true,
			allowed:                 true,
			preferred:               true,
			wantRequestContentType:  "application/cbor",
			wantRequestAccept:       "application/json;q=0.9,application/cbor;q=1",
			wantResponseContentType: "application/cbor",
			wantResponseStatus:      http.StatusCreated,
			wantStatusError:         false,
			doRequest:               DoCreate,
		},
		{
			name:                    "sends cbor accepts both gets 415",
			serving:                 false,
			allowed:                 true,
			preferred:               true,
			wantRequestContentType:  "application/cbor",
			wantRequestAccept:       "application/json;q=0.9,application/cbor;q=1",
			wantResponseContentType: "application/json",
			wantResponseStatus:      http.StatusUnsupportedMediaType,
			wantStatusError:         true,
			doRequest:               DoCreate,
		},
		{
			name:                    "sends json accepts both gets cbor",
			serving:                 true,
			allowed:                 true,
			preferred:               false,
			wantRequestContentType:  "application/json",
			wantRequestAccept:       "application/json;q=0.9,application/cbor;q=1",
			wantResponseContentType: "application/cbor",
			wantResponseStatus:      http.StatusCreated,
			wantStatusError:         false,
			doRequest:               DoCreate,
		},
		{
			name:                    "sends json accepts both gets json",
			serving:                 false,
			allowed:                 true,
			preferred:               false,
			wantRequestContentType:  "application/json",
			wantRequestAccept:       "application/json;q=0.9,application/cbor;q=1",
			wantResponseContentType: "application/json",
			wantResponseStatus:      http.StatusCreated,
			wantStatusError:         false,
			doRequest:               DoCreate,
		},
		{
			name:                    "sends json accepts json gets json with serving enabled",
			serving:                 true,
			allowed:                 false,
			preferred:               false,
			wantRequestContentType:  "application/json",
			wantRequestAccept:       "application/json",
			wantResponseContentType: "application/json",
			wantResponseStatus:      http.StatusCreated,
			wantStatusError:         false,
			doRequest:               DoCreate,
		},
		{
			name:                    "sends json accepts json gets json with serving disabled",
			serving:                 false,
			allowed:                 false,
			preferred:               false,
			wantRequestContentType:  "application/json",
			wantRequestAccept:       "application/json",
			wantResponseContentType: "application/json",
			wantResponseStatus:      http.StatusCreated,
			wantStatusError:         false,
			doRequest:               DoCreate,
		},
		{
			name:                    "sends json without both gates enabled",
			serving:                 true,
			allowed:                 false,
			preferred:               true,
			wantRequestContentType:  "application/json",
			wantRequestAccept:       "application/json",
			wantResponseContentType: "application/json",
			wantResponseStatus:      http.StatusCreated,
			wantStatusError:         false,
			doRequest:               DoCreate,
		},
		{
			name:                    "apply sends cbor accepts both gets cbor",
			serving:                 true,
			allowed:                 true,
			preferred:               true,
			wantRequestContentType:  "application/apply-patch+cbor",
			wantRequestAccept:       "application/json;q=0.9,application/cbor;q=1",
			wantResponseContentType: "application/cbor",
			wantResponseStatus:      http.StatusCreated,
			wantStatusError:         false,
			doRequest:               DoApply,
		},
		{
			name:                    "apply sends json accepts both gets cbor",
			serving:                 true,
			allowed:                 true,
			preferred:               false,
			wantRequestContentType:  "application/apply-patch+yaml",
			wantRequestAccept:       "application/json;q=0.9,application/cbor;q=1",
			wantResponseContentType: "application/cbor",
			wantResponseStatus:      http.StatusCreated,
			wantStatusError:         false,
			doRequest:               DoApply,
		},
		{
			name:                    "apply sends cbor accepts both gets 415",
			serving:                 false,
			allowed:                 true,
			preferred:               true,
			wantRequestContentType:  "application/apply-patch+cbor",
			wantRequestAccept:       "application/json;q=0.9,application/cbor;q=1",
			wantResponseContentType: "application/json",
			wantResponseStatus:      http.StatusUnsupportedMediaType,
			wantStatusError:         true,
			doRequest:               DoApply,
		},
		{
			name:                    "watch accepts both gets cbor-seq",
			serving:                 true,
			allowed:                 true,
			preferred:               false,
			wantRequestAccept:       "application/json;q=0.9,application/cbor;q=1",
			wantResponseContentType: "application/cbor-seq",
			wantResponseStatus:      http.StatusOK,
			wantStatusError:         false,
			doRequest:               DoWatch,
		},
	}

	for _, serving := range []bool{true, false} {
		t.Run(fmt.Sprintf("serving=%t", serving), func(t *testing.T) {
			if serving {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, true)
			}

			server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
			defer server.TearDownFn()

			for _, tc := range testCases {
				if serving != tc.serving {
					continue
				}

				t.Run(tc.name, func(t *testing.T) {
					clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsAllowCBOR, tc.allowed)
					clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsPreferCBOR, tc.preferred)

					config := rest.CopyConfig(server.ClientConfig)
					config.Wrap(func(rt http.RoundTripper) http.RoundTripper {
						return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
							response, err := rt.RoundTrip(request)
							if got := response.Request.Header.Get("Content-Type"); got != tc.wantRequestContentType {
								t.Errorf("want request content type %q, got %q", tc.wantRequestContentType, got)
							}
							if got := response.Request.Header.Get("Accept"); got != tc.wantRequestAccept {
								t.Errorf("want request accept %q, got %q", tc.wantRequestAccept, got)
							}
							if got := response.Header.Get("Content-Type"); got != tc.wantResponseContentType {
								t.Errorf("want response content type %q, got %q", tc.wantResponseContentType, got)
							}
							if got := response.StatusCode; got != tc.wantResponseStatus {
								t.Errorf("want response status %d, got %d", tc.wantResponseStatus, got)
							}
							return response, err
						})
					})
					err := tc.doRequest(t, config)
					switch {
					case tc.wantStatusError && errors.IsUnsupportedMediaType(err):
						// ok
					case !tc.wantStatusError && err == nil:
						// ok
					default:
						t.Errorf("unexpected error: %v", err)
					}
				})
			}
		})
	}
}

func TestUnsupportedMediaTypeCircuitBreakerDynamicClient(t *testing.T) {
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsAllowCBOR, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsPreferCBOR, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	config := rest.CopyConfig(server.ClientConfig)

	client, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := client.Resource(corev1.SchemeGroupVersion.WithResource("namespaces")).Create(
		context.TODO(),
		&unstructured.Unstructured{Object: map[string]interface{}{"metadata": map[string]interface{}{"name": "test-dynamic-client-415"}}},
		metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}},
	); !errors.IsUnsupportedMediaType(err) {
		t.Errorf("expected to receive unsupported media type on first cbor request, got: %v", err)
	}

	// Requests from this client should fall back from application/cbor to application/json.
	if _, err := client.Resource(corev1.SchemeGroupVersion.WithResource("namespaces")).Create(
		context.TODO(),
		&unstructured.Unstructured{Object: map[string]interface{}{"metadata": map[string]interface{}{"name": "test-dynamic-client-415"}}},
		metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}},
	); err != nil {
		t.Errorf("expected to receive nil error on subsequent cbor request, got: %v", err)
	}

	// The circuit breaker trips on a per-client basis, so it should not begin tripped for a
	// fresh client with identical config.
	client, err = dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := client.Resource(corev1.SchemeGroupVersion.WithResource("namespaces")).Create(
		context.TODO(),
		&unstructured.Unstructured{Object: map[string]interface{}{"metadata": map[string]interface{}{"name": "test-dynamic-client-415"}}},
		metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}},
	); !errors.IsUnsupportedMediaType(err) {
		t.Errorf("expected to receive unsupported media type on cbor request with fresh client, got: %v", err)
	}
}
