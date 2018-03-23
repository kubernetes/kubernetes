/*
Copyright 2018 The Kubernetes Authors.

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

package wait

import (
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	dynamicfakeclient "k8s.io/client-go/dynamic/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/printers"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/resource"
)

func newUnstructured(gvk schema.GroupVersionKind, namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": gvk.GroupVersion().String(),
			"kind":       gvk.Kind,
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
				"uid":       "some-UID-value",
			},
		},
	}
}

func addCondition(in *unstructured.Unstructured, name, status string) *unstructured.Unstructured {
	conditions, _, _ := unstructured.NestedSlice(in.Object, "status", "conditions")
	conditions = append(conditions, map[string]interface{}{
		"type":   name,
		"status": status,
	})
	unstructured.SetNestedSlice(in.Object, conditions, "status", "conditions")
	return in
}

func TestWaitForDeletion(t *testing.T) {
	scheme := runtime.NewScheme()

	name := "name-foo"
	namespace := "ns-foo"
	info := &resource.Info{
		Mapping: &meta.RESTMapping{
			Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
			GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
		},
		Name:      name,
		Namespace: namespace,
	}

	tests := []struct {
		name       string
		fakeClient func() *dynamicfakeclient.FakeDynamicClient
		timeout    time.Duration
		uidMap     UIDMap

		expectedErr     error
		validateActions func(t *testing.T, actions []clienttesting.Action)
	}{
		{
			name: "not present at all",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClient(scheme)
			},
			timeout:     10 * time.Second,
			expectedErr: nil,
		},
		{
			name: "times out",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClient(scheme,
					newUnstructured(info.Mapping.GroupVersionKind, namespace, name),
				)
			},
			timeout:     1 * time.Second,
			expectedErr: wait.ErrWaitTimeout,
		},
		{
			name: "handles watch delete",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme,
					newUnstructured(info.Mapping.GroupVersionKind, namespace, name),
				)
				fakeClient.PrependWatchReactor("pods", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Deleted, newUnstructured(info.Mapping.GroupVersionKind, namespace, name))
					return true, fakeWatch, nil
				})

				return fakeClient
			},
			timeout:     10 * time.Second,
			expectedErr: nil,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				found := false
				for _, action := range actions {
					if action.Matches("watch", "pods") {
						found = true
						break
					}
				}

				if !found {
					t.Errorf("no 'watch' action has been recorded: %s", spew.Sdump(actions))
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := test.fakeClient()
			o := &WaitOptions{
				ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(info),
				UIDMap:         test.uidMap,
				DynamicClient:  fakeClient,
				Timeout:        test.timeout,

				Printer:     printers.NewDiscardingPrinter(),
				ConditionFn: IsDeleted,
				IOStreams:   genericclioptions.NewTestIOStreamsDiscard(),
			}
			err := o.RunWait()

			if !reflect.DeepEqual(err, test.expectedErr) {
				t.Fatalf("expected error %v, got: %v", test.expectedErr, err)
			}

			if test.validateActions != nil {
				test.validateActions(t, fakeClient.Actions())
			}
		})
	}
}

func TestWaitForCondition(t *testing.T) {
	var scheme *runtime.Scheme

	name := "name-foo"
	namespace := "ns-foo"
	info := &resource.Info{
		Mapping: &meta.RESTMapping{
			Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
			GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
		},
		Name:      name,
		Namespace: namespace,
	}

	tests := []struct {
		name       string
		fakeClient func() *dynamicfakeclient.FakeDynamicClient
		timeout    time.Duration

		expectedErr     error
		validateActions func(t *testing.T, actions []clienttesting.Action)
	}{
		{
			name: "already present",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClient(scheme,
					addCondition(
						newUnstructured(info.Mapping.GroupVersionKind, namespace, name),
						"the-condition", "status-value",
					))
			},
			timeout:     10 * time.Second,
			expectedErr: nil,
		},
		{
			name: "times out",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClient(scheme)
			},
			timeout:     1 * time.Second,
			expectedErr: wait.ErrWaitTimeout,
		},
		{
			name: "handles watch condition change",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructured(info.Mapping.GroupVersionKind, namespace, name), nil
				})
				fakeClient.PrependWatchReactor("pods", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Modified, addCondition(
						newUnstructured(info.Mapping.GroupVersionKind, namespace, name),
						"the-condition", "status-value",
					))
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout:     10 * time.Second,
			expectedErr: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// We need to create scheme to avoid a data race in fake client
			// https://github.com/kubernetes/kubernetes/pull/66078/files#r203342325
			scheme = runtime.NewScheme()
			fakeClient := test.fakeClient()
			o := &WaitOptions{
				ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(info),
				DynamicClient:  fakeClient,
				Timeout:        test.timeout,

				Printer:     printers.NewDiscardingPrinter(),
				ConditionFn: ConditionalWait{conditionName: "the-condition", conditionStatus: "status-value"}.IsConditionMet,
				IOStreams:   genericclioptions.NewTestIOStreamsDiscard(),
			}
			err := o.RunWait()

			if !reflect.DeepEqual(err, test.expectedErr) {
				t.Fatalf("expected %v, got: %v", test.expectedErr, err)
			}

			if test.validateActions != nil {
				test.validateActions(t, fakeClient.Actions())
			}
		})
	}
}
