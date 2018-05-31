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
	"testing"

	"time"

	"strings"

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

func newUnstructured(apiVersion, kind, namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": apiVersion,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
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

	tests := []struct {
		name       string
		info       *resource.Info
		fakeClient func() *dynamicfakeclient.FakeDynamicClient
		timeout    time.Duration

		expectedErr     string
		validateActions func(t *testing.T, actions []clienttesting.Action)
	}{
		{
			name: "missing on get",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClient(scheme)
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 1 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "times out",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"), nil
				})
				return fakeClient
			},
			timeout: 1 * time.Second,

			expectedErr: wait.ErrWaitTimeout.Error(),
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch close out",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"), nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					if count == 0 {
						count++
						fakeWatch := watch.NewRaceFreeFake()
						go func() {
							time.Sleep(100 * time.Millisecond)
							fakeWatch.Stop()
						}()
						return true, fakeWatch, nil
					}
					fakeWatch := watch.NewRaceFreeFake()
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 3 * time.Second,

			expectedErr: wait.ErrWaitTimeout.Error(),
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 4 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
				if !actions[2].Matches("get", "theresource") || actions[2].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[3].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch delete",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"), nil
				})
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Deleted, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"))
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := test.fakeClient()
			o := &WaitOptions{
				ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(test.info),
				DynamicClient:  fakeClient,
				Timeout:        test.timeout,

				Printer:     printers.NewDiscardingPrinter(),
				ConditionFn: IsDeleted,
				IOStreams:   genericclioptions.NewTestIOStreamsDiscard(),
			}
			err := o.RunWait()
			switch {
			case err == nil && len(test.expectedErr) == 0:
			case err != nil && len(test.expectedErr) == 0:
				t.Fatal(err)
			case err == nil && len(test.expectedErr) != 0:
				t.Fatalf("missing: %q", test.expectedErr)
			case err != nil && len(test.expectedErr) != 0:
				if !strings.Contains(err.Error(), test.expectedErr) {
					t.Fatalf("expected %q, got %q", test.expectedErr, err.Error())
				}
			}

			test.validateActions(t, fakeClient.Actions())
		})
	}
}

func TestWaitForCondition(t *testing.T) {
	scheme := runtime.NewScheme()

	tests := []struct {
		name       string
		info       *resource.Info
		fakeClient func() *dynamicfakeclient.FakeDynamicClient
		timeout    time.Duration

		expectedErr     string
		validateActions func(t *testing.T, actions []clienttesting.Action)
	}{
		{
			name: "present on get",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, addCondition(
						newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
						"the-condition", "status-value",
					), nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 1 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "times out",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, addCondition(
						newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
						"some-other-condition", "status-value",
					), nil
				})
				return fakeClient
			},
			timeout: 1 * time.Second,

			expectedErr: wait.ErrWaitTimeout.Error(),
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch close out",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"), nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					if count == 0 {
						count++
						fakeWatch := watch.NewRaceFreeFake()
						go func() {
							time.Sleep(100 * time.Millisecond)
							fakeWatch.Stop()
						}()
						return true, fakeWatch, nil
					}
					fakeWatch := watch.NewRaceFreeFake()
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 3 * time.Second,

			expectedErr: wait.ErrWaitTimeout.Error(),
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 4 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
				if !actions[2].Matches("get", "theresource") || actions[2].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[3].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch condition change",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"), nil
				})
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Modified, addCondition(
						newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
						"the-condition", "status-value",
					))
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch created",
			info: &resource.Info{
				Mapping: &meta.RESTMapping{
					Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
				},
				Name:      "name-foo",
				Namespace: "ns-foo",
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Added, addCondition(
						newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
						"the-condition", "status-value",
					))
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := test.fakeClient()
			o := &WaitOptions{
				ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(test.info),
				DynamicClient:  fakeClient,
				Timeout:        test.timeout,

				Printer:     printers.NewDiscardingPrinter(),
				ConditionFn: ConditionalWait{conditionName: "the-condition", conditionStatus: "status-value"}.IsConditionMet,
				IOStreams:   genericclioptions.NewTestIOStreamsDiscard(),
			}
			err := o.RunWait()
			switch {
			case err == nil && len(test.expectedErr) == 0:
			case err != nil && len(test.expectedErr) == 0:
				t.Fatal(err)
			case err == nil && len(test.expectedErr) != 0:
				t.Fatalf("missing: %q", test.expectedErr)
			case err != nil && len(test.expectedErr) != 0:
				if !strings.Contains(err.Error(), test.expectedErr) {
					t.Fatalf("expected %q, got %q", test.expectedErr, err.Error())
				}
			}

			test.validateActions(t, fakeClient.Actions())
		})
	}
}
