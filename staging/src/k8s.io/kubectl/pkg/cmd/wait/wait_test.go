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
	"io/ioutil"
	"strings"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	dynamicfakeclient "k8s.io/client-go/dynamic/fake"
	clienttesting "k8s.io/client-go/testing"
)

const (
	None    string = ""
	podYAML string = `
apiVersion: v1
kind: Pod
metadata:
    creationTimestamp: "1998-10-21T18:39:43Z"
    generateName: foo-b6699dcfb-
    labels:
      app: nginx
      pod-template-hash: b6699dcfb
    name: foo-b6699dcfb-rnv7t
    namespace: default
    ownerReferences:
    - apiVersion: apps/v1
      blockOwnerDeletion: true
      controller: true
      kind: ReplicaSet
      name: foo-b6699dcfb
      uid: 8fc1088c-15d5-4a8c-8502-4dfcedef97b8
    resourceVersion: "14203463"
    uid: e2cc99fa-5a28-44da-b880-4dded28882ef
spec:
    containers:
    - image: nginx
      imagePullPolicy: IfNotPresent
      name: nginx
      ports:
      - containerPort: 80
      protocol: TCP
      resources:
        limits:
          cpu: 500m
          memory: 128Mi
        requests:
          cpu: 250m
          memory: 64Mi  
      terminationMessagePath: /dev/termination-log
      terminationMessagePolicy: File
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-s64k4
        readOnly: true
    dnsPolicy: ClusterFirst
    enableServiceLinks: true
    nodeName: knode0
    preemptionPolicy: PreemptLowerPriority
    priority: 0
    restartPolicy: Always
    schedulerName: default-scheduler
    securityContext: {}
    serviceAccount: default
    serviceAccountName: default
    terminationGracePeriodSeconds: 30
    tolerations:
    - effect: NoExecute
      key: node.kubernetes.io/not-ready
      operator: Exists
      tolerationSeconds: 300
    - effect: NoExecute
      key: node.kubernetes.io/unreachable
      operator: Exists
      tolerationSeconds: 300
    volumes:
    - name: kube-api-access-s64k4
      projected:
        defaultMode: 420
        sources:
        - serviceAccountToken:
            expirationSeconds: 3607
            path: token
        - configMap:
            items:
            - key: ca.crt
              path: ca.crt
            name: kube-root-ca.crt
status:
    conditions:
    - lastProbeTime: null
      lastTransitionTime: "1998-10-21T18:39:37Z"
      status: "True"
      type: Initialized
    - lastProbeTime: null
      lastTransitionTime: "1998-10-21T18:39:42Z"
      status: "True"
      type: Ready
    - lastProbeTime: null
      lastTransitionTime: "1998-10-21T18:39:42Z"
      status: "True"
      type: ContainersReady
    - lastProbeTime: null
      lastTransitionTime: "1998-10-21T18:39:37Z"
      status: "True"
      type: PodScheduled
    containerStatuses:
    - containerID: containerd://e35792ba1d6e9a56629659b35dbdb93dacaa0a413962ee04775319f5438e493c
      image: docker.io/library/nginx:latest
      imageID: docker.io/library/nginx@sha256:644a70516a26004c97d0d85c7fe1d0c3a67ea8ab7ddf4aff193d9f301670cf36
      lastState: {}
      name: nginx
      ready: true
      restartCount: 0
      started: true
      state:
        running:
          startedAt: "1998-10-21T18:39:41Z"
    hostIP: 192.168.0.22
    phase: Running
    podIP: 10.42.1.203
    podIPs:
    - ip: 10.42.1.203
    qosClass: Burstable
    startTime: "1998-10-21T18:39:37Z"
`
)

func newUnstructuredList(items ...*unstructured.Unstructured) *unstructured.UnstructuredList {
	list := &unstructured.UnstructuredList{}
	for i := range items {
		list.Items = append(list.Items, *items[i])
	}
	return list
}

func newUnstructured(apiVersion, kind, namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": apiVersion,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
				"uid":       "some-UID-value",
			},
		},
	}
}

func newUnstructuredWithGeneration(apiVersion, kind, namespace, name string, generation int64) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": apiVersion,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"namespace":  namespace,
				"name":       name,
				"uid":        "some-UID-value",
				"generation": generation,
			},
		},
	}
}

func newUnstructuredStatus(status *metav1.Status) runtime.Unstructured {
	obj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(status)
	if err != nil {
		panic(err)
	}
	return &unstructured.Unstructured{
		Object: obj,
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

func addConditionWithObservedGeneration(in *unstructured.Unstructured, name, status string, observedGeneration int64) *unstructured.Unstructured {
	conditions, _, _ := unstructured.NestedSlice(in.Object, "status", "conditions")
	conditions = append(conditions, map[string]interface{}{
		"type":               name,
		"status":             status,
		"observedGeneration": observedGeneration,
	})
	unstructured.SetNestedSlice(in.Object, conditions, "status", "conditions")
	return in
}

// createUnstructured parses the yaml string into a map[string]interface{}.  Verifies that the string does not have
// any tab characters.
func createUnstructured(t *testing.T, config string) *unstructured.Unstructured {
	t.Helper()
	result := map[string]interface{}{}

	require.False(t, strings.Contains(config, "\t"), "Yaml %s cannot contain tabs", config)
	require.NoError(t, yaml.Unmarshal([]byte(config), &result), "Could not parse config:\n\n%s\n", config)

	return &unstructured.Unstructured{
		Object: result,
	}
}

func TestWaitForDeletion(t *testing.T) {
	scheme := runtime.NewScheme()
	listMapping := map[schema.GroupVersionResource]string{
		{Group: "group", Version: "version", Resource: "theresource"}:   "TheKindList",
		{Group: "group", Version: "version", Resource: "theresource-1"}: "TheKindList",
		{Group: "group", Version: "version", Resource: "theresource-2"}: "TheKindList",
	}

	tests := []struct {
		name       string
		infos      []*resource.Info
		fakeClient func() *dynamicfakeclient.FakeDynamicClient
		timeout    time.Duration
		uidMap     UIDMap

		expectedErr     string
		validateActions func(t *testing.T, actions []clienttesting.Action)
	}{
		{
			name: "missing on get",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 1 {
					t.Fatal(spew.Sdump(actions))
				}

				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions[0]))
				}
			},
		},
		{
			name:  "handles no infos",
			infos: []*resource.Info{},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClient(scheme)
			},
			timeout:     10 * time.Second,
			expectedErr: errNoMatchingResources.Error(),

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 0 {
					t.Fatal(spew.Sdump(actions))
				}
			},
		},
		{
			name: "uid conflict on get",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					if count == 0 {
						count++
						fakeWatch := watch.NewRaceFreeFake()
						go func() {
							time.Sleep(1 * time.Second)
							fakeWatch.Stop()
						}()
						return true, fakeWatch, nil
					}
					fakeWatch := watch.NewRaceFreeFake()
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,
			uidMap: UIDMap{
				ResourceLocation{Namespace: "ns-foo", Name: "name-foo"}:                                                                               types.UID("some-UID-value"),
				ResourceLocation{GroupResource: schema.GroupResource{Group: "group", Resource: "theresource"}, Namespace: "ns-foo", Name: "name-foo"}: types.UID("some-nonmatching-UID-value"),
			},

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
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				return fakeClient
			},
			timeout: 1 * time.Second,

			expectedErr: "timed out waiting for the condition on theresource/name-foo",
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 3 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("list", "theresource") || actions[1].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[2].Matches("watch", "theresource") || actions[2].(clienttesting.WatchAction).GetWatchRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "delete for existing resource with no timeout",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				return fakeClient
			},
			timeout: 0 * time.Second,

			expectedErr:     "condition not met for theresource/name-foo",
			validateActions: nil,
		},
		{
			name: "delete for nonexisting resource with no timeout",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "thenonexistentresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				return fakeClient
			},
			timeout: 0 * time.Second,

			expectedErr:     "",
			validateActions: nil,
		},
		{
			name: "handles watch close out",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					unstructuredObj := newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")
					unstructuredObj.SetResourceVersion("123")
					unstructuredList := newUnstructuredList(unstructuredObj)
					unstructuredList.SetResourceVersion("234")
					return true, unstructuredList, nil
				})
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					unstructuredObj := newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")
					unstructuredObj.SetResourceVersion("123")
					unstructuredList := newUnstructuredList(unstructuredObj)
					unstructuredList.SetResourceVersion("234")
					return true, unstructuredList, nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					if count == 0 {
						count++
						fakeWatch := watch.NewRaceFreeFake()
						go func() {
							time.Sleep(1 * time.Second)
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

			expectedErr: "timed out waiting for the condition on theresource/name-foo",
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 4 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions[0]))
				}
				if !actions[1].Matches("list", "theresource") || actions[1].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions[1]))
				}
				if !actions[2].Matches("watch", "theresource") || actions[2].(clienttesting.WatchAction).GetWatchRestrictions().ResourceVersion != "234" {
					t.Error(spew.Sdump(actions[2]))
				}
				if !actions[3].Matches("watch", "theresource") || actions[3].(clienttesting.WatchAction).GetWatchRestrictions().ResourceVersion != "234" {
					t.Error(spew.Sdump(actions[3]))
				}
			},
		},
		{
			name: "handles watch delete",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
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
				if len(actions) != 1 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource") || actions[0].(clienttesting.GetAction).GetName() != "name-foo" {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch delete multiple",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource-1"},
					},
					Name:      "name-foo-1",
					Namespace: "ns-foo",
				},
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource-2"},
					},
					Name:      "name-foo-2",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("get", "theresource-1", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo-1"), nil
				})
				fakeClient.PrependReactor("get", "theresource-2", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo-2"), nil
				})
				fakeClient.PrependWatchReactor("theresource-1", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Deleted, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo-1"))
					return true, fakeWatch, nil
				})
				fakeClient.PrependWatchReactor("theresource-2", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Deleted, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo-2"))
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 6 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("get", "theresource-1") || actions[0].(clienttesting.GetAction).GetName() != "name-foo-1" {
					t.Error(spew.Sdump(actions[0]))
				}
				if !actions[1].Matches("list", "theresource-1") || actions[1].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo-1" {
					t.Error(spew.Sdump(actions[1]))
				}
				if !actions[2].Matches("watch", "theresource-1") || actions[2].(clienttesting.WatchAction).GetWatchRestrictions().Fields.String() != "metadata.name=name-foo-1" {
					t.Error(spew.Sdump(actions[2]))
				}
				if !actions[3].Matches("get", "theresource-2") || actions[3].(clienttesting.GetAction).GetName() != "name-foo-2" {
					t.Error(spew.Sdump(actions[3]))
				}
				if !actions[4].Matches("list", "theresource-2") || actions[4].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo-2" {
					t.Error(spew.Sdump(actions[4]))
				}
				if !actions[5].Matches("watch", "theresource-2") || actions[5].(clienttesting.WatchAction).GetWatchRestrictions().Fields.String() != "metadata.name=name-foo-2" {
					t.Error(spew.Sdump(actions[5]))
				}
			},
		},
		{
			name: "ignores watch error",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					if count == 0 {
						fakeWatch.Error(newUnstructuredStatus(&metav1.Status{
							TypeMeta: metav1.TypeMeta{Kind: "Status", APIVersion: "v1"},
							Status:   "Failure",
							Code:     500,
							Message:  "Bad",
						}))
						fakeWatch.Stop()
					} else {
						fakeWatch.Action(watch.Deleted, newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"))
					}
					count++
					return true, fakeWatch, nil
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
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := test.fakeClient()
			o := &WaitOptions{
				ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(test.infos...),
				UIDMap:         test.uidMap,
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

			if test.validateActions != nil {
				test.validateActions(t, fakeClient.Actions())
			}
		})
	}
}

func TestWaitForCondition(t *testing.T) {
	scheme := runtime.NewScheme()
	listMapping := map[schema.GroupVersionResource]string{
		{Group: "group", Version: "version", Resource: "theresource"}: "TheKindList",
	}

	tests := []struct {
		name       string
		infos      []*resource.Info
		fakeClient func() *dynamicfakeclient.FakeDynamicClient
		timeout    time.Duration

		expectedErr     string
		validateActions func(t *testing.T, actions []clienttesting.Action)
	}{
		{
			name: "present on get",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(addCondition(
						newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
						"the-condition", "status-value",
					)), nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") {
					t.Error(spew.Sdump(actions))
				} else if actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				} else if actions[1].(clienttesting.WatchAction).GetWatchRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name:  "handles no infos",
			infos: []*resource.Info{},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClient(scheme)
			},
			timeout:     10 * time.Second,
			expectedErr: errNoMatchingResources.Error(),

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 0 {
					t.Fatal(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles empty object name",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
			},
			timeout:     10 * time.Second,
			expectedErr: "resource name must be provided",

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 0 {
					t.Fatal(spew.Sdump(actions))
				}
			},
		},
		{
			name: "times out",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, addCondition(
						newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
						"some-other-condition", "status-value",
					), nil
				})
				return fakeClient
			},
			timeout: 1 * time.Second,

			expectedErr: `theresource.group "name-foo" not found`,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "for nonexisting resource with no timeout",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "thenonexistingresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				return fakeClient
			},
			timeout: 0 * time.Second,

			expectedErr:     "thenonexistingresource.group \"name-foo\" not found",
			validateActions: nil,
		},
		{
			name: "for existing resource with no timeout",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("get", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				return fakeClient
			},
			timeout: 0 * time.Second,

			expectedErr:     "condition not met for theresource/name-foo",
			validateActions: nil,
		},
		{
			name: "handles watch close out",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					unstructuredObj := newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")
					unstructuredObj.SetResourceVersion("123")
					unstructuredList := newUnstructuredList(unstructuredObj)
					unstructuredList.SetResourceVersion("234")
					return true, unstructuredList, nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					if count == 0 {
						count++
						fakeWatch := watch.NewRaceFreeFake()
						go func() {
							time.Sleep(1 * time.Second)
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

			expectedErr: "timed out waiting for the condition on theresource/name-foo",
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 3 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") || actions[1].(clienttesting.WatchAction).GetWatchRestrictions().ResourceVersion != "234" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[2].Matches("watch", "theresource") || actions[2].(clienttesting.WatchAction).GetWatchRestrictions().ResourceVersion != "234" {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch condition change",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
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
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch created",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
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
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "ignores watch error",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "ns-foo", "name-foo")), nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					if count == 0 {
						fakeWatch.Error(newUnstructuredStatus(&metav1.Status{
							TypeMeta: metav1.TypeMeta{Kind: "Status", APIVersion: "v1"},
							Status:   "Failure",
							Code:     500,
							Message:  "Bad",
						}))
						fakeWatch.Stop()
					} else {
						fakeWatch.Action(watch.Modified, addCondition(
							newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
							"the-condition", "status-value",
						))
					}
					count++
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 4 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
				if !actions[2].Matches("list", "theresource") || actions[2].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[3].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "times out due to stale .status.conditions[0].observedGeneration",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, addConditionWithObservedGeneration(
						newUnstructuredWithGeneration("group/version", "TheKind", "ns-foo", "name-foo", 2),
						"the-condition", "status-value", 1,
					), nil
				})
				return fakeClient
			},
			timeout: 1 * time.Second,

			expectedErr: `theresource.group "name-foo" not found`,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch .status.conditions[0].observedGeneration change",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(addConditionWithObservedGeneration(newUnstructuredWithGeneration("group/version", "TheKind", "ns-foo", "name-foo", 2), "the-condition", "status-value", 1)), nil
				})
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Modified, addConditionWithObservedGeneration(
						newUnstructuredWithGeneration("group/version", "TheKind", "ns-foo", "name-foo", 2),
						"the-condition", "status-value", 2,
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
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "times out due to stale .status.observedGeneration",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					instance := addCondition(
						newUnstructuredWithGeneration("group/version", "TheKind", "ns-foo", "name-foo", 2),
						"the-condition", "status-value")
					unstructured.SetNestedField(instance.Object, int64(1), "status", "observedGeneration")
					return true, instance, nil
				})
				return fakeClient
			},
			timeout: 1 * time.Second,

			expectedErr: `theresource.group "name-foo" not found`,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch .status.observedGeneration change",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "name-foo",
					Namespace: "ns-foo",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					instance := addCondition(
						newUnstructuredWithGeneration("group/version", "TheKind", "ns-foo", "name-foo", 2),
						"the-condition", "status-value")
					unstructured.SetNestedField(instance.Object, int64(1), "status", "observedGeneration")
					return true, newUnstructuredList(instance), nil
				})
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					instance := addCondition(
						newUnstructuredWithGeneration("group/version", "TheKind", "ns-foo", "name-foo", 2),
						"the-condition", "status-value")
					unstructured.SetNestedField(instance.Object, int64(2), "status", "observedGeneration")
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Modified, instance)
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout: 10 * time.Second,

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=name-foo" {
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
				ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(test.infos...),
				DynamicClient:  fakeClient,
				Timeout:        test.timeout,

				Printer:     printers.NewDiscardingPrinter(),
				ConditionFn: ConditionalWait{conditionName: "the-condition", conditionStatus: "status-value", errOut: ioutil.Discard}.IsConditionMet,
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

			if test.validateActions != nil {
				test.validateActions(t, fakeClient.Actions())
			}
		})
	}
}

func TestWaitForDeletionIgnoreNotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	listMapping := map[schema.GroupVersionResource]string{
		{Group: "group", Version: "version", Resource: "theresource"}: "TheKindList",
	}
	infos := []*resource.Info{}
	fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)

	o := &WaitOptions{
		ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(infos...),
		DynamicClient:  fakeClient,
		Printer:        printers.NewDiscardingPrinter(),
		ConditionFn:    IsDeleted,
		IOStreams:      genericclioptions.NewTestIOStreamsDiscard(),
		ForCondition:   "delete",
	}
	err := o.RunWait()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestWaitForDifferentJSONPathCondition will run tests on different types of
// JSONPath expression to check the JSONPath can be parsed correctly from a Pod Yaml
// and check if the comparison returns as expected.
func TestWaitForDifferentJSONPathExpression(t *testing.T) {
	scheme := runtime.NewScheme()
	listMapping := map[schema.GroupVersionResource]string{
		{Group: "group", Version: "version", Resource: "theresource"}: "TheKindList",
	}
	listReactionfunc := func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
		return true, newUnstructuredList(createUnstructured(t, podYAML)), nil
	}
	infos := []*resource.Info{
		{
			Mapping: &meta.RESTMapping{
				Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
			},
			Name:      "foo-b6699dcfb-rnv7t",
			Namespace: "default",
		},
	}

	tests := []struct {
		name         string
		fakeClient   func() *dynamicfakeclient.FakeDynamicClient
		jsonPathExp  string
		jsonPathCond string

		expectedErr string
	}{
		{
			name: "JSONPath entry not exist",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.foo.bar}",
			jsonPathCond: "baz",

			expectedErr: "foo is not found",
		},
		{
			name: "compare boolean JSONPath entry",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.status.containerStatuses[0].ready}",
			jsonPathCond: "true",

			expectedErr: None,
		},
		{
			name: "compare boolean JSONPath entry wrong value",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.status.containerStatuses[0].ready}",
			jsonPathCond: "false",

			expectedErr: "timed out waiting for the condition on theresource/foo-b6699dcfb-rnv7t",
		},
		{
			name: "compare integer JSONPath entry",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.spec.containers[0].ports[0].containerPort}",
			jsonPathCond: "80",

			expectedErr: None,
		},
		{
			name: "compare integer JSONPath entry wrong value",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.spec.containers[0].ports[0].containerPort}",
			jsonPathCond: "81",

			expectedErr: "timed out waiting for the condition on theresource/foo-b6699dcfb-rnv7t",
		},
		{
			name: "compare string JSONPath entry",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.spec.nodeName}",
			jsonPathCond: "knode0",

			expectedErr: None,
		},
		{
			name: "compare string JSONPath entry wrong value",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.spec.nodeName}",
			jsonPathCond: "kmaster",

			expectedErr: "timed out waiting for the condition on theresource/foo-b6699dcfb-rnv7t",
		},
		{
			name: "matches more than one value",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.status.conditions[*]}",
			jsonPathCond: "foo",

			expectedErr: "given jsonpath expression matches more than one value",
		},
		{
			name: "matches more than one list",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{range .status.conditions[*]}[{.status}] {end}",
			jsonPathCond: "foo",

			expectedErr: "given jsonpath expression matches more than one list",
		},
		{
			name: "unsupported type []interface{}",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", listReactionfunc)
				return fakeClient
			},
			jsonPathExp:  "{.status.conditions}",
			jsonPathCond: "True",

			expectedErr: "jsonpath leads to a nested object or list which is not supported",
		},
		{
			name: "unsupported type map[string]interface{}",
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(createUnstructured(t, podYAML)), nil
				})
				return fakeClient
			},
			jsonPathExp:  "{.spec}",
			jsonPathCond: "foo",

			expectedErr: "jsonpath leads to a nested object or list which is not supported",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := test.fakeClient()
			j, _ := newJSONPathParser(test.jsonPathExp)
			o := &WaitOptions{
				ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(infos...),
				DynamicClient:  fakeClient,
				Timeout:        1 * time.Second,

				Printer: printers.NewDiscardingPrinter(),
				ConditionFn: JSONPathWait{
					jsonPathCondition: test.jsonPathCond,
					jsonPathParser:    j,
					errOut:            ioutil.Discard}.IsJSONPathConditionMet,
				IOStreams: genericclioptions.NewTestIOStreamsDiscard(),
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
		})
	}
}

// TestWaitForJSONPathCondition will run tests to check whether
// the List actions and Watch actions match what we expected
func TestWaitForJSONPathCondition(t *testing.T) {
	scheme := runtime.NewScheme()
	listMapping := map[schema.GroupVersionResource]string{
		{Group: "group", Version: "version", Resource: "theresource"}: "TheKindList",
	}

	tests := []struct {
		name         string
		infos        []*resource.Info
		fakeClient   func() *dynamicfakeclient.FakeDynamicClient
		timeout      time.Duration
		jsonPathExp  string
		jsonPathCond string

		expectedErr     string
		validateActions func(t *testing.T, actions []clienttesting.Action)
	}{
		{
			name: "present on get",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "foo-b6699dcfb-rnv7t",
					Namespace: "default",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(
						createUnstructured(t, podYAML)), nil
				})
				return fakeClient
			},
			timeout:      3 * time.Second,
			jsonPathExp:  "{.metadata.name}",
			jsonPathCond: "foo-b6699dcfb-rnv7t",

			expectedErr: None,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") ||
					actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=foo-b6699dcfb-rnv7t" &&
						actions[1].(clienttesting.WatchAction).GetWatchRestrictions().Fields.String() != "metadata.name=foo-b6699dcfb-rnv7t" {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name:  "handles no infos",
			infos: []*resource.Info{},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClient(scheme)
			},
			timeout:     10 * time.Second,
			expectedErr: errNoMatchingResources.Error(),

			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 0 {
					t.Fatal(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles empty object name",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Namespace: "default",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				return dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
			},
			timeout: 10 * time.Second,

			expectedErr: "resource name must be provided",
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 0 {
					t.Fatal(spew.Sdump(actions))
				}
			},
		},
		{
			name: "times out",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "foo-b6699dcfb-rnv7t",
					Namespace: "default",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, createUnstructured(t, podYAML), nil
				})
				return fakeClient
			},
			timeout: 1 * time.Second,

			expectedErr: `theresource.group "foo-b6699dcfb-rnv7t" not found`,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=foo-b6699dcfb-rnv7t" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch close out",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "foo-b6699dcfb-rnv7t",
					Namespace: "default",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					unstructuredObj := createUnstructured(t, podYAML)
					unstructuredObj.SetResourceVersion("123")
					unstructuredList := newUnstructuredList(unstructuredObj)
					unstructuredList.SetResourceVersion("234")
					return true, unstructuredList, nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					if count == 0 {
						count++
						fakeWatch := watch.NewRaceFreeFake()
						go func() {
							time.Sleep(1 * time.Second)
							fakeWatch.Stop()
						}()
						return true, fakeWatch, nil
					}
					fakeWatch := watch.NewRaceFreeFake()
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout:      3 * time.Second,
			jsonPathExp:  "{.metadata.name}",
			jsonPathCond: "foo", // use incorrect name so it'll keep waiting

			expectedErr: "timed out waiting for the condition on theresource/foo-b6699dcfb-rnv7t",
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 3 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=foo-b6699dcfb-rnv7t" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") || actions[1].(clienttesting.WatchAction).GetWatchRestrictions().ResourceVersion != "234" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[2].Matches("watch", "theresource") || actions[2].(clienttesting.WatchAction).GetWatchRestrictions().ResourceVersion != "234" {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch condition change",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "foo-b6699dcfb-rnv7t",
					Namespace: "default",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					unstructuredObj := createUnstructured(t, podYAML)
					unstructuredObj.SetName("foo")
					return true, newUnstructuredList(), nil
				})
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Modified, createUnstructured(t, podYAML))
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout:      10 * time.Second,
			jsonPathExp:  "{.metadata.name}",
			jsonPathCond: "foo-b6699dcfb-rnv7t",

			expectedErr: None,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=foo-b6699dcfb-rnv7t" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "handles watch created",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "foo-b6699dcfb-rnv7t",
					Namespace: "default",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					fakeWatch.Action(watch.Added, createUnstructured(t, podYAML))
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout:      10 * time.Second,
			jsonPathExp:  "{.spec.containers[0].image}",
			jsonPathCond: "nginx",

			expectedErr: None,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=foo-b6699dcfb-rnv7t" {
					t.Error(spew.Sdump(actions))
				}
				if !actions[1].Matches("watch", "theresource") {
					t.Error(spew.Sdump(actions))
				}
			},
		},
		{
			name: "ignores watch error",
			infos: []*resource.Info{
				{
					Mapping: &meta.RESTMapping{
						Resource: schema.GroupVersionResource{Group: "group", Version: "version", Resource: "theresource"},
					},
					Name:      "foo-b6699dcfb-rnv7t",
					Namespace: "default",
				},
			},
			fakeClient: func() *dynamicfakeclient.FakeDynamicClient {
				fakeClient := dynamicfakeclient.NewSimpleDynamicClientWithCustomListKinds(scheme, listMapping)
				fakeClient.PrependReactor("list", "theresource", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, newUnstructuredList(newUnstructured("group/version", "TheKind", "default", "foo-b6699dcfb-rnv7t")), nil
				})
				count := 0
				fakeClient.PrependWatchReactor("theresource", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
					fakeWatch := watch.NewRaceFreeFake()
					if count == 0 {
						fakeWatch.Error(newUnstructuredStatus(&metav1.Status{
							TypeMeta: metav1.TypeMeta{Kind: "Status", APIVersion: "v1"},
							Status:   "Failure",
							Code:     500,
							Message:  "Bad",
						}))
						fakeWatch.Stop()
					} else {
						fakeWatch.Action(watch.Modified, createUnstructured(t, podYAML))
					}
					count++
					return true, fakeWatch, nil
				})
				return fakeClient
			},
			timeout:      10 * time.Second,
			jsonPathExp:  "{.metadata.name}",
			jsonPathCond: "foo-b6699dcfb-rnv7t",

			expectedErr: None,
			validateActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 2 {
					t.Fatal(spew.Sdump(actions))
				}
				if !actions[0].Matches("list", "theresource") || actions[0].(clienttesting.ListAction).GetListRestrictions().Fields.String() != "metadata.name=foo-b6699dcfb-rnv7t" {
					t.Error(spew.Sdump(actions[0]))
				}
				if !actions[1].Matches("watch", "theresource") || actions[1].(clienttesting.WatchAction).GetWatchRestrictions().Fields.String() != "metadata.name=foo-b6699dcfb-rnv7t" {
					t.Error(spew.Sdump(actions[1]))
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := test.fakeClient()
			j, _ := newJSONPathParser(test.jsonPathExp)
			o := &WaitOptions{
				ResourceFinder: genericclioptions.NewSimpleFakeResourceFinder(test.infos...),
				DynamicClient:  fakeClient,
				Timeout:        test.timeout,

				Printer: printers.NewDiscardingPrinter(),
				ConditionFn: JSONPathWait{
					jsonPathCondition: test.jsonPathCond,
					jsonPathParser:    j, errOut: ioutil.Discard}.IsJSONPathConditionMet,
				IOStreams: genericclioptions.NewTestIOStreamsDiscard(),
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
