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

package v1

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// TestCompatibility verifies that the types in extender/v1 can be successfully encoded to json and decoded back, even when lowercased,
// since these types were written around JSON tags and we need to enforce consistency on them now.
// @TODO(88634): v2 of these types should be defined with proper JSON tags to enforce field casing to a single approach
func TestCompatibility(t *testing.T) {
	testcases := []struct {
		emptyObj   interface{}
		obj        interface{}
		expectJSON string
	}{
		{
			emptyObj: &ExtenderPreemptionResult{},
			obj: &ExtenderPreemptionResult{
				NodeNameToMetaVictims: map[string]*MetaVictims{"foo": {Pods: []*MetaPod{{UID: "myuid"}}, NumPDBViolations: 1}},
			},
			expectJSON: `{"NodeNameToMetaVictims":{"foo":{"Pods":[{"UID":"myuid"}],"NumPDBViolations":1}}}`,
		},
		{
			emptyObj: &ExtenderPreemptionArgs{},
			obj: &ExtenderPreemptionArgs{
				Pod:                   &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "podname"}},
				NodeNameToVictims:     map[string]*Victims{"foo": {Pods: []*v1.Pod{&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "podname"}}}, NumPDBViolations: 1}},
				NodeNameToMetaVictims: map[string]*MetaVictims{"foo": {Pods: []*MetaPod{{UID: "myuid"}}, NumPDBViolations: 1}},
			},
			expectJSON: `{"Pod":{"metadata":{"name":"podname","creationTimestamp":null},"spec":{"containers":null},"status":{}},"NodeNameToVictims":{"foo":{"Pods":[{"metadata":{"name":"podname","creationTimestamp":null},"spec":{"containers":null},"status":{}}],"NumPDBViolations":1}},"NodeNameToMetaVictims":{"foo":{"Pods":[{"UID":"myuid"}],"NumPDBViolations":1}}}`,
		},
		{
			emptyObj: &ExtenderArgs{},
			obj: &ExtenderArgs{
				Pod:       &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "podname"}},
				Nodes:     &corev1.NodeList{Items: []corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "nodename"}}}},
				NodeNames: &[]string{"node1"},
			},
			expectJSON: `{"Pod":{"metadata":{"name":"podname","creationTimestamp":null},"spec":{"containers":null},"status":{}},"Nodes":{"metadata":{},"items":[{"metadata":{"name":"nodename","creationTimestamp":null},"spec":{},"status":{"daemonEndpoints":{"kubeletEndpoint":{"Port":0}},"nodeInfo":{"machineID":"","systemUUID":"","bootID":"","kernelVersion":"","osImage":"","containerRuntimeVersion":"","kubeletVersion":"","kubeProxyVersion":"","operatingSystem":"","architecture":""}}}]},"NodeNames":["node1"]}`,
		},
		{
			emptyObj: &ExtenderFilterResult{},
			obj: &ExtenderFilterResult{
				Nodes:       &corev1.NodeList{Items: []corev1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "nodename"}}}},
				NodeNames:   &[]string{"node1"},
				FailedNodes: FailedNodesMap{"foo": "bar"},
				Error:       "myerror",
			},
			expectJSON: `{"Nodes":{"metadata":{},"items":[{"metadata":{"name":"nodename","creationTimestamp":null},"spec":{},"status":{"daemonEndpoints":{"kubeletEndpoint":{"Port":0}},"nodeInfo":{"machineID":"","systemUUID":"","bootID":"","kernelVersion":"","osImage":"","containerRuntimeVersion":"","kubeletVersion":"","kubeProxyVersion":"","operatingSystem":"","architecture":""}}}]},"NodeNames":["node1"],"FailedNodes":{"foo":"bar"},"Error":"myerror"}`,
		},
		{
			emptyObj: &ExtenderBindingArgs{},
			obj: &ExtenderBindingArgs{
				PodName:      "mypodname",
				PodNamespace: "mypodnamespace",
				PodUID:       types.UID("mypoduid"),
				Node:         "mynode",
			},
			expectJSON: `{"PodName":"mypodname","PodNamespace":"mypodnamespace","PodUID":"mypoduid","Node":"mynode"}`,
		},
		{
			emptyObj:   &ExtenderBindingResult{},
			obj:        &ExtenderBindingResult{Error: "myerror"},
			expectJSON: `{"Error":"myerror"}`,
		},
		{
			emptyObj:   &HostPriority{},
			obj:        &HostPriority{Host: "myhost", Score: 1},
			expectJSON: `{"Host":"myhost","Score":1}`,
		},
	}

	for _, tc := range testcases {
		t.Run(reflect.TypeOf(tc.obj).String(), func(t *testing.T) {
			data, err := json.Marshal(tc.obj)
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != tc.expectJSON {
				t.Fatalf("expected %s, got %s", tc.expectJSON, string(data))
			}
			if err := json.Unmarshal([]byte(strings.ToLower(string(data))), tc.emptyObj); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(tc.emptyObj, tc.obj) {
				t.Fatalf("round-tripped case-insensitive diff: %s", cmp.Diff(tc.obj, tc.emptyObj))
			}
		})
	}
}
