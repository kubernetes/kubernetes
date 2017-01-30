/*
Copyright 2017 The Kubernetes Authors.

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

package ttl

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/legacylisters"

	"github.com/stretchr/testify/assert"
)

func TestPatchNode(t *testing.T) {
	testCases := []struct {
		node       *v1.Node
		ttlSeconds int
		patch      string
	}{
		{
			node:       &v1.Node{},
			ttlSeconds: 0,
			patch:      "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"0\"}}}",
		},
		{
			node:       &v1.Node{},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name"}},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}}},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"node.kubernetes.io/ttl":"0"}}},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"node.kubernetes.io/ttl":"0", "a": "b"}}},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"node.kubernetes.io/ttl":"10", "a": "b"}}},
			ttlSeconds: 10,
			patch:      "{}",
		},
	}

	for i, testCase := range testCases {
		fakeClient := &fake.Clientset{}
		ttlController := &TTLController{
			kubeClient: fakeClient,
		}
		err := ttlController.patchNodeWithAnnotation(testCase.node, ObjectTTLAnnotation, testCase.ttlSeconds)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		actions := fakeClient.Actions()
		assert.Equal(t, 1, len(actions), "unexpected actions: %#v", actions)
		patchAction := actions[0].(core.PatchActionImpl)
		if testCase.patch != string(patchAction.Patch) {
			t.Errorf("%d: unexpected patch: %s", i, string(patchAction.Patch))
		}
	}
}

func TestUpdateNodeIfNeeded(t *testing.T) {
	testCases := []struct {
		node      *v1.Node
		nodeCount int32
		patch     string
	}{
		{
			node:      &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name"}},
			nodeCount: 10,
			patch:     "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"0\"}}}",
		},
		{
			node:      &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name"}},
			nodeCount: 204,
			patch:     "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"30\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name", Annotations: map[string]string{"node.kubernetes.io/ttl":"0"}}},
			nodeCount:  1081,
			patch:      "{\"metadata\":{\"annotations\":{\"node.kubernetes.io/ttl\":\"60\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name", Annotations: map[string]string{"node.kubernetes.io/ttl":"60"}}},
			nodeCount:  1081,
			patch:      "",
		},
	}

	for i, testCase := range testCases {
		fakeClient := &fake.Clientset{}
		nodeStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
		nodeStore.Add(testCase.node)
		ttlController := &TTLController{
			kubeClient: fakeClient,
			nodeStore:  listers.StoreToNodeLister{nodeStore},
			nodeCount:  testCase.nodeCount,
		}
		if err := ttlController.updateNodeIfNeeded(testCase.node.Name); err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		actions := fakeClient.Actions()
		if testCase.patch == "" {
			assert.Equal(t, 0, len(actions), "unexpected actions: %#v", actions)
		} else {
			assert.Equal(t, 1, len(actions), "unexpected actions: %#v", actions)
			patchAction := actions[0].(core.PatchActionImpl)
			if testCase.patch != string(patchAction.Patch) {
				t.Errorf("%d: unexpected patch: %s", i, string(patchAction.Patch))
			}
		}
	}
}
