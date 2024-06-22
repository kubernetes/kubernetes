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
	"context"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	listers "k8s.io/client-go/listers/core/v1"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2/ktesting"

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
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"0\"}}}",
		},
		{
			node:       &v1.Node{},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name"}},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}}},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"node.alpha.kubernetes.io/ttl": "0"}}},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"node.alpha.kubernetes.io/ttl": "0", "a": "b"}}},
			ttlSeconds: 10,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"10\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"node.alpha.kubernetes.io/ttl": "10", "a": "b"}}},
			ttlSeconds: 10,
			patch:      "{}",
		},
	}

	for i, testCase := range testCases {
		fakeClient := &fake.Clientset{}
		ttlController := &Controller{
			kubeClient: fakeClient,
		}
		err := ttlController.patchNodeWithAnnotation(context.TODO(), testCase.node, v1.ObjectTTLAnnotationKey, testCase.ttlSeconds)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		actions := fakeClient.Actions()
		assert.Equal(t, 1, len(actions), "unexpected actions: %#v", actions)
		patchAction := actions[0].(core.PatchActionImpl)
		assert.Equal(t, testCase.patch, string(patchAction.Patch), "%d: unexpected patch: %s", i, string(patchAction.Patch))
	}
}

func TestUpdateNodeIfNeeded(t *testing.T) {
	testCases := []struct {
		node       *v1.Node
		desiredTTL int
		patch      string
	}{
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name"}},
			desiredTTL: 0,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"0\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name"}},
			desiredTTL: 15,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"15\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name"}},
			desiredTTL: 30,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"30\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name", Annotations: map[string]string{"node.alpha.kubernetes.io/ttl": "0"}}},
			desiredTTL: 60,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"60\"}}}",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name", Annotations: map[string]string{"node.alpha.kubernetes.io/ttl": "60"}}},
			desiredTTL: 60,
			patch:      "",
		},
		{
			node:       &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name", Annotations: map[string]string{"node.alpha.kubernetes.io/ttl": "60"}}},
			desiredTTL: 30,
			patch:      "{\"metadata\":{\"annotations\":{\"node.alpha.kubernetes.io/ttl\":\"30\"}}}",
		},
	}

	for i, testCase := range testCases {
		fakeClient := &fake.Clientset{}
		nodeStore := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
		nodeStore.Add(testCase.node)
		ttlController := &Controller{
			kubeClient:        fakeClient,
			nodeStore:         listers.NewNodeLister(nodeStore),
			desiredTTLSeconds: testCase.desiredTTL,
		}
		if err := ttlController.updateNodeIfNeeded(context.TODO(), testCase.node.Name); err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		actions := fakeClient.Actions()
		if testCase.patch == "" {
			assert.Equal(t, 0, len(actions), "unexpected actions: %#v", actions)
		} else {
			assert.Equal(t, 1, len(actions), "unexpected actions: %#v", actions)
			patchAction := actions[0].(core.PatchActionImpl)
			assert.Equal(t, testCase.patch, string(patchAction.Patch), "%d: unexpected patch: %s", i, string(patchAction.Patch))
		}
	}
}

func TestDesiredTTL(t *testing.T) {
	testCases := []struct {
		addNode      bool
		deleteNode   bool
		nodeCount    int
		desiredTTL   int
		boundaryStep int
		expectedTTL  int
	}{
		{
			addNode:      true,
			nodeCount:    0,
			desiredTTL:   0,
			boundaryStep: 0,
			expectedTTL:  0,
		},
		{
			addNode:      true,
			nodeCount:    99,
			desiredTTL:   0,
			boundaryStep: 0,
			expectedTTL:  0,
		},
		{
			addNode:      true,
			nodeCount:    100,
			desiredTTL:   0,
			boundaryStep: 0,
			expectedTTL:  15,
		},
		{
			deleteNode:   true,
			nodeCount:    101,
			desiredTTL:   15,
			boundaryStep: 1,
			expectedTTL:  15,
		},
		{
			deleteNode:   true,
			nodeCount:    91,
			desiredTTL:   15,
			boundaryStep: 1,
			expectedTTL:  15,
		},
		{
			addNode:      true,
			nodeCount:    91,
			desiredTTL:   15,
			boundaryStep: 1,
			expectedTTL:  15,
		},
		{
			deleteNode:   true,
			nodeCount:    90,
			desiredTTL:   15,
			boundaryStep: 1,
			expectedTTL:  0,
		},
		{
			deleteNode:   true,
			nodeCount:    1800,
			desiredTTL:   300,
			boundaryStep: 4,
			expectedTTL:  60,
		},
		{
			deleteNode:   true,
			nodeCount:    10000,
			desiredTTL:   300,
			boundaryStep: 4,
			expectedTTL:  300,
		},
	}

	for i, testCase := range testCases {
		ttlController := &Controller{
			queue:             workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
			nodeCount:         testCase.nodeCount,
			desiredTTLSeconds: testCase.desiredTTL,
			boundaryStep:      testCase.boundaryStep,
		}
		if testCase.addNode {
			logger, _ := ktesting.NewTestContext(t)
			ttlController.addNode(logger, &v1.Node{})
		}
		if testCase.deleteNode {
			ttlController.deleteNode(&v1.Node{})
		}
		assert.Equal(t, testCase.expectedTTL, ttlController.getDesiredTTLSeconds(),
			"%d: unexpected ttl: %d", i, ttlController.getDesiredTTLSeconds())
	}
}
