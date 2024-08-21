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

package helpers

import (
	"context"
	"errors"
	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/utils/ktesting"
	"reflect"
	"testing"
)

func TestTaintExists(t *testing.T) {
	testingTaints := []v1.Taint{
		{
			Key:    "foo_1",
			Value:  "bar_1",
			Effect: v1.TaintEffectNoExecute,
		},
		{
			Key:    "foo_2",
			Value:  "bar_2",
			Effect: v1.TaintEffectNoSchedule,
		},
	}

	cases := []struct {
		name           string
		taintToFind    *v1.Taint
		matchingFn     matchTaintFunc
		expectedResult bool
	}{
		{
			name:           "taint exists",
			taintToFind:    &v1.Taint{Key: "foo_1", Value: "bar_1", Effect: v1.TaintEffectNoExecute},
			matchingFn:     (*v1.Taint).MatchTaint,
			expectedResult: true,
		},
		{
			name:           "different key",
			taintToFind:    &v1.Taint{Key: "no_such_key", Value: "bar_1", Effect: v1.TaintEffectNoExecute},
			matchingFn:     (*v1.Taint).MatchTaint,
			expectedResult: false,
		},
		{
			name:           "different effect",
			taintToFind:    &v1.Taint{Key: "foo_1", Value: "bar_1", Effect: v1.TaintEffectNoSchedule},
			matchingFn:     (*v1.Taint).MatchTaint,
			expectedResult: false,
		},
		{
			name:           "same key, match by key",
			taintToFind:    &v1.Taint{Key: "foo_1", Value: "bar_1", Effect: v1.TaintEffectNoSchedule},
			matchingFn:     (*v1.Taint).MatchTaintByKey,
			expectedResult: true,
		},
		{
			name:           "different effect, match by key",
			taintToFind:    &v1.Taint{Key: "foo_1", Value: "bar_1", Effect: v1.TaintEffectNoSchedule},
			matchingFn:     (*v1.Taint).MatchTaintByKey,
			expectedResult: true,
		},
		{
			name:           "different key, match by key",
			taintToFind:    &v1.Taint{Key: "no_such_key", Value: "bar_1", Effect: v1.TaintEffectNoSchedule},
			matchingFn:     (*v1.Taint).MatchTaintByKey,
			expectedResult: false,
		},
	}

	for _, c := range cases {
		result := taintExists(testingTaints, c.taintToFind, c.matchingFn)

		if result != c.expectedResult {
			t.Errorf("[%s] unexpected results: %v", c.name, result)
			continue
		}
	}
}

func TestRemoveTaint(t *testing.T) {
	cases := []struct {
		name           string
		node           *v1.Node
		taintToRemove  *v1.Taint
		matchingFn     matchTaintFunc
		expectedTaints []v1.Taint
		expectedResult bool
	}{
		{
			name: "remove taint unsuccessfully",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "foo",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo_1",
				Effect: v1.TaintEffectNoSchedule,
			},
			matchingFn: (*v1.Taint).MatchTaint,
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "remove taint successfully",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "foo",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo",
				Effect: v1.TaintEffectNoSchedule,
			},
			matchingFn:     (*v1.Taint).MatchTaint,
			expectedTaints: []v1.Taint{},
			expectedResult: true,
		},
		{
			name: "remove taint from node with no taint",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo",
				Effect: v1.TaintEffectNoSchedule,
			},
			matchingFn:     (*v1.Taint).MatchTaint,
			expectedTaints: []v1.Taint{},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		newNode, result, err := removeMatchingTaint(c.node, c.taintToRemove, c.matchingFn)
		if err != nil {
			t.Errorf("[%s] should not raise error but got: %v", c.name, err)
		}
		if result != c.expectedResult {
			t.Errorf("[%s] should return %t, but got: %t", c.name, c.expectedResult, result)
		}
		if !reflect.DeepEqual(newNode.Spec.Taints, c.expectedTaints) {
			t.Errorf("[%s] the new node object should have taints %v, but got: %v", c.name, c.expectedTaints, newNode.Spec.Taints)
		}
	}
}

func TestDeleteTaint(t *testing.T) {
	cases := []struct {
		name           string
		taints         []v1.Taint
		taintToDelete  *v1.Taint
		matchingFn     matchTaintFunc
		expectedTaints []v1.Taint
		expectedResult bool
	}{
		{
			name: "delete taint with different name",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete: &v1.Taint{Key: "foo_1", Effect: v1.TaintEffectNoSchedule},
			matchingFn:    (*v1.Taint).MatchTaint,
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "delete taint with different effect",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete: &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoExecute},
			matchingFn:    (*v1.Taint).MatchTaint,
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "delete taint successfully",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete:  &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoSchedule},
			matchingFn:     (*v1.Taint).MatchTaint,
			expectedTaints: []v1.Taint{},
			expectedResult: true,
		},
		{
			name:           "delete taint from empty taint array",
			taints:         []v1.Taint{},
			taintToDelete:  &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoSchedule},
			matchingFn:     (*v1.Taint).MatchTaint,
			expectedTaints: []v1.Taint{},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		taints, result := deleteMatchingTaint(c.taints, c.taintToDelete, c.matchingFn)
		if result != c.expectedResult {
			t.Errorf("[%s] should return %t, but got: %t", c.name, c.expectedResult, result)
		}
		if !reflect.DeepEqual(taints, c.expectedTaints) {
			t.Errorf("[%s] the result taints should be %v, but got: %v", c.name, c.expectedTaints, taints)
		}
	}
}

func ready(_ context.Context, _ clientset.Interface, _, _ string) error {
	return nil
}

func notReady(_ context.Context, _ clientset.Interface, _, _ string) error {
	return errors.New("not ready")
}

func TestRemoveNotReadyTaint(t *testing.T) {
	cases := []struct {
		name           string
		c              kubernetes.Interface
		nodeName       string
		componentName  string
		readyFn        ComponentReadyFunc
		expectedTaints []v1.Taint
		shouldErr      bool
	}{
		{
			name: "remove single taint",
			c: fake.NewSimpleClientset(&v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec:       v1.NodeSpec{Taints: []v1.Taint{{Key: "driver/agent-not-ready", Effect: v1.TaintEffectNoExecute}}},
			}),
			nodeName:       "node1",
			componentName:  "driver",
			expectedTaints: nil,
			readyFn:        ready,
			shouldErr:      false,
		},
		{
			name: "remove several taints",
			c: fake.NewSimpleClientset(&v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec: v1.NodeSpec{Taints: []v1.Taint{
					{Key: "driver/agent-not-ready", Effect: v1.TaintEffectNoExecute},
					{Key: "driver/agent-not-ready", Effect: v1.TaintEffectNoSchedule},
				}},
			}),
			nodeName:       "node1",
			componentName:  "driver",
			expectedTaints: nil,
			readyFn:        ready,
			shouldErr:      false,
		},
		{
			name: "remove only matching taints",
			c: fake.NewSimpleClientset(&v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec: v1.NodeSpec{Taints: []v1.Taint{
					{Key: "driver/agent-not-ready", Effect: v1.TaintEffectNoExecute},
					{Key: "foo", Effect: v1.TaintEffectNoSchedule},
				}},
			}),
			nodeName:       "node1",
			componentName:  "driver",
			expectedTaints: []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}},
			readyFn:        ready,
			shouldErr:      false,
		},
		{
			name: "don't remove taint if not ready",
			c: fake.NewSimpleClientset(&v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec:       v1.NodeSpec{Taints: []v1.Taint{{Key: "driver/agent-not-ready", Effect: v1.TaintEffectNoExecute}}},
			}),
			nodeName:       "node1",
			componentName:  "driver",
			expectedTaints: []v1.Taint{{Key: "driver/agent-not-ready", Effect: v1.TaintEffectNoExecute}},
			readyFn:        notReady,
			shouldErr:      true,
		},
		{
			name: "noop if taint doesn't exist",
			c: fake.NewSimpleClientset(&v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec:       v1.NodeSpec{},
			}),
			nodeName:       "node1",
			componentName:  "driver",
			expectedTaints: nil,
			readyFn:        ready,
			shouldErr:      false,
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			err := RemoveNotReadyTaint(tt.c, tt.nodeName, tt.componentName, tt.readyFn)
			if tt.shouldErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			actualNode, err := tt.c.CoreV1().Nodes().Get(ctx, tt.nodeName, metav1.GetOptions{})
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedTaints, actualNode.Spec.Taints)
		})
	}
}
