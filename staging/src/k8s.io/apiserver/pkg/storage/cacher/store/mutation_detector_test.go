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

package store

import (
	"context"
	"fmt"
	goruntime "runtime"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/example"
)

func TestMutationDetector(t *testing.T) {
	for _, test := range []struct {
		name             string
		object           runtime.Object
		mutateObjectFn   func(runtime.Object)
		wantPanicMessage string
	}{
		{
			name:   "unchanged object",
			object: makePodFn("ns", "target", "42"),
		},
		{
			name:   "replacing a nil map with an empty map is a mutation",
			object: makePodFn("ns", "target", "42"),
			mutateObjectFn: func(obj runtime.Object) {
				obj.(*example.Pod).Annotations = map[string]string{}
			},
			wantPanicMessage: mutationPanicMessage("*example.Pod", "ns/target", "42"),
		},
		{
			name: "detects a change to an existing label",
			object: func() *example.Pod {
				pod := makePodFn("ns", "target", "42")
				pod.Labels = map[string]string{"team": "blue"}
				return pod
			}(),
			mutateObjectFn: func(obj runtime.Object) {
				obj.(*example.Pod).Labels["team"] = "red"
			},
			wantPanicMessage: mutationPanicMessage("*example.Pod", "ns/target", "42"),
		},
		{
			name: "reports the original name and RV of a cluster-scoped object",
			object: &unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "example.com/v1",
				"kind":       "Widget",
				"metadata":   map[string]any{"name": "target", "resourceVersion": "42"},
			}},
			mutateObjectFn: func(object runtime.Object) {
				obj := object.(*unstructured.Unstructured)
				obj.SetName("changed")
				obj.SetResourceVersion("43")
			},
			wantPanicMessage: mutationPanicMessage("*unstructured.Unstructured", "target", "42"),
		},
		{
			name:   "uses the cache key when the object has no metadata",
			object: &mockObject{val: "original"},
			mutateObjectFn: func(obj runtime.Object) {
				obj.(*mockObject).val = "changed"
			},
			wantPanicMessage: `watch cache mutation detected for *store.mockObject with cache key "/objects/target"`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			detector := &mutationDetector{}
			elem := &Element{Key: "/objects/target", Object: test.object}
			detector.track(elem)
			if test.mutateObjectFn != nil {
				test.mutateObjectFn(elem.Object)
			}

			checkMutations := func() { detector.checkMutations(t.Context()) }
			if test.wantPanicMessage == "" {
				require.NotPanics(t, checkMutations)
			} else {
				require.PanicsWithValue(t, test.wantPanicMessage, checkMutations)
			}
			// The detector only holds a weak reference. Keep the element live through the scan.
			goruntime.KeepAlive(elem)
		})
	}
}

func TestMutationDetectorRepeatedTrackingPreservesBaseline(t *testing.T) {
	pod := makePodFn("ns", "target", "42")
	elem := &Element{Key: "/objects/target", Object: pod}
	detector := &mutationDetector{}
	detector.track(elem)
	pod.ResourceVersion = "43"
	// Replacing the original baseline on the second call would hide the mutation.
	detector.track(elem)

	require.PanicsWithValue(t, mutationPanicMessage("*example.Pod", "ns/target", "42"), func() {
		detector.checkMutations(t.Context())
	})
	// The detector only holds a weak reference. Keep the element live through the scan.
	goruntime.KeepAlive(elem)
}

func TestMutationDetectorStopsScanOnCancellation(t *testing.T) {
	pod := makePodFn("ns", "target", "42")
	elem := &Element{Key: "/objects/target", Object: pod}
	detector := &mutationDetector{}
	detector.track(elem)
	pod.ResourceVersion = "43"

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	require.NotPanics(t, func() { detector.checkMutations(ctx) })
	goruntime.KeepAlive(elem)
}

func makePodFn(namespace, name, rv string) *example.Pod {
	return &example.Pod{ObjectMeta: metav1.ObjectMeta{
		Namespace:       namespace,
		Name:            name,
		ResourceVersion: rv,
	}}
}

func mutationPanicMessage(objectType, name, rv string) string {
	return fmt.Sprintf("watch cache mutation detected for %s %q, admitted at RV %q", objectType, name, rv)
}
