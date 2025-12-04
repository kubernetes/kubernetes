/*
Copyright 2024 The Kubernetes Authors.

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

package cache

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/util/consistencydetector"
	"k8s.io/klog/v2/ktesting"
)

func TestReflectorDataConsistencyDetector(t *testing.T) {
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, true)
	restore := consistencydetector.SetDataConsistencyDetectionForWatchListEnabledForTest(true)
	defer restore()

	markTransformed := func(obj interface{}) (interface{}, error) {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			return obj, nil
		}
		newPod := pod.DeepCopy()
		if newPod.Labels == nil {
			newPod.Labels = make(map[string]string)
		}
		newPod.Labels["transformed"] = "true"
		return newPod, nil
	}

	for _, inOrder := range []bool{false, true} {
		t.Run(fmt.Sprintf("InOrder=%v", inOrder), func(t *testing.T) {
			clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.InOrderInformers, inOrder)
			for _, transformerEnabled := range []bool{false, true} {
				var transformer TransformFunc
				if transformerEnabled {
					transformer = markTransformed
				}
				t.Run(fmt.Sprintf("Transformer=%v", transformerEnabled), func(t *testing.T) {
					runTestReflectorDataConsistencyDetector(t, transformer)
				})
			}
		})
	}
}

func runTestReflectorDataConsistencyDetector(t *testing.T, transformer TransformFunc) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	store := NewStore(MetaNamespaceKeyFunc)
	fifo := newQueueFIFO(store, transformer)

	lw := &ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "1"},
				Items: []v1.Pod{
					{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", ResourceVersion: "1"}},
				},
			}, nil
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			w := watch.NewFake()
			go func() {
				w.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", ResourceVersion: "1"}})
				w.Action(watch.Bookmark, &v1.Pod{ObjectMeta: metav1.ObjectMeta{
					Name:            "pod-1",
					ResourceVersion: "1",
					Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
				}})
			}()
			return w, nil
		},
	}

	r := NewReflector(lw, &v1.Pod{}, fifo, 0)

	go func() {
		_ = wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
			return r.LastSyncResourceVersion() != "", nil
		})
		cancel()
	}()

	err := r.ListAndWatchWithContext(ctx)
	if err != nil {
		t.Errorf("ListAndWatchWithContext returned error: %v", err)
	}
}
