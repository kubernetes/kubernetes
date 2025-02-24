/*
Copyright 2025 The Kubernetes Authors.

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

package apps

import (
	"context"
	"fmt"
	"maps"
	"strconv"
	"sync"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
)

type IndexedPodAnnotationTracker struct {
	sync.Mutex
	ownerName            string
	ownerNs              string
	labelSelector        string
	podIndexAnnotation   string
	podTrackedAnnotation string
	trackedAnnotations   map[int][]string
}

func NewIndexedPodAnnotationTracker(ownerName, ownerNs, labelSelector, podIndexAnnotation, podTrackedAnnotation string) *IndexedPodAnnotationTracker {
	return &IndexedPodAnnotationTracker{
		ownerName:            ownerName,
		ownerNs:              ownerNs,
		labelSelector:        labelSelector,
		podIndexAnnotation:   podIndexAnnotation,
		podTrackedAnnotation: podTrackedAnnotation,
		trackedAnnotations:   make(map[int][]string),
	}
}

func (t *IndexedPodAnnotationTracker) Start(ctx context.Context, c clientset.Interface) context.CancelFunc {
	ownerKey := klog.KRef(t.ownerNs, t.ownerName)
	podClient := c.CoreV1().Pods(t.ownerNs)
	podsList, err := podClient.List(ctx, metav1.ListOptions{LabelSelector: t.labelSelector})
	framework.ExpectNoError(err, "failed to list Pods")

	trackerCtx, trackerCancel := context.WithCancel(ctx)

	go func() {
		defer ginkgo.GinkgoRecover()

		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = t.labelSelector
				return podClient.Watch(ctx, options)
			},
		}

		ginkgo.By(fmt.Sprintf("Start the Pod watch for owner: %s", ownerKey))
		_, err = watchtools.Until(trackerCtx, podsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if event.Type == watch.Added {
				if pod, ok := event.Object.(*v1.Pod); ok {
					framework.Logf("Observed event for Pod %q with index=%v, annotation value=%v",
						klog.KObj(pod), pod.Annotations[t.podIndexAnnotation], pod.Annotations[t.podTrackedAnnotation])
					podIndex, err := strconv.Atoi(pod.Annotations[t.podIndexAnnotation])
					if err != nil {
						return true, fmt.Errorf("failed to parse pod index for Pod %q: %v", klog.KObj(pod), err.Error())
					}
					t.Lock()
					defer t.Unlock()
					t.trackedAnnotations[podIndex] = append(t.trackedAnnotations[podIndex], pod.Annotations[t.podTrackedAnnotation])
					return false, nil
				}
			}
			return false, nil
		})
		// We ignore the error corresponding to the context getting interrupted.
		// The test code is expected to assert on the map before cancelling the
		// context.
		if err != nil && !wait.Interrupted(err) {
			framework.Failf("failed to track Pod annotation %q for owner %q: %v", t.podTrackedAnnotation, ownerKey, err.Error())
		}
	}()
	return trackerCancel
}

func (t *IndexedPodAnnotationTracker) GetMap() map[int][]string {
	t.Lock()
	defer t.Unlock()
	return maps.Clone(t.trackedAnnotations)
}
