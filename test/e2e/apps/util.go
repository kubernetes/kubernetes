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
	"maps"
	"strconv"
	"sync"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
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
	trackerCtx, trackerCancel := context.WithCancel(ctx)
	_, podTracker := cache.NewInformerWithOptions(cache.InformerOptions{
		ListerWatcher: &cache.ListWatch{
			ListWithContextFunc: func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
				options.LabelSelector = t.labelSelector
				obj, err := c.CoreV1().Pods(t.ownerNs).List(ctx, options)
				return runtime.Object(obj), err
			},
			WatchFuncWithContext: func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = t.labelSelector
				return c.CoreV1().Pods(t.ownerNs).Watch(ctx, options)
			},
		},
		ObjectType: &v1.Pod{},
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				defer ginkgo.GinkgoRecover()
				if pod, ok := obj.(*v1.Pod); ok {
					framework.Logf("Observed event for Pod %q with index=%v, annotation value=%v",
						klog.KObj(pod), pod.Annotations[t.podIndexAnnotation], pod.Annotations[t.podTrackedAnnotation])
					podIndex, err := strconv.Atoi(pod.Annotations[t.podIndexAnnotation])
					if err != nil {
						framework.Failf("failed to parse pod index for Pod %q: %v", klog.KObj(pod), err.Error())
					} else {
						t.Lock()
						defer t.Unlock()
						t.trackedAnnotations[podIndex] = append(t.trackedAnnotations[podIndex], pod.Annotations[t.podTrackedAnnotation])
					}
				}
			},
			UpdateFunc: func(old, new interface{}) {
				defer ginkgo.GinkgoRecover()
				oldPod, oldOk := old.(*v1.Pod)
				newPod, newOk := new.(*v1.Pod)
				if !oldOk || !newOk {
					return
				}
				if oldPod.Annotations[t.podTrackedAnnotation] != newPod.Annotations[t.podTrackedAnnotation] {
					framework.Failf("Unexepected mutation of the annotation %q for Pod %q, old=%q, new=%q",
						t.podTrackedAnnotation,
						klog.KObj(newPod),
						oldPod.Annotations[t.podTrackedAnnotation],
						newPod.Annotations[t.podTrackedAnnotation],
					)
				}
			},
		},
	})
	go podTracker.RunWithContext(trackerCtx)
	return trackerCancel
}

func (t *IndexedPodAnnotationTracker) cloneTrackedAnnotations() map[int][]string {
	t.Lock()
	defer t.Unlock()
	return maps.Clone(t.trackedAnnotations)
}
