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

package polymorphichelpers

import (
	"context"
	"fmt"
	"sort"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	coreclientv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
)

// GetFirstPod returns a pod matching the namespace and label selector
// and the number of all pods that match the label selector.
func GetFirstPod(client coreclientv1.PodsGetter, namespace string, selector string, timeout time.Duration, sortBy func([]*corev1.Pod) sort.Interface) (*corev1.Pod, int, error) {
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.LabelSelector = selector
			return client.Pods(namespace).List(options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.LabelSelector = selector
			return client.Pods(namespace).Watch(options)
		},
	}

	var initialPods []*corev1.Pod
	preconditionFunc := func(store cache.Store) (bool, error) {
		items := store.List()
		if len(items) > 0 {
			for _, item := range items {
				pod, ok := item.(*corev1.Pod)
				if !ok {
					return true, fmt.Errorf("unexpected store item type: %#v", item)
				}

				initialPods = append(initialPods, pod)
			}

			sort.Sort(sortBy(initialPods))

			return true, nil
		}

		// Continue by watching for a new pod to appear
		return false, nil
	}

	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), timeout)
	defer cancel()
	event, err := watchtools.UntilWithSync(ctx, lw, &corev1.Pod{}, preconditionFunc, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Added, watch.Modified:
			// Any pod is good enough
			return true, nil

		case watch.Deleted:
			return true, fmt.Errorf("object was deleted before reaching a %#v", event.Object)

		case watch.Error:
			return true, fmt.Errorf("unexpected error %#v", event.Object)

		default:
			return true, fmt.Errorf("unexpected event type: %T", event.Type)
		}
	})
	if err != nil {
		return nil, 0, err
	}

	if len(initialPods) > 0 {
		return initialPods[0], len(initialPods), nil
	}

	pod, ok := event.Object.(*corev1.Pod)
	if !ok {
		return nil, 0, fmt.Errorf("%#v is not a pod event", event)
	}

	return pod, 1, nil
}

// SelectorsForObject returns the pod label selector for a given object
func SelectorsForObject(object runtime.Object) (namespace string, selector labels.Selector, err error) {
	switch t := object.(type) {
	case *extensionsv1beta1.ReplicaSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1.ReplicaSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1beta2.ReplicaSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *corev1.ReplicationController:
		namespace = t.Namespace
		selector = labels.SelectorFromSet(t.Spec.Selector)

	case *appsv1.StatefulSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1beta1.StatefulSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1beta2.StatefulSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *extensionsv1beta1.DaemonSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1.DaemonSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1beta2.DaemonSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *extensionsv1beta1.Deployment:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1.Deployment:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1beta1.Deployment:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}
	case *appsv1beta2.Deployment:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *batchv1.Job:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return "", nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *corev1.Service:
		namespace = t.Namespace
		if t.Spec.Selector == nil || len(t.Spec.Selector) == 0 {
			return "", nil, fmt.Errorf("invalid service '%s': Service is defined without a selector", t.Name)
		}
		selector = labels.SelectorFromSet(t.Spec.Selector)

	default:
		return "", nil, fmt.Errorf("selector for %T not implemented", object)
	}

	return namespace, selector, nil
}
