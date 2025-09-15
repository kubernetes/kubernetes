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
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	watchtools "k8s.io/client-go/tools/watch"
)

// GetPodList returns a PodList matching the namespace and label selector
func GetPodList(client coreclient.PodsGetter, namespace string, selector string, timeout time.Duration, sortBy func([]*corev1.Pod) sort.Interface) (*corev1.PodList, error) {
	options := metav1.ListOptions{LabelSelector: selector}

	podList, err := client.Pods(namespace).List(context.TODO(), options)
	if err != nil {
		return nil, err
	}

	pods := []*corev1.Pod{}
	for i := range podList.Items {
		pod := podList.Items[i]
		pods = append(pods, &pod)
	}
	if len(pods) > 0 {
		sort.Sort(sortBy(pods))
		for i, pod := range pods {
			podList.Items[i] = *pod
		}
		return podList, nil
	}

	// Watch until we observe a pod
	options.ResourceVersion = podList.ResourceVersion
	w, err := client.Pods(namespace).Watch(context.TODO(), options)
	if err != nil {
		return nil, err
	}
	defer w.Stop()

	condition := func(event watch.Event) (bool, error) {
		return event.Type == watch.Added || event.Type == watch.Modified, nil
	}

	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), timeout)
	defer cancel()
	event, err := watchtools.UntilWithoutRetry(ctx, w, condition)
	if err != nil {
		return nil, err
	}

	po, ok := event.Object.(*corev1.Pod)
	if !ok {
		return nil, fmt.Errorf("%#v is not a pod event", event)
	} else {
		podList.Items = append(podList.Items, *po)
	}
	return podList, nil
}

// GetFirstPod returns a pod matching the namespace and label selector
// and the number of all pods that match the label selector.
func GetFirstPod(client coreclient.PodsGetter, namespace string, selector string, timeout time.Duration, sortBy func([]*corev1.Pod) sort.Interface) (*corev1.Pod, int, error) {

	podList, err := GetPodList(client, namespace, selector, timeout, sortBy)
	if err != nil {
		return nil, 0, err
	}

	return &podList.Items[0], len(podList.Items), nil

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
