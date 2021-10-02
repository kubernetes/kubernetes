/*
Copyright 2021 The Kubernetes Authors.

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

package admission

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
)

// PodListerFromClient returns a PodLister that does live lists using the provided client.
func PodListerFromClient(client kubernetes.Interface) PodLister {
	return &clientPodLister{client}
}

type clientPodLister struct {
	client kubernetes.Interface
}

func (p *clientPodLister) ListPods(ctx context.Context, namespace string) ([]*corev1.Pod, error) {
	list, err := p.client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	pods := make([]*corev1.Pod, len(list.Items))
	for i := range list.Items {
		pods[i] = &list.Items[i]
	}
	return pods, nil
}

// PodListerFromInformer returns a PodLister that does cached lists using the provided lister.
func PodListerFromInformer(lister corev1listers.PodLister) PodLister {
	return &informerPodLister{lister}
}

type informerPodLister struct {
	lister corev1listers.PodLister
}

func (p *informerPodLister) ListPods(ctx context.Context, namespace string) ([]*corev1.Pod, error) {
	return p.lister.Pods(namespace).List(labels.Everything())
}
