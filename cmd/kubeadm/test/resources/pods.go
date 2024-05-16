/*
Copyright 2020 The Kubernetes Authors.

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

package resources

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"

	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

// FakeStaticPod represents a fake static pod
type FakeStaticPod struct {
	NodeName    string
	Component   string
	Annotations map[string]string
}

// Pod returns a pod structure representing the fake static pod with a
// given suffix
func (p *FakeStaticPod) Pod(suffix string) *v1.Pod {
	pod := staticpodutil.ComponentPod(
		v1.Container{
			Name:  p.Component,
			Image: fmt.Sprintf("%s-image:tag", p.Component),
		},
		map[string]v1.Volume{},
		p.Annotations,
	)
	if len(suffix) > 0 {
		pod.ObjectMeta.Name = fmt.Sprintf("%s-%s-%s", p.Component, p.NodeName, suffix)
	} else {
		pod.ObjectMeta.Name = fmt.Sprintf("%s-%s", p.Component, p.NodeName)
	}
	pod.Spec.NodeName = p.NodeName
	return &pod
}

// Create creates a fake static pod using the provided client
func (p *FakeStaticPod) Create(client clientset.Interface) error {
	return p.CreateWithPodSuffix(client, "")
}

// CreateWithPodSuffix creates a fake static pod using the provided
// client and suffix
func (p *FakeStaticPod) CreateWithPodSuffix(client clientset.Interface, suffix string) error {
	_, err := client.CoreV1().Pods(metav1.NamespaceSystem).Create(context.TODO(), p.Pod(suffix), metav1.CreateOptions{})
	return err
}
