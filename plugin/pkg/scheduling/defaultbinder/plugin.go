/*
Copyright 2019 The Kubernetes Authors.

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

package defaultbinder

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// DefaultBinder .
type DefaultBinder struct {
	Client clientset.Interface
}

var _ framework.BindPlugin = &DefaultBinder{}

// Name is the name of the default binder plugin
const Name = "default-binder-plugin"

// Name returns default binder's name.
func (p DefaultBinder) Name() string {
	return Name
}

// Bind binds the pod to the given node.
func (p DefaultBinder) Bind(pc *framework.PluginContext, pod *v1.Pod, nodeName string) *framework.Status {
	binding := &v1.Binding{
		ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name, UID: pod.UID},
		Target: v1.ObjectReference{
			Kind: "Node",
			Name: nodeName,
		},
	}

	if err := p.Client.CoreV1().Pods(binding.Namespace).Bind(binding); err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	return nil
}

// New returns a default binder plugin.
func New(_ *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	return &DefaultBinder{
		Client: handle.Clientset(),
	}, nil
}
