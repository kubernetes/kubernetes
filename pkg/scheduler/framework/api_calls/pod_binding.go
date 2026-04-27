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

package apicalls

import (
	"context"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
)

// PodBindingCall is used to bind the pod using the binding details.
type PodBindingCall struct {
	binding *v1.Binding
}

func NewPodBindingCall(binding *v1.Binding) *PodBindingCall {
	return &PodBindingCall{
		binding: binding,
	}
}

func (pbc *PodBindingCall) CallType() fwk.APICallType {
	return PodBinding
}

func (pbc *PodBindingCall) UID() types.UID {
	return pbc.binding.UID
}

func (pbc *PodBindingCall) Execute(ctx context.Context, client clientset.Interface) error {
	logger := klog.FromContext(ctx)
	logger.V(3).Info("Attempting to bind pod to node", "pod", klog.KObj(&pbc.binding.ObjectMeta), "node", pbc.binding.Target.Name)

	return client.CoreV1().Pods(pbc.binding.Namespace).Bind(ctx, pbc.binding, metav1.CreateOptions{})
}

func (pbc *PodBindingCall) Sync(obj metav1.Object) (metav1.Object, error) {
	// Don't need to store or update an object.
	return obj, nil
}

func (pbc *PodBindingCall) Merge(oldCall fwk.APICall) error {
	// Merge should just overwrite the previous call.
	return nil
}

func (pbc *PodBindingCall) IsNoOp() bool {
	return false
}
