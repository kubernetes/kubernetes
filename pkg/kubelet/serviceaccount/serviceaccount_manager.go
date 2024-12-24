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

package serviceaccount

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/util/manager"
)

// Manager manages Kubernetes serviceaccounts. This includes retrieving
// serviceaccounts or registering/unregistering them via Pods.
type Manager interface {
	// Get serviceaccount by serviceaccount namespace and name.
	GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error)

	// WARNING: Register/UnregisterPod functions should be efficient,
	// i.e. should not block on network operations.

	// RegisterPod register serviceaccount from a given pod.
	RegisterPod(pod *v1.Pod)

	// UnregisterPod unregisters serviceaccount from a given pod that are not
	// used by any other registered pod.
	UnregisterPod(pod *v1.Pod)
}

// serviceAccountManager keeps a store with serviceaccounts necessary
// for registered pods.
type serviceAccountManager struct {
	manager manager.Manager
}

func (s *serviceAccountManager) GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error) {
	object, err := s.manager.GetObject(namespace, name)
	if err != nil {
		return nil, err
	}
	if serviceAccount, ok := object.(*v1.ServiceAccount); ok {
		return serviceAccount, nil
	}
	return nil, fmt.Errorf("unexpected object type: %v", object)
}

func (s *serviceAccountManager) RegisterPod(pod *v1.Pod) {
	s.manager.RegisterPod(pod)
}

func (s *serviceAccountManager) UnregisterPod(pod *v1.Pod) {
	s.manager.UnregisterPod(pod)
}

func getServiceAccountName(pod *v1.Pod) sets.Set[string] {
	if len(pod.Spec.ServiceAccountName) == 0 {
		return nil
	}
	return sets.New[string](pod.Spec.ServiceAccountName)

}

// NewWatchingServiceAccountManager creates a manager that keeps a cache of all serviceaccounts
// necessary for registered pods.
// It implements the following logic:
//   - whenever a pod is created or updated, we start individual watches for all
//     referenced objects that aren't referenced from other registered pods
//   - every GetObject() returns a value from local cache propagated via watches
func NewWatchingServiceAccountManager(kubeClient clientset.Interface) Manager {
	listServiceAccount := func(namespace string, opts metav1.ListOptions) (runtime.Object, error) {
		return kubeClient.CoreV1().ServiceAccounts(namespace).List(context.TODO(), opts)
	}
	watchServiceAccount := func(namespace string, opts metav1.ListOptions) (watch.Interface, error) {
		return kubeClient.CoreV1().ServiceAccounts(namespace).Watch(context.TODO(), opts)
	}
	newServiceAccount := func() runtime.Object {
		return &v1.ServiceAccount{}
	}
	isImmutable := func(object runtime.Object) bool {
		return false
	}
	gr := corev1.Resource("serviceaccount")
	return &serviceAccountManager{
		manager: manager.NewWatchBasedManager(listServiceAccount, watchServiceAccount, newServiceAccount, isImmutable, gr, 0, getServiceAccountName),
	}
}

type NoopManager struct{}

var _ Manager = (*NoopManager)(nil)

func (n *NoopManager) GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error) {
	return nil, fmt.Errorf("not implemented")
}

func (n *NoopManager) RegisterPod(pod *v1.Pod) {}

func (n *NoopManager) UnregisterPod(pod *v1.Pod) {}
