/*
Copyright 2017 The Kubernetes Authors.

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

package federatedtypes

import (
	"fmt"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/controller/namespace/deletion"

	"github.com/golang/glog"
)

const (
	NamespaceKind           = "namespace"
	NamespaceControllerName = "namespaces"
)

func init() {
	RegisterFederatedType(NamespaceKind, NamespaceControllerName, []schema.GroupVersionResource{apiv1.SchemeGroupVersion.WithResource(NamespaceControllerName)}, NewNamespaceAdapter)
}

type NamespaceAdapter struct {
	client  federationclientset.Interface
	deleter deletion.NamespacedResourcesDeleterInterface
}

func NewNamespaceAdapter(client federationclientset.Interface, config *restclient.Config, adapterSpecificArgs map[string]interface{}) FederatedTypeAdapter {
	dynamicClientPool := dynamic.NewDynamicClientPool(config)
	discoverResourcesFunc := client.Discovery().ServerPreferredNamespacedResources
	deleter := deletion.NewNamespacedResourcesDeleter(
		client.Core().Namespaces(),
		dynamicClientPool,
		nil,
		discoverResourcesFunc,
		apiv1.FinalizerKubernetes,
		false)
	return &NamespaceAdapter{client: client, deleter: deleter}
}

func (a *NamespaceAdapter) Kind() string {
	return NamespaceKind
}

func (a *NamespaceAdapter) ObjectType() pkgruntime.Object {
	return &apiv1.Namespace{}
}

func (a *NamespaceAdapter) IsExpectedType(obj interface{}) bool {
	_, ok := obj.(*apiv1.Namespace)
	return ok
}

func (a *NamespaceAdapter) Copy(obj pkgruntime.Object) pkgruntime.Object {
	namespace := obj.(*apiv1.Namespace)
	return &apiv1.Namespace{
		ObjectMeta: util.DeepCopyRelevantObjectMeta(namespace.ObjectMeta),
		Spec:       *(util.DeepCopyApiTypeOrPanic(&namespace.Spec).(*apiv1.NamespaceSpec)),
	}
}

func (a *NamespaceAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	return util.ObjectMetaAndSpecEquivalent(obj1, obj2)
}

func (a *NamespaceAdapter) QualifiedName(obj pkgruntime.Object) QualifiedName {
	namespace := obj.(*apiv1.Namespace)
	return QualifiedName{Name: namespace.Name}
}

func (a *NamespaceAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*apiv1.Namespace).ObjectMeta
}

func (a *NamespaceAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	namespace := obj.(*apiv1.Namespace)
	return a.client.CoreV1().Namespaces().Create(namespace)
}

func (a *NamespaceAdapter) FedDelete(qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return a.client.CoreV1().Namespaces().Delete(qualifiedName.Name, options)
}

func (a *NamespaceAdapter) FedGet(qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return a.client.CoreV1().Namespaces().Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *NamespaceAdapter) FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return a.client.CoreV1().Namespaces().List(options)
}

func (a *NamespaceAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	namespace := obj.(*apiv1.Namespace)
	return a.client.CoreV1().Namespaces().Update(namespace)
}

func (a *NamespaceAdapter) FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return a.client.CoreV1().Namespaces().Watch(options)
}

func (a *NamespaceAdapter) ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	namespace := obj.(*apiv1.Namespace)
	return client.CoreV1().Namespaces().Create(namespace)
}

func (a *NamespaceAdapter) ClusterDelete(client kubeclientset.Interface, qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return client.CoreV1().Namespaces().Delete(qualifiedName.Name, options)
}

func (a *NamespaceAdapter) ClusterGet(client kubeclientset.Interface, qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return client.CoreV1().Namespaces().Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *NamespaceAdapter) ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return client.CoreV1().Namespaces().List(options)
}

func (a *NamespaceAdapter) ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	namespace := obj.(*apiv1.Namespace)
	return client.CoreV1().Namespaces().Update(namespace)
}

func (a *NamespaceAdapter) ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return client.CoreV1().Namespaces().Watch(options)
}

func (a *NamespaceAdapter) IsSchedulingAdapter() bool {
	return false
}

func (a *NamespaceAdapter) NewTestObject(namespace string) pkgruntime.Object {
	return &apiv1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-namespace-",
		},
		Spec: apiv1.NamespaceSpec{
			Finalizers: []apiv1.FinalizerName{apiv1.FinalizerKubernetes},
		},
	}
}

// CleanUpNamespace deletes all resources in a given namespace.
func (a *NamespaceAdapter) CleanUpNamespace(obj pkgruntime.Object, eventRecorder record.EventRecorder) (pkgruntime.Object, error) {
	namespace := obj.(*apiv1.Namespace)
	name := namespace.Name

	// Set Terminating status.
	updatedNamespace := &apiv1.Namespace{
		ObjectMeta: namespace.ObjectMeta,
		Spec:       namespace.Spec,
		Status: apiv1.NamespaceStatus{
			Phase: apiv1.NamespaceTerminating,
		},
	}
	var err error
	if namespace.Status.Phase != apiv1.NamespaceTerminating {
		glog.V(2).Infof("Marking ns %s as terminating", name)
		eventRecorder.Event(namespace, api.EventTypeNormal, "DeleteNamespace", fmt.Sprintf("Marking for deletion"))
		_, err = a.FedUpdate(updatedNamespace)
		if err != nil {
			return nil, fmt.Errorf("failed to update namespace: %v", err)
		}
	}

	if hasFinalizerInSpec(updatedNamespace, apiv1.FinalizerKubernetes) {
		// Delete resources in this namespace.
		err = a.deleter.Delete(name)
		if err != nil {
			return nil, fmt.Errorf("error in deleting resources in namespace %s: %v", name, err)
		}
		glog.V(2).Infof("Removed kubernetes finalizer from ns %s", name)
		// Fetch the updated Namespace.
		obj, err = a.FedGet(QualifiedName{Name: name})
		updatedNamespace = obj.(*apiv1.Namespace)
		if err != nil {
			return nil, fmt.Errorf("error in fetching updated namespace %s: %s", name, err)
		}
	}

	return updatedNamespace, nil
}

func hasFinalizerInSpec(namespace *apiv1.Namespace, finalizer apiv1.FinalizerName) bool {
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] == finalizer {
			return true
		}
	}
	return false
}
