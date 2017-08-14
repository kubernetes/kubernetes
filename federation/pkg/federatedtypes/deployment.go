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
	apiv1 "k8s.io/api/core/v1"
	extensionsv1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
)

const (
	DeploymentKind                     = "deployment"
	DeploymentControllerName           = "deployments"
	FedDeploymentPreferencesAnnotation = "federation.kubernetes.io/deployment-preferences"
)

func init() {
	RegisterFederatedType(DeploymentKind, DeploymentControllerName, []schema.GroupVersionResource{extensionsv1.SchemeGroupVersion.WithResource(DeploymentControllerName)}, NewDeploymentAdapter)
}

type DeploymentAdapter struct {
	*replicaSchedulingAdapter
	client federationclientset.Interface
}

func NewDeploymentAdapter(client federationclientset.Interface, config *restclient.Config, adapterSpecificArgs map[string]interface{}) FederatedTypeAdapter {
	schedulingAdapter := replicaSchedulingAdapter{
		preferencesAnnotationName: FedDeploymentPreferencesAnnotation,
		updateStatusFunc: func(obj pkgruntime.Object, schedulingInfo interface{}) error {
			deployment := obj.(*extensionsv1.Deployment)
			typedStatus := schedulingInfo.(*ReplicaSchedulingInfo).Status
			if typedStatus.Replicas != deployment.Status.Replicas || typedStatus.UpdatedReplicas != deployment.Status.UpdatedReplicas ||
				typedStatus.ReadyReplicas != deployment.Status.ReadyReplicas || typedStatus.AvailableReplicas != deployment.Status.AvailableReplicas {
				deployment.Status = extensionsv1.DeploymentStatus{
					Replicas:          typedStatus.Replicas,
					UpdatedReplicas:   typedStatus.UpdatedReplicas,
					ReadyReplicas:     typedStatus.ReadyReplicas,
					AvailableReplicas: typedStatus.AvailableReplicas,
				}
				_, err := client.Extensions().Deployments(deployment.Namespace).UpdateStatus(deployment)
				return err
			}
			return nil
		},
	}

	return &DeploymentAdapter{&schedulingAdapter, client}
}

func (a *DeploymentAdapter) Kind() string {
	return DeploymentKind
}

func (a *DeploymentAdapter) ObjectType() pkgruntime.Object {
	return &extensionsv1.Deployment{}
}

func (a *DeploymentAdapter) IsExpectedType(obj interface{}) bool {
	_, ok := obj.(*extensionsv1.Deployment)
	return ok
}

func (a *DeploymentAdapter) Copy(obj pkgruntime.Object) pkgruntime.Object {
	deployment := obj.(*extensionsv1.Deployment)
	return fedutil.DeepCopyDeployment(deployment)
}

func (a *DeploymentAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	deployment1 := obj1.(*extensionsv1.Deployment)
	deployment2 := obj2.(*extensionsv1.Deployment)
	return fedutil.DeploymentEquivalent(deployment1, deployment2)
}

func (a *DeploymentAdapter) QualifiedName(obj pkgruntime.Object) QualifiedName {
	deployment := obj.(*extensionsv1.Deployment)
	return QualifiedName{Namespace: deployment.Namespace, Name: deployment.Name}
}

func (a *DeploymentAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*extensionsv1.Deployment).ObjectMeta
}

func (a *DeploymentAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	deployment := obj.(*extensionsv1.Deployment)
	return a.client.Extensions().Deployments(deployment.Namespace).Create(deployment)
}

func (a *DeploymentAdapter) FedDelete(qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return a.client.Extensions().Deployments(qualifiedName.Namespace).Delete(qualifiedName.Name, options)
}

func (a *DeploymentAdapter) FedGet(qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return a.client.Extensions().Deployments(qualifiedName.Namespace).Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *DeploymentAdapter) FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return a.client.Extensions().Deployments(namespace).List(options)
}

func (a *DeploymentAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	deployment := obj.(*extensionsv1.Deployment)
	return a.client.Extensions().Deployments(deployment.Namespace).Update(deployment)
}

func (a *DeploymentAdapter) FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return a.client.Extensions().Deployments(namespace).Watch(options)
}

func (a *DeploymentAdapter) ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	deployment := obj.(*extensionsv1.Deployment)
	return client.Extensions().Deployments(deployment.Namespace).Create(deployment)
}

func (a *DeploymentAdapter) ClusterDelete(client kubeclientset.Interface, qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return client.Extensions().Deployments(qualifiedName.Namespace).Delete(qualifiedName.Name, options)
}

func (a *DeploymentAdapter) ClusterGet(client kubeclientset.Interface, qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return client.Extensions().Deployments(qualifiedName.Namespace).Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *DeploymentAdapter) ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return client.Extensions().Deployments(namespace).List(options)
}

func (a *DeploymentAdapter) ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	deployment := obj.(*extensionsv1.Deployment)
	return client.Extensions().Deployments(deployment.Namespace).Update(deployment)
}

func (a *DeploymentAdapter) ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return client.Extensions().Deployments(namespace).Watch(options)
}

func (a *DeploymentAdapter) EquivalentIgnoringSchedule(obj1, obj2 pkgruntime.Object) bool {
	deployment1 := obj1.(*extensionsv1.Deployment)
	deployment2 := a.Copy(obj2).(*extensionsv1.Deployment)
	deployment2.Spec.Replicas = deployment1.Spec.Replicas
	return fedutil.DeploymentEquivalent(deployment1, deployment2)
}

func (a *DeploymentAdapter) NewTestObject(namespace string) pkgruntime.Object {
	replicas := int32(3)
	zero := int64(0)
	return &extensionsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-deployment-",
			Namespace:    namespace,
		},
		Spec: extensionsv1.DeploymentSpec{
			Replicas: &replicas,
			Template: apiv1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: apiv1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []apiv1.Container{
						{
							Name:  "nginx",
							Image: "nginx",
						},
					},
				},
			},
		},
	}
}
