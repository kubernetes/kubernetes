/*
Copyright 2022 The KCP Authors.

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

// +kcp-code-generator:skip

package clientsethack

import (
	kcpkubernetesclientset "github.com/kcp-dev/client-go/kubernetes"

	"k8s.io/client-go/discovery"
	"k8s.io/client-go/kubernetes"
	admissionregistrationv1 "k8s.io/client-go/kubernetes/typed/admissionregistration/v1"
	admissionregistrationv1alpha1 "k8s.io/client-go/kubernetes/typed/admissionregistration/v1alpha1"
	admissionregistrationv1beta1 "k8s.io/client-go/kubernetes/typed/admissionregistration/v1beta1"
	internalv1alpha1 "k8s.io/client-go/kubernetes/typed/apiserverinternal/v1alpha1"
	appsv1 "k8s.io/client-go/kubernetes/typed/apps/v1"
	appsv1beta1 "k8s.io/client-go/kubernetes/typed/apps/v1beta1"
	appsv1beta2 "k8s.io/client-go/kubernetes/typed/apps/v1beta2"
	authenticationv1 "k8s.io/client-go/kubernetes/typed/authentication/v1"
	"k8s.io/client-go/kubernetes/typed/authentication/v1alpha1"
	authenticationv1beta1 "k8s.io/client-go/kubernetes/typed/authentication/v1beta1"
	authorizationv1 "k8s.io/client-go/kubernetes/typed/authorization/v1"
	authorizationv1beta1 "k8s.io/client-go/kubernetes/typed/authorization/v1beta1"
	autoscalingv1 "k8s.io/client-go/kubernetes/typed/autoscaling/v1"
	autoscalingv2 "k8s.io/client-go/kubernetes/typed/autoscaling/v2"
	autoscalingv2beta1 "k8s.io/client-go/kubernetes/typed/autoscaling/v2beta1"
	autoscalingv2beta2 "k8s.io/client-go/kubernetes/typed/autoscaling/v2beta2"
	batchv1 "k8s.io/client-go/kubernetes/typed/batch/v1"
	batchv1beta1 "k8s.io/client-go/kubernetes/typed/batch/v1beta1"
	certificatesv1 "k8s.io/client-go/kubernetes/typed/certificates/v1"
	certificatesv1alpha1 "k8s.io/client-go/kubernetes/typed/certificates/v1alpha1"
	certificatesv1beta1 "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	coordinationv1 "k8s.io/client-go/kubernetes/typed/coordination/v1"
	coordinationV1alpha1 "k8s.io/client-go/kubernetes/typed/coordination/v1alpha1"
	coordinationv1beta1 "k8s.io/client-go/kubernetes/typed/coordination/v1beta1"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	discoveryv1 "k8s.io/client-go/kubernetes/typed/discovery/v1"
	discoveryv1beta1 "k8s.io/client-go/kubernetes/typed/discovery/v1beta1"
	eventsv1 "k8s.io/client-go/kubernetes/typed/events/v1"
	eventsv1beta1 "k8s.io/client-go/kubernetes/typed/events/v1beta1"
	extensionsv1beta1 "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	flowcontrolv1 "k8s.io/client-go/kubernetes/typed/flowcontrol/v1"
	flowcontrolv1beta1 "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta1"
	flowcontrolv1beta2 "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta2"
	flowcontrolv1beta3 "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta3"
	networkingv1 "k8s.io/client-go/kubernetes/typed/networking/v1"
	networkingv1alpha1 "k8s.io/client-go/kubernetes/typed/networking/v1alpha1"
	networkingv1beta1 "k8s.io/client-go/kubernetes/typed/networking/v1beta1"
	nodev1 "k8s.io/client-go/kubernetes/typed/node/v1"
	nodev1alpha1 "k8s.io/client-go/kubernetes/typed/node/v1alpha1"
	nodev1beta1 "k8s.io/client-go/kubernetes/typed/node/v1beta1"
	policyv1 "k8s.io/client-go/kubernetes/typed/policy/v1"
	policyv1beta1 "k8s.io/client-go/kubernetes/typed/policy/v1beta1"
	rbacv1 "k8s.io/client-go/kubernetes/typed/rbac/v1"
	rbacv1alpha1 "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/client-go/kubernetes/typed/rbac/v1beta1"
	resourcev1alpha3 "k8s.io/client-go/kubernetes/typed/resource/v1alpha3"
	schedulingv1 "k8s.io/client-go/kubernetes/typed/scheduling/v1"
	schedulingv1alpha1 "k8s.io/client-go/kubernetes/typed/scheduling/v1alpha1"
	schedulingv1beta1 "k8s.io/client-go/kubernetes/typed/scheduling/v1beta1"
	storagev1 "k8s.io/client-go/kubernetes/typed/storage/v1"
	storagev1alpha1 "k8s.io/client-go/kubernetes/typed/storage/v1alpha1"
	storagev1beta1 "k8s.io/client-go/kubernetes/typed/storage/v1beta1"
	storagemigrationv1alpha1 "k8s.io/client-go/kubernetes/typed/storagemigration/v1alpha1"
)

// Interface allows us to hold onto a strongly-typed cluster-aware clients here, while
// passing in a cluster-unaware (but non-functional) clients to k8s libraries. We export this type so that we
// can get the cluster-aware clients back using casting in admission plugin initialization.
type Interface interface {
	kubernetes.Interface
	ClusterAware() kcpkubernetesclientset.ClusterInterface
}

var _ Interface = (*hack)(nil)

// Wrap adapts a cluster-aware informer factory to a cluster-unaware wrapper that can divulge it after casting.
func Wrap(clusterAware kcpkubernetesclientset.ClusterInterface) Interface {
	return &hack{clusterAware: clusterAware}
}

// Unwrap extracts a cluster-aware informer factory from the cluster-unaware wrapper, or panics if we get the wrong input.
func Unwrap(clusterUnaware kubernetes.Interface) kcpkubernetesclientset.ClusterInterface {
	return clusterUnaware.(Interface).ClusterAware()
}

type hack struct {
	clusterAware kcpkubernetesclientset.ClusterInterface
}

func (h *hack) AdmissionregistrationV1alpha1() admissionregistrationv1alpha1.AdmissionregistrationV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AuthenticationV1alpha1() v1alpha1.AuthenticationV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) NetworkingV1alpha1() networkingv1alpha1.NetworkingV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) ResourceV1alpha3() resourcev1alpha3.ResourceV1alpha3Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AdmissionregistrationV1() admissionregistrationv1.AdmissionregistrationV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AdmissionregistrationV1beta1() admissionregistrationv1beta1.AdmissionregistrationV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) InternalV1alpha1() internalv1alpha1.InternalV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AppsV1() appsv1.AppsV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AppsV1beta1() appsv1beta1.AppsV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AppsV1beta2() appsv1beta2.AppsV1beta2Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AuthenticationV1() authenticationv1.AuthenticationV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AuthenticationV1beta1() authenticationv1beta1.AuthenticationV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AuthorizationV1() authorizationv1.AuthorizationV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AuthorizationV1beta1() authorizationv1beta1.AuthorizationV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AutoscalingV1() autoscalingv1.AutoscalingV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AutoscalingV2() autoscalingv2.AutoscalingV2Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AutoscalingV2beta1() autoscalingv2beta1.AutoscalingV2beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) AutoscalingV2beta2() autoscalingv2beta2.AutoscalingV2beta2Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) BatchV1() batchv1.BatchV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) BatchV1beta1() batchv1beta1.BatchV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) CertificatesV1() certificatesv1.CertificatesV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) CertificatesV1alpha1() certificatesv1alpha1.CertificatesV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) CertificatesV1beta1() certificatesv1beta1.CertificatesV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) CoordinationV1beta1() coordinationv1beta1.CoordinationV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) CoordinationV1alpha1() coordinationV1alpha1.CoordinationV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) CoordinationV1() coordinationv1.CoordinationV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) CoreV1() corev1.CoreV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) DiscoveryV1() discoveryv1.DiscoveryV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) DiscoveryV1beta1() discoveryv1beta1.DiscoveryV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) EventsV1() eventsv1.EventsV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) EventsV1beta1() eventsv1beta1.EventsV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) ExtensionsV1beta1() extensionsv1beta1.ExtensionsV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) FlowcontrolV1beta1() flowcontrolv1beta1.FlowcontrolV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) FlowcontrolV1beta2() flowcontrolv1beta2.FlowcontrolV1beta2Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) FlowcontrolV1beta3() flowcontrolv1beta3.FlowcontrolV1beta3Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) FlowcontrolV1() flowcontrolv1.FlowcontrolV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) NetworkingV1() networkingv1.NetworkingV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) NetworkingV1beta1() networkingv1beta1.NetworkingV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) NodeV1() nodev1.NodeV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) NodeV1alpha1() nodev1alpha1.NodeV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) NodeV1beta1() nodev1beta1.NodeV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) PolicyV1() policyv1.PolicyV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) PolicyV1beta1() policyv1beta1.PolicyV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) RbacV1() rbacv1.RbacV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) RbacV1beta1() rbacv1beta1.RbacV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) RbacV1alpha1() rbacv1alpha1.RbacV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) SchedulingV1alpha1() schedulingv1alpha1.SchedulingV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) SchedulingV1beta1() schedulingv1beta1.SchedulingV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) SchedulingV1() schedulingv1.SchedulingV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) StorageV1beta1() storagev1beta1.StorageV1beta1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) StorageV1() storagev1.StorageV1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) StorageV1alpha1() storagev1alpha1.StorageV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) StoragemigrationV1alpha1() storagemigrationv1alpha1.StoragemigrationV1alpha1Interface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) Discovery() discovery.DiscoveryInterface {
	panic("programmer error: using a cluster-unaware clientset, need to cast this to use the cluster-aware one!")
}

func (h *hack) ClusterAware() kcpkubernetesclientset.ClusterInterface {
	return h.clusterAware
}
