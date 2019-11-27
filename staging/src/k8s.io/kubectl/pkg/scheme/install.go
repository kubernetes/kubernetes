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

package scheme

import (
	admissionv1 "k8s.io/api/admission/v1"
	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2beta1 "k8s.io/api/autoscaling/v2beta1"
	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	batchv2alpha1 "k8s.io/api/batch/v2alpha1"
	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	imagepolicyv1alpha1 "k8s.io/api/imagepolicy/v1alpha1"
	networkingv1 "k8s.io/api/networking/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	rbacv1alpha1 "k8s.io/api/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	schedulingv1alpha1 "k8s.io/api/scheduling/v1alpha1"
	settingsv1alpha1 "k8s.io/api/settings/v1alpha1"
	storagev1 "k8s.io/api/storage/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes/scheme"
)

// Register all groups in the kubectl's registry, but no componentconfig group since it's not in k8s.io/api
// The code in this file mostly duplicate the install under k8s.io/kubernetes/pkg/api and k8s.io/kubernetes/pkg/apis,
// but does NOT register the internal types.
func init() {
	// Register external types for Scheme
	metav1.AddToGroupVersion(Scheme, schema.GroupVersion{Version: "v1"})
	utilruntime.Must(metav1beta1.AddMetaToScheme(Scheme))
	utilruntime.Must(metav1.AddMetaToScheme(Scheme))
	utilruntime.Must(scheme.AddToScheme(Scheme))

	utilruntime.Must(Scheme.SetVersionPriority(corev1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(admissionv1beta1.SchemeGroupVersion, admissionv1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(admissionregistrationv1beta1.SchemeGroupVersion, admissionregistrationv1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(appsv1beta1.SchemeGroupVersion, appsv1beta2.SchemeGroupVersion, appsv1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(authenticationv1.SchemeGroupVersion, authenticationv1beta1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(authorizationv1.SchemeGroupVersion, authorizationv1beta1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(autoscalingv1.SchemeGroupVersion, autoscalingv2beta1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(batchv1.SchemeGroupVersion, batchv1beta1.SchemeGroupVersion, batchv2alpha1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(certificatesv1beta1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(extensionsv1beta1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(imagepolicyv1alpha1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(networkingv1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(policyv1beta1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(rbacv1.SchemeGroupVersion, rbacv1beta1.SchemeGroupVersion, rbacv1alpha1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(schedulingv1alpha1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(settingsv1alpha1.SchemeGroupVersion))
	utilruntime.Must(Scheme.SetVersionPriority(storagev1.SchemeGroupVersion, storagev1beta1.SchemeGroupVersion))
}
