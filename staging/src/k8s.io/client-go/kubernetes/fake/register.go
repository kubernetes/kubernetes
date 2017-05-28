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

package fake

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	serializer "k8s.io/apimachinery/pkg/runtime/serializer"
	corev1 "k8s.io/client-go/pkg/api/v1"
	admissionregistrationv1alpha1 "k8s.io/client-go/pkg/apis/admissionregistration/v1alpha1"
	appsv1beta1 "k8s.io/client-go/pkg/apis/apps/v1beta1"
	authenticationv1 "k8s.io/client-go/pkg/apis/authentication/v1"
	authenticationv1beta1 "k8s.io/client-go/pkg/apis/authentication/v1beta1"
	authorizationv1 "k8s.io/client-go/pkg/apis/authorization/v1"
	authorizationv1beta1 "k8s.io/client-go/pkg/apis/authorization/v1beta1"
	autoscalingv1 "k8s.io/client-go/pkg/apis/autoscaling/v1"
	autoscalingv2alpha1 "k8s.io/client-go/pkg/apis/autoscaling/v2alpha1"
	batchv1 "k8s.io/client-go/pkg/apis/batch/v1"
	batchv2alpha1 "k8s.io/client-go/pkg/apis/batch/v2alpha1"
	certificatesv1beta1 "k8s.io/client-go/pkg/apis/certificates/v1beta1"
	extensionsv1beta1 "k8s.io/client-go/pkg/apis/extensions/v1beta1"
	networkingv1 "k8s.io/client-go/pkg/apis/networking/v1"
	policyv1beta1 "k8s.io/client-go/pkg/apis/policy/v1beta1"
	rbacv1alpha1 "k8s.io/client-go/pkg/apis/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/client-go/pkg/apis/rbac/v1beta1"
	settingsv1alpha1 "k8s.io/client-go/pkg/apis/settings/v1alpha1"
	storagev1 "k8s.io/client-go/pkg/apis/storage/v1"
	storagev1beta1 "k8s.io/client-go/pkg/apis/storage/v1beta1"
)

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)
var parameterCodec = runtime.NewParameterCodec(scheme)

func init() {
	v1.AddToGroupVersion(scheme, schema.GroupVersion{Version: "v1"})
	AddToScheme(scheme)
}

// AddToScheme adds all types of this clientset into the given scheme. This allows composition
// of clientsets, like in:
//
//   import (
//     "k8s.io/client-go/kubernetes"
//     clientsetscheme "k8s.io/client-go/kuberentes/scheme"
//     aggregatorclientsetscheme "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/scheme"
//   )
//
//   kclientset, _ := kubernetes.NewForConfig(c)
//   aggregatorclientsetscheme.AddToScheme(clientsetscheme.Scheme)
//
// After this, RawExtensions in Kubernetes types will serialize kube-aggregator types
// correctly.
func AddToScheme(scheme *runtime.Scheme) {
	admissionregistrationv1alpha1.AddToScheme(scheme)
	corev1.AddToScheme(scheme)
	appsv1beta1.AddToScheme(scheme)
	authenticationv1.AddToScheme(scheme)
	authenticationv1beta1.AddToScheme(scheme)
	authorizationv1.AddToScheme(scheme)
	authorizationv1beta1.AddToScheme(scheme)
	autoscalingv1.AddToScheme(scheme)
	autoscalingv2alpha1.AddToScheme(scheme)
	batchv1.AddToScheme(scheme)
	batchv2alpha1.AddToScheme(scheme)
	certificatesv1beta1.AddToScheme(scheme)
	extensionsv1beta1.AddToScheme(scheme)
	networkingv1.AddToScheme(scheme)
	policyv1beta1.AddToScheme(scheme)
	rbacv1beta1.AddToScheme(scheme)
	rbacv1alpha1.AddToScheme(scheme)
	settingsv1alpha1.AddToScheme(scheme)
	storagev1beta1.AddToScheme(scheme)
	storagev1.AddToScheme(scheme)

}
