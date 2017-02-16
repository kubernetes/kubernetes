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
	runtime "k8s.io/apimachinery/pkg/runtime"
	serializer "k8s.io/apimachinery/pkg/runtime/serializer"
	corev1 "k8s.io/kubernetes/pkg/api/v1"
	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	authenticationv1 "k8s.io/kubernetes/pkg/apis/authentication/v1"
	authenticationv1beta1 "k8s.io/kubernetes/pkg/apis/authentication/v1beta1"
	authorizationv1 "k8s.io/kubernetes/pkg/apis/authorization/v1"
	authorizationv1beta1 "k8s.io/kubernetes/pkg/apis/authorization/v1beta1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	autoscalingv2alpha1 "k8s.io/kubernetes/pkg/apis/autoscaling/v2alpha1"
	batchv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batchv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	certificatesv1beta1 "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	policyv1beta1 "k8s.io/kubernetes/pkg/apis/policy/v1beta1"
	rbacv1alpha1 "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	storagev1beta1 "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
)

var Scheme = runtime.NewScheme()
var Codecs = serializer.NewCodecFactory(Scheme)
var ParameterCodec = runtime.NewParameterCodec(Scheme)

func init() {
	corev1.AddToScheme(Scheme)
	appsv1beta1.AddToScheme(Scheme)
	authenticationv1.AddToScheme(Scheme)
	authenticationv1beta1.AddToScheme(Scheme)
	authorizationv1.AddToScheme(Scheme)
	authorizationv1beta1.AddToScheme(Scheme)
	autoscalingv1.AddToScheme(Scheme)
	autoscalingv2alpha1.AddToScheme(Scheme)
	batchv1.AddToScheme(Scheme)
	batchv2alpha1.AddToScheme(Scheme)
	certificatesv1beta1.AddToScheme(Scheme)
	extensionsv1beta1.AddToScheme(Scheme)
	policyv1beta1.AddToScheme(Scheme)
	rbacv1beta1.AddToScheme(Scheme)
	rbacv1alpha1.AddToScheme(Scheme)
	storagev1beta1.AddToScheme(Scheme)

}
