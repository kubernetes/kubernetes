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
	admissionv1alpha1 "k8s.io/api/admission/v1beta1"
	admissionregistrationv1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
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
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/scheme"
)

// Register all groups in the kubectl's registry, but no componentconfig group since it's not in k8s.io/api
// The code in this file mostly duplicate the install under k8s.io/kubernetes/pkg/api and k8s.io/kubernetes/pkg/apis,
// but does NOT register the internal types.
func init() {
	// Register external types for Scheme
	v1.AddToGroupVersion(Scheme, schema.GroupVersion{Version: "v1"})
	scheme.AddToScheme(Scheme)

	// Register external types for Registry
	Versions = append(Versions, corev1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              corev1.GroupName,
			VersionPreferenceOrder: []string{corev1.SchemeGroupVersion.Version},
			RootScopedKinds: sets.NewString(
				"Node",
				"Namespace",
				"PersistentVolume",
				"ComponentStatus",
			),
			IgnoredKinds: sets.NewString(
				"ListOptions",
				"DeleteOptions",
				"Status",
				"PodLogOptions",
				"PodExecOptions",
				"PodAttachOptions",
				"PodPortForwardOptions",
				"PodProxyOptions",
				"NodeProxyOptions",
				"ServiceProxyOptions",
			),
		},
		announced.VersionToSchemeFunc{
			corev1.SchemeGroupVersion.Version: corev1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, admissionv1alpha1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              admissionv1alpha1.GroupName,
			VersionPreferenceOrder: []string{admissionv1alpha1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("AdmissionReview"),
		},
		announced.VersionToSchemeFunc{
			admissionv1alpha1.SchemeGroupVersion.Version: admissionv1alpha1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, admissionregistrationv1alpha1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              admissionregistrationv1alpha1.GroupName,
			RootScopedKinds:        sets.NewString("InitializerConfiguration", "ValidatingWebhookConfiguration", "MutatingWebhookConfiguration"),
			VersionPreferenceOrder: []string{admissionregistrationv1alpha1.SchemeGroupVersion.Version},
		},
		announced.VersionToSchemeFunc{
			admissionregistrationv1alpha1.SchemeGroupVersion.Version: admissionregistrationv1alpha1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, appsv1.SchemeGroupVersion, appsv1beta2.SchemeGroupVersion, appsv1beta1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              appsv1.GroupName,
			VersionPreferenceOrder: []string{appsv1beta1.SchemeGroupVersion.Version, appsv1beta2.SchemeGroupVersion.Version, appsv1.SchemeGroupVersion.Version},
		},
		announced.VersionToSchemeFunc{
			appsv1beta1.SchemeGroupVersion.Version: appsv1beta1.AddToScheme,
			appsv1beta2.SchemeGroupVersion.Version: appsv1beta2.AddToScheme,
			appsv1.SchemeGroupVersion.Version:      appsv1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, authenticationv1.SchemeGroupVersion, authenticationv1beta1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              authenticationv1beta1.GroupName,
			VersionPreferenceOrder: []string{authenticationv1.SchemeGroupVersion.Version, authenticationv1beta1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("TokenReview"),
		},
		announced.VersionToSchemeFunc{
			authenticationv1beta1.SchemeGroupVersion.Version: authenticationv1beta1.AddToScheme,
			authenticationv1.SchemeGroupVersion.Version:      authenticationv1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, authorizationv1.SchemeGroupVersion, authorizationv1beta1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              authorizationv1.GroupName,
			VersionPreferenceOrder: []string{authorizationv1.SchemeGroupVersion.Version, authorizationv1beta1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("SubjectAccessReview", "SelfSubjectAccessReview", "SelfSubjectRulesReview"),
		},
		announced.VersionToSchemeFunc{
			authorizationv1beta1.SchemeGroupVersion.Version: authorizationv1beta1.AddToScheme,
			authorizationv1.SchemeGroupVersion.Version:      authorizationv1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, autoscalingv1.SchemeGroupVersion, autoscalingv2beta1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              autoscalingv1.GroupName,
			VersionPreferenceOrder: []string{autoscalingv1.SchemeGroupVersion.Version, autoscalingv2beta1.SchemeGroupVersion.Version},
		},
		announced.VersionToSchemeFunc{
			autoscalingv1.SchemeGroupVersion.Version:      autoscalingv1.AddToScheme,
			autoscalingv2beta1.SchemeGroupVersion.Version: autoscalingv2beta1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, batchv1.SchemeGroupVersion, batchv1beta1.SchemeGroupVersion, batchv2alpha1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              batchv1.GroupName,
			VersionPreferenceOrder: []string{batchv1.SchemeGroupVersion.Version, batchv1beta1.SchemeGroupVersion.Version, batchv2alpha1.SchemeGroupVersion.Version},
		},
		announced.VersionToSchemeFunc{
			batchv1.SchemeGroupVersion.Version:       batchv1.AddToScheme,
			batchv1beta1.SchemeGroupVersion.Version:  batchv1beta1.AddToScheme,
			batchv2alpha1.SchemeGroupVersion.Version: batchv2alpha1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, certificatesv1beta1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              certificatesv1beta1.GroupName,
			VersionPreferenceOrder: []string{certificatesv1beta1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("CertificateSigningRequest"),
		},
		announced.VersionToSchemeFunc{
			certificatesv1beta1.SchemeGroupVersion.Version: certificatesv1beta1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, extensionsv1beta1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              extensionsv1beta1.GroupName,
			VersionPreferenceOrder: []string{extensionsv1beta1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("PodSecurityPolicy"),
		},
		announced.VersionToSchemeFunc{
			extensionsv1beta1.SchemeGroupVersion.Version: extensionsv1beta1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, imagepolicyv1alpha1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              imagepolicyv1alpha1.GroupName,
			VersionPreferenceOrder: []string{imagepolicyv1alpha1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("ImageReview"),
		},
		announced.VersionToSchemeFunc{
			imagepolicyv1alpha1.SchemeGroupVersion.Version: imagepolicyv1alpha1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, networkingv1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              networkingv1.GroupName,
			VersionPreferenceOrder: []string{networkingv1.SchemeGroupVersion.Version},
		},
		announced.VersionToSchemeFunc{
			networkingv1.SchemeGroupVersion.Version: networkingv1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, policyv1beta1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              policyv1beta1.GroupName,
			VersionPreferenceOrder: []string{policyv1beta1.SchemeGroupVersion.Version},
		},
		announced.VersionToSchemeFunc{
			policyv1beta1.SchemeGroupVersion.Version: policyv1beta1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, rbacv1.SchemeGroupVersion, rbacv1beta1.SchemeGroupVersion, rbacv1alpha1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              rbacv1.GroupName,
			VersionPreferenceOrder: []string{rbacv1.SchemeGroupVersion.Version, rbacv1beta1.SchemeGroupVersion.Version, rbacv1alpha1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("ClusterRole", "ClusterRoleBinding"),
		},
		announced.VersionToSchemeFunc{
			rbacv1.SchemeGroupVersion.Version:       rbacv1.AddToScheme,
			rbacv1beta1.SchemeGroupVersion.Version:  rbacv1beta1.AddToScheme,
			rbacv1alpha1.SchemeGroupVersion.Version: rbacv1alpha1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, schedulingv1alpha1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              schedulingv1alpha1.GroupName,
			VersionPreferenceOrder: []string{schedulingv1alpha1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("PriorityClass"),
		},
		announced.VersionToSchemeFunc{
			schedulingv1alpha1.SchemeGroupVersion.Version: schedulingv1alpha1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, settingsv1alpha1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              settingsv1alpha1.GroupName,
			VersionPreferenceOrder: []string{settingsv1alpha1.SchemeGroupVersion.Version},
		},
		announced.VersionToSchemeFunc{
			settingsv1alpha1.SchemeGroupVersion.Version: settingsv1alpha1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}

	Versions = append(Versions, storagev1.SchemeGroupVersion, storagev1beta1.SchemeGroupVersion)
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:              storagev1.GroupName,
			VersionPreferenceOrder: []string{storagev1.SchemeGroupVersion.Version, storagev1beta1.SchemeGroupVersion.Version},
			RootScopedKinds:        sets.NewString("StorageClass"),
		},
		announced.VersionToSchemeFunc{
			storagev1.SchemeGroupVersion.Version:      storagev1.AddToScheme,
			storagev1beta1.SchemeGroupVersion.Version: storagev1beta1.AddToScheme,
		},
	).Announce(GroupFactoryRegistry).RegisterAndEnable(Registry, Scheme); err != nil {
		panic(err)
	}
}
