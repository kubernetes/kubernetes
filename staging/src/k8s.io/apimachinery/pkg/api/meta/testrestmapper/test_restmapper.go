/*
Copyright 2018 The Kubernetes Authors.

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

package testrestmapper

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// TestOnlyStaticRESTMapper returns a union RESTMapper of all known types with priorities chosen in the following order:
//  1. legacy kube group preferred version, extensions preferred version, metrics preferred version, legacy
//     kube any version, extensions any version, metrics any version, all other groups alphabetical preferred version,
//     all other groups alphabetical.
// TODO callers of this method should be updated to build their own specific restmapper based on their scheme for their tests
// TODO the things being tested are related to whether various cases are handled, not tied to the particular types being checked.
func TestOnlyStaticRESTMapper(scheme *runtime.Scheme, versionPatterns ...schema.GroupVersion) meta.RESTMapper {
	unionMapper := meta.MultiRESTMapper{}
	unionedGroups := sets.NewString()
	for _, enabledVersion := range scheme.PrioritizedVersionsAllGroups() {
		if !unionedGroups.Has(enabledVersion.Group) {
			unionedGroups.Insert(enabledVersion.Group)
			unionMapper = append(unionMapper, newRESTMapper(enabledVersion.Group, scheme))
		}
	}

	if len(versionPatterns) != 0 {
		resourcePriority := []schema.GroupVersionResource{}
		kindPriority := []schema.GroupVersionKind{}
		for _, versionPriority := range versionPatterns {
			resourcePriority = append(resourcePriority, versionPriority.WithResource(meta.AnyResource))
			kindPriority = append(kindPriority, versionPriority.WithKind(meta.AnyKind))
		}

		return meta.PriorityRESTMapper{Delegate: unionMapper, ResourcePriority: resourcePriority, KindPriority: kindPriority}
	}

	prioritizedGroups := []string{"", "extensions", "metrics"}
	resourcePriority, kindPriority := prioritiesForGroups(scheme, prioritizedGroups...)

	prioritizedGroupsSet := sets.NewString(prioritizedGroups...)
	remainingGroups := sets.String{}
	for _, enabledVersion := range scheme.PrioritizedVersionsAllGroups() {
		if !prioritizedGroupsSet.Has(enabledVersion.Group) {
			remainingGroups.Insert(enabledVersion.Group)
		}
	}

	remainingResourcePriority, remainingKindPriority := prioritiesForGroups(scheme, remainingGroups.List()...)
	resourcePriority = append(resourcePriority, remainingResourcePriority...)
	kindPriority = append(kindPriority, remainingKindPriority...)

	return meta.PriorityRESTMapper{Delegate: unionMapper, ResourcePriority: resourcePriority, KindPriority: kindPriority}
}

// prioritiesForGroups returns the resource and kind priorities for a PriorityRESTMapper, preferring the preferred version of each group first,
// then any non-preferred version of the group second.
func prioritiesForGroups(scheme *runtime.Scheme, groups ...string) ([]schema.GroupVersionResource, []schema.GroupVersionKind) {
	resourcePriority := []schema.GroupVersionResource{}
	kindPriority := []schema.GroupVersionKind{}

	for _, group := range groups {
		availableVersions := scheme.PrioritizedVersionsForGroup(group)
		if len(availableVersions) > 0 {
			resourcePriority = append(resourcePriority, availableVersions[0].WithResource(meta.AnyResource))
			kindPriority = append(kindPriority, availableVersions[0].WithKind(meta.AnyKind))
		}
	}
	for _, group := range groups {
		resourcePriority = append(resourcePriority, schema.GroupVersionResource{Group: group, Version: meta.AnyVersion, Resource: meta.AnyResource})
		kindPriority = append(kindPriority, schema.GroupVersionKind{Group: group, Version: meta.AnyVersion, Kind: meta.AnyKind})
	}

	return resourcePriority, kindPriority
}

func newRESTMapper(group string, scheme *runtime.Scheme) meta.RESTMapper {
	mapper := meta.NewDefaultRESTMapper(scheme.PrioritizedVersionsForGroup(group))
	for _, gv := range scheme.PrioritizedVersionsForGroup(group) {
		for kind := range scheme.KnownTypes(gv) {
			if ignoredKinds.Has(kind) {
				continue
			}
			scope := meta.RESTScopeNamespace
			if rootScopedKinds[gv.WithKind(kind).GroupKind()] {
				scope = meta.RESTScopeRoot
			}
			mapper.Add(gv.WithKind(kind), scope)
		}
	}

	return mapper
}

// hardcoded is good enough for the test we're running
var rootScopedKinds = map[schema.GroupKind]bool{
	{Group: "admission.k8s.io", Kind: "AdmissionReview"}: true,

	{Group: "admissionregistration.k8s.io", Kind: "ValidatingWebhookConfiguration"}: true,
	{Group: "admissionregistration.k8s.io", Kind: "MutatingWebhookConfiguration"}:   true,

	{Group: "authentication.k8s.io", Kind: "TokenReview"}: true,

	{Group: "authorization.k8s.io", Kind: "SubjectAccessReview"}:     true,
	{Group: "authorization.k8s.io", Kind: "SelfSubjectAccessReview"}: true,
	{Group: "authorization.k8s.io", Kind: "SelfSubjectRulesReview"}:  true,

	{Group: "certificates.k8s.io", Kind: "CertificateSigningRequest"}: true,

	{Group: "", Kind: "Node"}:             true,
	{Group: "", Kind: "Namespace"}:        true,
	{Group: "", Kind: "PersistentVolume"}: true,
	{Group: "", Kind: "ComponentStatus"}:  true,

	{Group: "extensions", Kind: "PodSecurityPolicy"}: true,

	{Group: "policy", Kind: "PodSecurityPolicy"}: true,

	{Group: "extensions", Kind: "PodSecurityPolicy"}: true,

	{Group: "rbac.authorization.k8s.io", Kind: "ClusterRole"}:        true,
	{Group: "rbac.authorization.k8s.io", Kind: "ClusterRoleBinding"}: true,

	{Group: "scheduling.k8s.io", Kind: "PriorityClass"}: true,

	{Group: "storage.k8s.io", Kind: "StorageClass"}:     true,
	{Group: "storage.k8s.io", Kind: "VolumeAttachment"}: true,

	{Group: "apiextensions.k8s.io", Kind: "CustomResourceDefinition"}: true,

	{Group: "apiserver.k8s.io", Kind: "AdmissionConfiguration"}: true,

	{Group: "audit.k8s.io", Kind: "Event"}:  true,
	{Group: "audit.k8s.io", Kind: "Policy"}: true,

	{Group: "apiregistration.k8s.io", Kind: "APIService"}: true,

	{Group: "metrics.k8s.io", Kind: "NodeMetrics"}: true,

	{Group: "wardle.k8s.io", Kind: "Fischer"}: true,
}

// hardcoded is good enough for the test we're running
var ignoredKinds = sets.NewString(
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
)
