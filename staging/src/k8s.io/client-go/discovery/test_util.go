/*
Copyright 2022 The Kubernetes Authors.

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

package discovery

import (
	"fmt"

	apidiscovery "k8s.io/api/apidiscovery/v2beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	CoreV1GroupName    = ""
	CoreV1GroupVersion = "v1"
)

var CoreV1Service = apidiscovery.APIResourceDiscovery{
	Resource: "services",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "Service",
	},
	Scope: apidiscovery.ScopeNamespace,
}

var AppsV1Deployment = apidiscovery.APIResourceDiscovery{
	Resource: "deployments",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "apps",
		Version: "v1",
		Kind:    "Deployment",
	},
	Scope: apidiscovery.ScopeNamespace,
}

var AppsV1StatefulSet = apidiscovery.APIResourceDiscovery{
	Resource: "statefulsets",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "apps",
		Version: "v1",
		Kind:    "StatefulSet",
	},
	Scope: apidiscovery.ScopeNamespace,
}

var AppsV2Deployment = apidiscovery.APIResourceDiscovery{
	Resource: "deployments",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "apps",
		Version: "v2",
		Kind:    "Deployment",
	},
	Scope: apidiscovery.ScopeNamespace,
}

var BatchV1Job = apidiscovery.APIResourceDiscovery{
	Resource: "jobs",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "batch",
		Version: "v1",
		Kind:    "Job",
	},
	Scope: apidiscovery.ScopeNamespace,
}

var BatchV1CronJob = apidiscovery.APIResourceDiscovery{
	Resource: "cronjobs",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "batch",
		Version: "v1",
		Kind:    "CronJob",
	},
	Scope: apidiscovery.ScopeNamespace,
}

var BatchV1Beta1Job = apidiscovery.APIResourceDiscovery{
	Resource: "jobs",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "batch",
		Version: "v1beta1",
		Kind:    "Job",
	},
	Scope: apidiscovery.ScopeNamespace,
}

var CoreV1Pod = apidiscovery.APIResourceDiscovery{
	Resource: "pods",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "Pod",
	},
	Scope: apidiscovery.ScopeNamespace,
}

var BatchV1Beta1CronJob = apidiscovery.APIResourceDiscovery{
	Resource: "cronjobs",
	ResponseKind: &metav1.GroupVersionKind{
		Group:   "batch",
		Version: "v1beta1",
		Kind:    "CronJob",
	},
	Scope: apidiscovery.ScopeNamespace,
}

func GroupNames(groups []*metav1.APIGroup) []string {
	result := []string{}
	for _, group := range groups {
		result = append(result, group.Name)
	}
	return result
}

func GroupNamesFromList(groups *metav1.APIGroupList) []string {
	result := []string{}
	for _, group := range groups.Groups {
		result = append(result, group.Name)
	}
	return result
}

func PreferredVersionsFromList(groups *metav1.APIGroupList) []string {
	result := []string{}
	for _, group := range groups.Groups {
		preferredGV := group.PreferredVersion.GroupVersion
		result = append(result, preferredGV)
	}
	return result
}

func GroupVersions(resources []*metav1.APIResourceList) []string {
	result := []string{}
	for _, resourceList := range resources {
		result = append(result, resourceList.GroupVersion)
	}
	return result
}

func GroupVersionsFromGroups(groups *metav1.APIGroupList) []string {
	result := []string{}
	for _, group := range groups.Groups {
		for _, version := range group.Versions {
			result = append(result, version.GroupVersion)
		}
	}
	return result
}

func GroupVersionKinds(resources []*metav1.APIResourceList) []string {
	result := []string{}
	for _, resourceList := range resources {
		for _, resource := range resourceList.APIResources {
			gvk := fmt.Sprintf("%s/%s/%s", resource.Group, resource.Version, resource.Kind)
			result = append(result, gvk)
		}
	}
	return result
}
