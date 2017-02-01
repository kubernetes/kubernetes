/*
Copyright 2016 The Kubernetes Authors.

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

package openapi

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"unicode"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/openapi"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/util/trie"
	"sort"
)

var verbs = trie.New([]string{"get", "log", "read", "replace", "patch", "delete", "deletecollection", "watch", "connect", "proxy", "list", "create", "patch"})

const (
	extensionGVK = "x-kubernetes-group-version-kind"
)

// ToValidOperationID makes an string a valid op ID (e.g. removing punctuations and whitespaces and make it camel case)
func ToValidOperationID(s string, capitalizeFirstLetter bool) string {
	var buffer bytes.Buffer
	capitalize := capitalizeFirstLetter
	for i, r := range s {
		if unicode.IsLetter(r) || r == '_' || (i != 0 && unicode.IsDigit(r)) {
			if capitalize {
				buffer.WriteRune(unicode.ToUpper(r))
				capitalize = false
			} else {
				buffer.WriteRune(r)
			}
		} else {
			capitalize = true
		}
	}
	return buffer.String()
}

// GetOperationIDAndTags returns a customize operation ID and a list of tags for kubernetes API server's OpenAPI spec to prevent duplicate IDs.
func GetOperationIDAndTags(servePath string, r *restful.Route) (string, []string, error) {
	op := r.Operation
	path := r.Path
	var tags []string
	// TODO: This is hacky, figure out where this name conflict is created and fix it at the root.
	if strings.HasPrefix(path, "/apis/extensions/v1beta1/namespaces/{namespace}/") && strings.HasSuffix(op, "ScaleScale") {
		op = op[:len(op)-10] + strings.Title(strings.Split(path[48:], "/")[0]) + "Scale"
	}
	switch servePath {
	case "/swagger.json":
		prefix, exists := verbs.GetPrefix(op)
		if !exists {
			return op, tags, fmt.Errorf("operation names should start with a verb. Cannot determine operation verb from %v", op)
		}
		op = op[len(prefix):]
		parts := strings.Split(strings.Trim(path, "/"), "/")
		// Assume /api is /apis/core, remove this when we actually server /api/... on /apis/core/...
		if len(parts) >= 1 && parts[0] == "api" {
			parts = append([]string{"apis", "core"}, parts[1:]...)
		}
		if len(parts) >= 2 && parts[0] == "apis" {
			prefix = prefix + ToValidOperationID(strings.TrimSuffix(parts[1], ".k8s.io"), prefix != "")
			tag := ToValidOperationID(strings.TrimSuffix(parts[1], ".k8s.io"), false)
			if len(parts) > 2 {
				prefix = prefix + ToValidOperationID(parts[2], prefix != "")
				tag = tag + "_" + ToValidOperationID(parts[2], false)
			}
			tags = append(tags, tag)
		} else if len(parts) >= 1 {
			tags = append(tags, ToValidOperationID(parts[0], false))
		}
		return prefix + ToValidOperationID(op, prefix != ""), tags, nil
	default:
		return op, tags, nil
	}
}

type groupVersionKinds []v1.GroupVersionKind

func (s groupVersionKinds) Len() int {
	return len(s)
}

func (s groupVersionKinds) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s groupVersionKinds) Less(i, j int) bool {
	if s[i].Group == s[j].Group {
		if s[i].Version == s[j].Version {
			return s[i].Kind < s[j].Kind
		}
		return s[i].Version < s[j].Version
	}
	return s[i].Group < s[j].Group
}

// DefinitionNamer is the type to customize OpenAPI definition name.
type DefinitionNamer struct {
	typeGroupVersionKinds map[string]groupVersionKinds
}

func gvkConvert(gvk schema.GroupVersionKind) v1.GroupVersionKind {
	return v1.GroupVersionKind{
		Group:   gvk.Group,
		Version: gvk.Version,
		Kind:    gvk.Kind,
	}
}

func friendlyName(name string) string {
	nameParts := strings.Split(name, "/")
	// Reverse first part. e.g., io.k8s... instead of k8s.io...
	if len(nameParts) > 0 && strings.Contains(nameParts[0], ".") {
		parts := strings.Split(nameParts[0], ".")
		for i, j := 0, len(parts)-1; i < j; i, j = i+1, j-1 {
			parts[i], parts[j] = parts[j], parts[i]
		}
		nameParts[0] = strings.Join(parts, ".")
	}
	return strings.Join(nameParts, ".")
}

func typeName(t reflect.Type) string {
	return fmt.Sprintf("%s.%s", t.PkgPath(), t.Name())
}

// NewDefinitionNamer constructs a new DefinitionNamer to be used to customize OpenAPI spec.
func NewDefinitionNamer(s *runtime.Scheme) DefinitionNamer {
	ret := DefinitionNamer{
		typeGroupVersionKinds: map[string]groupVersionKinds{},
	}
	for gvk, rtype := range s.AllKnownTypes() {
		ret.typeGroupVersionKinds[typeName(rtype)] = append(ret.typeGroupVersionKinds[typeName(rtype)], gvkConvert(gvk))
	}
	for _, gvk := range ret.typeGroupVersionKinds {
		sort.Sort(gvk)
	}
	return ret
}

// GetDefinitionName returns the name and tags for a given definition
func (d *DefinitionNamer) GetDefinitionName(servePath string, name string) (string, spec.Extensions) {
	if groupVersionKinds, ok := d.typeGroupVersionKinds[name]; ok {
		return friendlyName(name), spec.Extensions{
			extensionGVK: []v1.GroupVersionKind(groupVersionKinds),
		}
	}
	return friendlyName(name), nil
}

// PostProcessSpec finalize OpenAPI spec and add removed definition for backward compatibility
func PostProcessSpec(s *spec.Swagger) (*spec.Swagger, error) {
	compatibilityMap := map[string]string{
		"v1beta1.DeploymentStatus":            "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentStatus",
		"v1beta1.ReplicaSetList":              "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSetList",
		"v1beta1.Eviction":                    "k8s.io/kubernetes/pkg/apis/policy/v1beta1.Eviction",
		"v1beta1.StatefulSetList":             "k8s.io/kubernetes/pkg/apis/apps/v1beta1.StatefulSetList",
		"v1beta1.RoleBinding":                 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.RoleBinding",
		"v1beta1.PodSecurityPolicyList":       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.PodSecurityPolicyList",
		"v1.NodeSpec":                         "k8s.io/kubernetes/pkg/api/v1.NodeSpec",
		"v1.FlockerVolumeSource":              "k8s.io/kubernetes/pkg/api/v1.FlockerVolumeSource",
		"v1.ContainerState":                   "k8s.io/kubernetes/pkg/api/v1.ContainerState",
		"v1beta1.ClusterRole":                 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.ClusterRole",
		"v1beta1.StorageClass":                "k8s.io/kubernetes/pkg/apis/storage/v1beta1.StorageClass",
		"v1.FlexVolumeSource":                 "k8s.io/kubernetes/pkg/api/v1.FlexVolumeSource",
		"v1.SecretKeySelector":                "k8s.io/kubernetes/pkg/api/v1.SecretKeySelector",
		"v1.DeleteOptions":                    "k8s.io/kubernetes/pkg/api/v1.DeleteOptions",
		"v1.PodStatus":                        "k8s.io/kubernetes/pkg/api/v1.PodStatus",
		"v1.NodeStatus":                       "k8s.io/kubernetes/pkg/api/v1.NodeStatus",
		"v1.ServiceSpec":                      "k8s.io/kubernetes/pkg/api/v1.ServiceSpec",
		"v1.AttachedVolume":                   "k8s.io/kubernetes/pkg/api/v1.AttachedVolume",
		"v1.PersistentVolume":                 "k8s.io/kubernetes/pkg/api/v1.PersistentVolume",
		"v1.LimitRangeList":                   "k8s.io/kubernetes/pkg/api/v1.LimitRangeList",
		"v1alpha1.Role":                       "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.Role",
		"v1.Affinity":                         "k8s.io/kubernetes/pkg/api/v1.Affinity",
		"v1beta1.PodDisruptionBudget":         "k8s.io/kubernetes/pkg/apis/policy/v1beta1.PodDisruptionBudget",
		"v1alpha1.RoleBindingList":            "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.RoleBindingList",
		"v1.PodAffinity":                      "k8s.io/kubernetes/pkg/api/v1.PodAffinity",
		"v1beta1.SELinuxStrategyOptions":      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.SELinuxStrategyOptions",
		"v1.ResourceQuotaList":                "k8s.io/kubernetes/pkg/api/v1.ResourceQuotaList",
		"v1.PodList":                          "k8s.io/kubernetes/pkg/api/v1.PodList",
		"v1.EnvVarSource":                     "k8s.io/kubernetes/pkg/api/v1.EnvVarSource",
		"v1beta1.TokenReviewStatus":           "k8s.io/kubernetes/pkg/apis/authentication/v1beta1.TokenReviewStatus",
		"v1.PersistentVolumeClaimList":        "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaimList",
		"v1beta1.RoleList":                    "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.RoleList",
		"v1.ListMeta":                         "k8s.io/apimachinery/pkg/apis/meta/v1.ListMeta",
		"v1.ObjectMeta":                       "k8s.io/apimachinery/pkg/apis/meta/v1.ObjectMeta",
		"v1.APIGroupList":                     "k8s.io/apimachinery/pkg/apis/meta/v1.APIGroupList",
		"v2alpha1.Job":                        "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.Job",
		"v1.EnvFromSource":                    "k8s.io/kubernetes/pkg/api/v1.EnvFromSource",
		"v1beta1.IngressStatus":               "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressStatus",
		"v1.Service":                          "k8s.io/kubernetes/pkg/api/v1.Service",
		"v1beta1.DaemonSetStatus":             "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DaemonSetStatus",
		"v1alpha1.Subject":                    "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.Subject",
		"v1.HorizontalPodAutoscaler":          "k8s.io/kubernetes/pkg/apis/autoscaling/v1.HorizontalPodAutoscaler",
		"v1.StatusCause":                      "k8s.io/apimachinery/pkg/apis/meta/v1.StatusCause",
		"v1.NodeSelectorRequirement":          "k8s.io/kubernetes/pkg/api/v1.NodeSelectorRequirement",
		"v1beta1.NetworkPolicyIngressRule":    "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicyIngressRule",
		"v1beta1.ThirdPartyResource":          "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ThirdPartyResource",
		"v1beta1.PodSecurityPolicy":           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.PodSecurityPolicy",
		"v1beta1.StatefulSet":                 "k8s.io/kubernetes/pkg/apis/apps/v1beta1.StatefulSet",
		"v1.LabelSelector":                    "k8s.io/apimachinery/pkg/apis/meta/v1.LabelSelector",
		"v1.ScaleSpec":                        "k8s.io/kubernetes/pkg/apis/autoscaling/v1.ScaleSpec",
		"v1.DownwardAPIVolumeFile":            "k8s.io/kubernetes/pkg/api/v1.DownwardAPIVolumeFile",
		"v1beta1.HorizontalPodAutoscaler":     "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HorizontalPodAutoscaler",
		"v1.AWSElasticBlockStoreVolumeSource": "k8s.io/kubernetes/pkg/api/v1.AWSElasticBlockStoreVolumeSource",
		"v1.ComponentStatus":                  "k8s.io/kubernetes/pkg/api/v1.ComponentStatus",
		"v2alpha1.JobSpec":                    "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobSpec",
		"v1.ContainerImage":                   "k8s.io/kubernetes/pkg/api/v1.ContainerImage",
		"v1.ReplicationControllerStatus":      "k8s.io/kubernetes/pkg/api/v1.ReplicationControllerStatus",
		"v1.ResourceQuota":                    "k8s.io/kubernetes/pkg/api/v1.ResourceQuota",
		"v1beta1.NetworkPolicyList":           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicyList",
		"v1beta1.NonResourceAttributes":       "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.NonResourceAttributes",
		"v1.JobCondition":                     "k8s.io/kubernetes/pkg/apis/batch/v1.JobCondition",
		"v1.LabelSelectorRequirement":         "k8s.io/apimachinery/pkg/apis/meta/v1.LabelSelectorRequirement",
		"v1beta1.Deployment":                  "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.Deployment",
		"v1.LoadBalancerIngress":              "k8s.io/kubernetes/pkg/api/v1.LoadBalancerIngress",
		"v1.SecretList":                       "k8s.io/kubernetes/pkg/api/v1.SecretList",
		"v1beta1.ReplicaSetSpec":              "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSetSpec",
		"v1beta1.RoleBindingList":             "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.RoleBindingList",
		"v1.ServicePort":                      "k8s.io/kubernetes/pkg/api/v1.ServicePort",
		"v1.Namespace":                        "k8s.io/kubernetes/pkg/api/v1.Namespace",
		"v1beta1.NetworkPolicyPeer":           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicyPeer",
		"v1.ReplicationControllerList":        "k8s.io/kubernetes/pkg/api/v1.ReplicationControllerList",
		"v1beta1.ReplicaSetCondition":         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSetCondition",
		"v1.ReplicationControllerCondition":   "k8s.io/kubernetes/pkg/api/v1.ReplicationControllerCondition",
		"v1.DaemonEndpoint":                   "k8s.io/kubernetes/pkg/api/v1.DaemonEndpoint",
		"v1beta1.NetworkPolicyPort":           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicyPort",
		"v1.NodeSystemInfo":                   "k8s.io/kubernetes/pkg/api/v1.NodeSystemInfo",
		"v1.LimitRangeItem":                   "k8s.io/kubernetes/pkg/api/v1.LimitRangeItem",
		"v1.ConfigMapVolumeSource":            "k8s.io/kubernetes/pkg/api/v1.ConfigMapVolumeSource",
		"v1beta1.ClusterRoleList":             "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.ClusterRoleList",
		"v1beta1.ResourceAttributes":          "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.ResourceAttributes",
		"v1.Pod":                              "k8s.io/kubernetes/pkg/api/v1.Pod",
		"v1.FCVolumeSource":                   "k8s.io/kubernetes/pkg/api/v1.FCVolumeSource",
		"v1beta1.SubresourceReference":        "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.SubresourceReference",
		"v1.ResourceQuotaStatus":              "k8s.io/kubernetes/pkg/api/v1.ResourceQuotaStatus",
		"v1alpha1.RoleBinding":                "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.RoleBinding",
		"v1.PodCondition":                     "k8s.io/kubernetes/pkg/api/v1.PodCondition",
		"v1.GroupVersionForDiscovery":         "k8s.io/apimachinery/pkg/apis/meta/v1.GroupVersionForDiscovery",
		"v1.NamespaceStatus":                  "k8s.io/kubernetes/pkg/api/v1.NamespaceStatus",
		"v1.Job":                              "k8s.io/kubernetes/pkg/apis/batch/v1.Job",
		"v1.PersistentVolumeClaimVolumeSource":        "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaimVolumeSource",
		"v1.Handler":                                  "k8s.io/kubernetes/pkg/api/v1.Handler",
		"v1.ComponentStatusList":                      "k8s.io/kubernetes/pkg/api/v1.ComponentStatusList",
		"v1.ServerAddressByClientCIDR":                "k8s.io/apimachinery/pkg/apis/meta/v1.ServerAddressByClientCIDR",
		"v1.PodAntiAffinity":                          "k8s.io/kubernetes/pkg/api/v1.PodAntiAffinity",
		"v1.ISCSIVolumeSource":                        "k8s.io/kubernetes/pkg/api/v1.ISCSIVolumeSource",
		"v1.ContainerStateRunning":                    "k8s.io/kubernetes/pkg/api/v1.ContainerStateRunning",
		"v1.WeightedPodAffinityTerm":                  "k8s.io/kubernetes/pkg/api/v1.WeightedPodAffinityTerm",
		"v1beta1.HostPortRange":                       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HostPortRange",
		"v1.HorizontalPodAutoscalerSpec":              "k8s.io/kubernetes/pkg/apis/autoscaling/v1.HorizontalPodAutoscalerSpec",
		"v1.HorizontalPodAutoscalerList":              "k8s.io/kubernetes/pkg/apis/autoscaling/v1.HorizontalPodAutoscalerList",
		"v1beta1.RoleRef":                             "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.RoleRef",
		"v1.Probe":                                    "k8s.io/kubernetes/pkg/api/v1.Probe",
		"v1beta1.IngressTLS":                          "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressTLS",
		"v1beta1.ThirdPartyResourceList":              "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ThirdPartyResourceList",
		"v1beta1.DaemonSet":                           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DaemonSet",
		"v1.APIGroup":                                 "k8s.io/apimachinery/pkg/apis/meta/v1.APIGroup",
		"v1beta1.Subject":                             "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.Subject",
		"v1beta1.DeploymentList":                      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentList",
		"v1.NodeAffinity":                             "k8s.io/kubernetes/pkg/api/v1.NodeAffinity",
		"v1beta1.RollingUpdateDeployment":             "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.RollingUpdateDeployment",
		"v1beta1.APIVersion":                          "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.APIVersion",
		"v1alpha1.CertificateSigningRequest":          "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequest",
		"v1.CinderVolumeSource":                       "k8s.io/kubernetes/pkg/api/v1.CinderVolumeSource",
		"v1.NamespaceSpec":                            "k8s.io/kubernetes/pkg/api/v1.NamespaceSpec",
		"v1beta1.PodDisruptionBudgetSpec":             "k8s.io/kubernetes/pkg/apis/policy/v1beta1.PodDisruptionBudgetSpec",
		"v1.Patch":                                    "k8s.io/apimachinery/pkg/apis/meta/v1.Patch",
		"v1beta1.ClusterRoleBinding":                  "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.ClusterRoleBinding",
		"v1beta1.HorizontalPodAutoscalerSpec":         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HorizontalPodAutoscalerSpec",
		"v1.PersistentVolumeClaimSpec":                "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaimSpec",
		"v1.Secret":                                   "k8s.io/kubernetes/pkg/api/v1.Secret",
		"v1.NodeCondition":                            "k8s.io/kubernetes/pkg/api/v1.NodeCondition",
		"v1.LocalObjectReference":                     "k8s.io/kubernetes/pkg/api/v1.LocalObjectReference",
		"runtime.RawExtension":                        "k8s.io/apimachinery/pkg/runtime.RawExtension",
		"v1.PreferredSchedulingTerm":                  "k8s.io/kubernetes/pkg/api/v1.PreferredSchedulingTerm",
		"v1.RBDVolumeSource":                          "k8s.io/kubernetes/pkg/api/v1.RBDVolumeSource",
		"v1.KeyToPath":                                "k8s.io/kubernetes/pkg/api/v1.KeyToPath",
		"v1.ScaleStatus":                              "k8s.io/kubernetes/pkg/apis/autoscaling/v1.ScaleStatus",
		"v1alpha1.PolicyRule":                         "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.PolicyRule",
		"v1.EndpointPort":                             "k8s.io/kubernetes/pkg/api/v1.EndpointPort",
		"v1beta1.IngressList":                         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressList",
		"v1.EndpointAddress":                          "k8s.io/kubernetes/pkg/api/v1.EndpointAddress",
		"v1.NodeSelector":                             "k8s.io/kubernetes/pkg/api/v1.NodeSelector",
		"v1beta1.StorageClassList":                    "k8s.io/kubernetes/pkg/apis/storage/v1beta1.StorageClassList",
		"v1.ServiceList":                              "k8s.io/kubernetes/pkg/api/v1.ServiceList",
		"v2alpha1.CronJobSpec":                        "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.CronJobSpec",
		"v1.ContainerStateTerminated":                 "k8s.io/kubernetes/pkg/api/v1.ContainerStateTerminated",
		"v1beta1.TokenReview":                         "k8s.io/kubernetes/pkg/apis/authentication/v1beta1.TokenReview",
		"v1beta1.IngressBackend":                      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressBackend",
		"v1.Time":                                     "k8s.io/apimachinery/pkg/apis/meta/v1.Time",
		"v1beta1.IngressSpec":                         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressSpec",
		"v2alpha1.JobTemplateSpec":                    "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobTemplateSpec",
		"v1.LimitRange":                               "k8s.io/kubernetes/pkg/api/v1.LimitRange",
		"v1beta1.UserInfo":                            "k8s.io/kubernetes/pkg/apis/authentication/v1beta1.UserInfo",
		"v1.ResourceQuotaSpec":                        "k8s.io/kubernetes/pkg/api/v1.ResourceQuotaSpec",
		"v1.ContainerPort":                            "k8s.io/kubernetes/pkg/api/v1.ContainerPort",
		"v1beta1.HTTPIngressRuleValue":                "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HTTPIngressRuleValue",
		"v1.AzureFileVolumeSource":                    "k8s.io/kubernetes/pkg/api/v1.AzureFileVolumeSource",
		"v1beta1.NetworkPolicySpec":                   "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicySpec",
		"v1.PodTemplateSpec":                          "k8s.io/kubernetes/pkg/api/v1.PodTemplateSpec",
		"v1.SecretVolumeSource":                       "k8s.io/kubernetes/pkg/api/v1.SecretVolumeSource",
		"v1.PodSpec":                                  "k8s.io/kubernetes/pkg/api/v1.PodSpec",
		"v1.CephFSVolumeSource":                       "k8s.io/kubernetes/pkg/api/v1.CephFSVolumeSource",
		"v1beta1.CPUTargetUtilization":                "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.CPUTargetUtilization",
		"v1.Volume":                                   "k8s.io/kubernetes/pkg/api/v1.Volume",
		"v1beta1.Ingress":                             "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.Ingress",
		"v1beta1.HorizontalPodAutoscalerList":         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HorizontalPodAutoscalerList",
		"v1.PersistentVolumeStatus":                   "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeStatus",
		"v1beta1.IDRange":                             "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IDRange",
		"v2alpha1.JobCondition":                       "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobCondition",
		"v1beta1.IngressRule":                         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressRule",
		"v1alpha1.RoleRef":                            "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.RoleRef",
		"v1.PodAffinityTerm":                          "k8s.io/kubernetes/pkg/api/v1.PodAffinityTerm",
		"v1.ObjectReference":                          "k8s.io/kubernetes/pkg/api/v1.ObjectReference",
		"v1.ServiceStatus":                            "k8s.io/kubernetes/pkg/api/v1.ServiceStatus",
		"v1.APIResource":                              "k8s.io/apimachinery/pkg/apis/meta/v1.APIResource",
		"v1beta1.Scale":                               "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.Scale",
		"v1.AzureDiskVolumeSource":                    "k8s.io/kubernetes/pkg/api/v1.AzureDiskVolumeSource",
		"v1beta1.SubjectAccessReviewStatus":           "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SubjectAccessReviewStatus",
		"v1.ConfigMap":                                "k8s.io/kubernetes/pkg/api/v1.ConfigMap",
		"v1.CrossVersionObjectReference":              "k8s.io/kubernetes/pkg/apis/autoscaling/v1.CrossVersionObjectReference",
		"v1.APIVersions":                              "k8s.io/apimachinery/pkg/apis/meta/v1.APIVersions",
		"v1alpha1.ClusterRoleList":                    "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.ClusterRoleList",
		"v1.Node":                                     "k8s.io/kubernetes/pkg/api/v1.Node",
		"resource.Quantity":                           "k8s.io/kubernetes/pkg/api/resource.Quantity",
		"v1.Event":                                    "k8s.io/kubernetes/pkg/api/v1.Event",
		"v1.JobStatus":                                "k8s.io/kubernetes/pkg/apis/batch/v1.JobStatus",
		"v1.PersistentVolumeSpec":                     "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeSpec",
		"v1beta1.SubjectAccessReviewSpec":             "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SubjectAccessReviewSpec",
		"v1.ResourceFieldSelector":                    "k8s.io/kubernetes/pkg/api/v1.ResourceFieldSelector",
		"v1.EndpointSubset":                           "k8s.io/kubernetes/pkg/api/v1.EndpointSubset",
		"v1alpha1.CertificateSigningRequestSpec":      "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequestSpec",
		"v1.HostPathVolumeSource":                     "k8s.io/kubernetes/pkg/api/v1.HostPathVolumeSource",
		"v1.LoadBalancerStatus":                       "k8s.io/kubernetes/pkg/api/v1.LoadBalancerStatus",
		"v1beta1.HTTPIngressPath":                     "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HTTPIngressPath",
		"v1beta1.Role":                                "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.Role",
		"v1beta1.DeploymentStrategy":                  "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentStrategy",
		"v1beta1.RunAsUserStrategyOptions":            "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.RunAsUserStrategyOptions",
		"v1beta1.DeploymentSpec":                      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentSpec",
		"v1.ExecAction":                               "k8s.io/kubernetes/pkg/api/v1.ExecAction",
		"v1beta1.PodSecurityPolicySpec":               "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.PodSecurityPolicySpec",
		"v1.HorizontalPodAutoscalerStatus":            "k8s.io/kubernetes/pkg/apis/autoscaling/v1.HorizontalPodAutoscalerStatus",
		"v1.PersistentVolumeList":                     "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeList",
		"v1alpha1.ClusterRole":                        "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.ClusterRole",
		"v1.JobSpec":                                  "k8s.io/kubernetes/pkg/apis/batch/v1.JobSpec",
		"v1beta1.DaemonSetSpec":                       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DaemonSetSpec",
		"v2alpha1.CronJobList":                        "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.CronJobList",
		"v1.Endpoints":                                "k8s.io/kubernetes/pkg/api/v1.Endpoints",
		"v1.SELinuxOptions":                           "k8s.io/kubernetes/pkg/api/v1.SELinuxOptions",
		"v1beta1.SelfSubjectAccessReviewSpec":         "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SelfSubjectAccessReviewSpec",
		"v1beta1.ScaleStatus":                         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ScaleStatus",
		"v1.NodeSelectorTerm":                         "k8s.io/kubernetes/pkg/api/v1.NodeSelectorTerm",
		"v1alpha1.CertificateSigningRequestStatus":    "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequestStatus",
		"v1.StatusDetails":                            "k8s.io/apimachinery/pkg/apis/meta/v1.StatusDetails",
		"v2alpha1.JobStatus":                          "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobStatus",
		"v1beta1.DeploymentRollback":                  "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentRollback",
		"v1.GlusterfsVolumeSource":                    "k8s.io/kubernetes/pkg/api/v1.GlusterfsVolumeSource",
		"v1.ServiceAccountList":                       "k8s.io/kubernetes/pkg/api/v1.ServiceAccountList",
		"v1.JobList":                                  "k8s.io/kubernetes/pkg/apis/batch/v1.JobList",
		"v1.EventList":                                "k8s.io/kubernetes/pkg/api/v1.EventList",
		"v1.ContainerStateWaiting":                    "k8s.io/kubernetes/pkg/api/v1.ContainerStateWaiting",
		"v1.APIResourceList":                          "k8s.io/apimachinery/pkg/apis/meta/v1.APIResourceList",
		"v1.ContainerStatus":                          "k8s.io/kubernetes/pkg/api/v1.ContainerStatus",
		"v2alpha1.JobList":                            "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobList",
		"v1.ConfigMapKeySelector":                     "k8s.io/kubernetes/pkg/api/v1.ConfigMapKeySelector",
		"v1.PhotonPersistentDiskVolumeSource":         "k8s.io/kubernetes/pkg/api/v1.PhotonPersistentDiskVolumeSource",
		"v1.PodTemplateList":                          "k8s.io/kubernetes/pkg/api/v1.PodTemplateList",
		"v1.PersistentVolumeClaimStatus":              "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaimStatus",
		"v1.ServiceAccount":                           "k8s.io/kubernetes/pkg/api/v1.ServiceAccount",
		"v1alpha1.CertificateSigningRequestList":      "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequestList",
		"v1beta1.SupplementalGroupsStrategyOptions":   "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.SupplementalGroupsStrategyOptions",
		"v1.HTTPHeader":                               "k8s.io/kubernetes/pkg/api/v1.HTTPHeader",
		"version.Info":                                "k8s.io/apimachinery/pkg/version.Info",
		"v1.EventSource":                              "k8s.io/kubernetes/pkg/api/v1.EventSource",
		"v1alpha1.ClusterRoleBindingList":             "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.ClusterRoleBindingList",
		"v1.OwnerReference":                           "k8s.io/apimachinery/pkg/apis/meta/v1.OwnerReference",
		"v1beta1.ClusterRoleBindingList":              "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.ClusterRoleBindingList",
		"v1beta1.ScaleSpec":                           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ScaleSpec",
		"v1.GitRepoVolumeSource":                      "k8s.io/kubernetes/pkg/api/v1.GitRepoVolumeSource",
		"v1beta1.NetworkPolicy":                       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicy",
		"v1.ConfigMapEnvSource":                       "k8s.io/kubernetes/pkg/api/v1.ConfigMapEnvSource",
		"v1.PodTemplate":                              "k8s.io/kubernetes/pkg/api/v1.PodTemplate",
		"v1beta1.DeploymentCondition":                 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentCondition",
		"v1beta1.PodDisruptionBudgetStatus":           "k8s.io/kubernetes/pkg/apis/policy/v1beta1.PodDisruptionBudgetStatus",
		"v1.EnvVar":                                   "k8s.io/kubernetes/pkg/api/v1.EnvVar",
		"v1.LimitRangeSpec":                           "k8s.io/kubernetes/pkg/api/v1.LimitRangeSpec",
		"v1.DownwardAPIVolumeSource":                  "k8s.io/kubernetes/pkg/api/v1.DownwardAPIVolumeSource",
		"v1.NodeDaemonEndpoints":                      "k8s.io/kubernetes/pkg/api/v1.NodeDaemonEndpoints",
		"v1.ComponentCondition":                       "k8s.io/kubernetes/pkg/api/v1.ComponentCondition",
		"v1alpha1.CertificateSigningRequestCondition": "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequestCondition",
		"v1.SecurityContext":                          "k8s.io/kubernetes/pkg/api/v1.SecurityContext",
		"v1beta1.LocalSubjectAccessReview":            "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.LocalSubjectAccessReview",
		"v1beta1.StatefulSetSpec":                     "k8s.io/kubernetes/pkg/apis/apps/v1beta1.StatefulSetSpec",
		"v1.NodeAddress":                              "k8s.io/kubernetes/pkg/api/v1.NodeAddress",
		"v1.QuobyteVolumeSource":                      "k8s.io/kubernetes/pkg/api/v1.QuobyteVolumeSource",
		"v1.Capabilities":                             "k8s.io/kubernetes/pkg/api/v1.Capabilities",
		"v1.GCEPersistentDiskVolumeSource":            "k8s.io/kubernetes/pkg/api/v1.GCEPersistentDiskVolumeSource",
		"v1beta1.ReplicaSet":                          "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSet",
		"v1beta1.HorizontalPodAutoscalerStatus":       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HorizontalPodAutoscalerStatus",
		"v1beta1.PolicyRule":                          "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.PolicyRule",
		"v1.ConfigMapList":                            "k8s.io/kubernetes/pkg/api/v1.ConfigMapList",
		"v1.Lifecycle":                                "k8s.io/kubernetes/pkg/api/v1.Lifecycle",
		"v1beta1.SelfSubjectAccessReview":             "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SelfSubjectAccessReview",
		"v2alpha1.CronJob":                            "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.CronJob",
		"v2alpha1.CronJobStatus":                      "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.CronJobStatus",
		"v1beta1.SubjectAccessReview":                 "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SubjectAccessReview",
		"v1.Preconditions":                            "k8s.io/kubernetes/pkg/api/v1.Preconditions",
		"v1beta1.DaemonSetList":                       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DaemonSetList",
		"v1.PersistentVolumeClaim":                    "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaim",
		"v1.Scale":                                    "k8s.io/kubernetes/pkg/apis/autoscaling/v1.Scale",
		"v1beta1.StatefulSetStatus":                   "k8s.io/kubernetes/pkg/apis/apps/v1beta1.StatefulSetStatus",
		"v1.NFSVolumeSource":                          "k8s.io/kubernetes/pkg/api/v1.NFSVolumeSource",
		"v1.ObjectFieldSelector":                      "k8s.io/kubernetes/pkg/api/v1.ObjectFieldSelector",
		"v1.ResourceRequirements":                     "k8s.io/kubernetes/pkg/api/v1.ResourceRequirements",
		"v1.WatchEvent":                               "k8s.io/apimachinery/pkg/apis/meta/v1.WatchEvent",
		"v1.ReplicationControllerSpec":                "k8s.io/kubernetes/pkg/api/v1.ReplicationControllerSpec",
		"v1.HTTPGetAction":                            "k8s.io/kubernetes/pkg/api/v1.HTTPGetAction",
		"v1beta1.RollbackConfig":                      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.RollbackConfig",
		"v1beta1.TokenReviewSpec":                     "k8s.io/kubernetes/pkg/apis/authentication/v1beta1.TokenReviewSpec",
		"v1.PodSecurityContext":                       "k8s.io/kubernetes/pkg/api/v1.PodSecurityContext",
		"v1beta1.PodDisruptionBudgetList":             "k8s.io/kubernetes/pkg/apis/policy/v1beta1.PodDisruptionBudgetList",
		"v1.VolumeMount":                              "k8s.io/kubernetes/pkg/api/v1.VolumeMount",
		"v1.ReplicationController":                    "k8s.io/kubernetes/pkg/api/v1.ReplicationController",
		"v1.NamespaceList":                            "k8s.io/kubernetes/pkg/api/v1.NamespaceList",
		"v1alpha1.ClusterRoleBinding":                 "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.ClusterRoleBinding",
		"v1.TCPSocketAction":                          "k8s.io/kubernetes/pkg/api/v1.TCPSocketAction",
		"v1.Binding":                                  "k8s.io/kubernetes/pkg/api/v1.Binding",
		"v1beta1.ReplicaSetStatus":                    "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSetStatus",
		"intstr.IntOrString":                          "k8s.io/kubernetes/pkg/util/intstr.IntOrString",
		"v1.EndpointsList":                            "k8s.io/kubernetes/pkg/api/v1.EndpointsList",
		"v1.Container":                                "k8s.io/kubernetes/pkg/api/v1.Container",
		"v1alpha1.RoleList":                           "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.RoleList",
		"v1.VsphereVirtualDiskVolumeSource":           "k8s.io/kubernetes/pkg/api/v1.VsphereVirtualDiskVolumeSource",
		"v1.NodeList":                                 "k8s.io/kubernetes/pkg/api/v1.NodeList",
		"v1.EmptyDirVolumeSource":                     "k8s.io/kubernetes/pkg/api/v1.EmptyDirVolumeSource",
		"v1beta1.FSGroupStrategyOptions":              "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.FSGroupStrategyOptions",
		"v1.Status":                                   "k8s.io/apimachinery/pkg/apis/meta/v1.Status",
	}

	for k, v := range compatibilityMap {
		if _, found := s.Definitions[v]; !found {
			continue
		}
		s.Definitions[k] = spec.Schema{
			SchemaProps: spec.SchemaProps{
				Ref:         spec.MustCreateRef("#/definitions/" + openapi.EscapeJsonPointer(v)),
				Description: fmt.Sprintf("Deprecated. Please use %s instead.", v),
			},
		}
	}
	return s, nil
}
