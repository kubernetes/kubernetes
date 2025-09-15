/*
Copyright 2015 The Kubernetes Authors.

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

package v1

import (
	"time"

	"k8s.io/utils/ptr"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/api/v1/service"
	corev1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/parsers"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_ResourceList(obj *v1.ResourceList) {
	for key, val := range *obj {
		// TODO(#18538): We round up resource values to milli scale to maintain API compatibility.
		// In the future, we should instead reject values that need rounding.
		const milliScale = -3
		val.RoundUp(milliScale)

		(*obj)[v1.ResourceName(key)] = val
	}
}

func SetDefaults_ReplicationController(obj *v1.ReplicationController) {
	var labels map[string]string
	if obj.Spec.Template != nil {
		labels = obj.Spec.Template.Labels
	}
	// TODO: support templates defined elsewhere when we support them in the API
	if labels != nil {
		if len(obj.Spec.Selector) == 0 {
			obj.Spec.Selector = labels
		}
		if len(obj.Labels) == 0 {
			obj.Labels = labels
		}
	}
	// obj.Spec.Replicas is defaulted declaratively
}
func SetDefaults_Volume(obj *v1.Volume) {
	if ptr.AllPtrFieldsNil(&obj.VolumeSource) {
		obj.VolumeSource = v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.ImageVolume) && obj.Image != nil && obj.Image.PullPolicy == "" {
		// PullPolicy defaults to Always if :latest tag is specified, or IfNotPresent otherwise.
		_, tag, _, _ := parsers.ParseImageName(obj.Image.Reference)
		if tag == "latest" {
			obj.Image.PullPolicy = v1.PullAlways
		} else {
			obj.Image.PullPolicy = v1.PullIfNotPresent
		}
	}
}
func SetDefaults_Container(obj *v1.Container) {
	if obj.ImagePullPolicy == "" {
		// Ignore error and assume it has been validated elsewhere
		_, tag, _, _ := parsers.ParseImageName(obj.Image)

		// Check image tag
		if tag == "latest" {
			obj.ImagePullPolicy = v1.PullAlways
		} else {
			obj.ImagePullPolicy = v1.PullIfNotPresent
		}
	}
	if obj.TerminationMessagePath == "" {
		obj.TerminationMessagePath = v1.TerminationMessagePathDefault
	}
	if obj.TerminationMessagePolicy == "" {
		obj.TerminationMessagePolicy = v1.TerminationMessageReadFile
	}
}

func SetDefaults_EphemeralContainer(obj *v1.EphemeralContainer) {
	SetDefaults_Container((*v1.Container)(&obj.EphemeralContainerCommon))
}

func SetDefaults_Service(obj *v1.Service) {
	if obj.Spec.SessionAffinity == "" {
		obj.Spec.SessionAffinity = v1.ServiceAffinityNone
	}
	if obj.Spec.SessionAffinity == v1.ServiceAffinityNone {
		obj.Spec.SessionAffinityConfig = nil
	}
	if obj.Spec.SessionAffinity == v1.ServiceAffinityClientIP {
		if obj.Spec.SessionAffinityConfig == nil || obj.Spec.SessionAffinityConfig.ClientIP == nil || obj.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds == nil {
			timeoutSeconds := v1.DefaultClientIPServiceAffinitySeconds
			obj.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: &timeoutSeconds,
				},
			}
		}
	}
	if obj.Spec.Type == "" {
		obj.Spec.Type = v1.ServiceTypeClusterIP
	}
	for i := range obj.Spec.Ports {
		sp := &obj.Spec.Ports[i]
		if sp.Protocol == "" {
			sp.Protocol = v1.ProtocolTCP
		}
		if sp.TargetPort == intstr.FromInt32(0) || sp.TargetPort == intstr.FromString("") {
			sp.TargetPort = intstr.FromInt32(sp.Port)
		}
	}
	// Defaults ExternalTrafficPolicy field for externally-accessible service
	// to Global for consistency.
	if service.ExternallyAccessible(obj) && obj.Spec.ExternalTrafficPolicy == "" {
		obj.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyCluster
	}

	if obj.Spec.InternalTrafficPolicy == nil {
		if obj.Spec.Type == v1.ServiceTypeNodePort || obj.Spec.Type == v1.ServiceTypeLoadBalancer || obj.Spec.Type == v1.ServiceTypeClusterIP {
			serviceInternalTrafficPolicyCluster := v1.ServiceInternalTrafficPolicyCluster
			obj.Spec.InternalTrafficPolicy = &serviceInternalTrafficPolicyCluster
		}
	}

	if obj.Spec.Type == v1.ServiceTypeLoadBalancer {
		if obj.Spec.AllocateLoadBalancerNodePorts == nil {
			obj.Spec.AllocateLoadBalancerNodePorts = ptr.To(true)
		}
	}

	if obj.Spec.Type == v1.ServiceTypeLoadBalancer {
		if utilfeature.DefaultFeatureGate.Enabled(features.LoadBalancerIPMode) {
			ipMode := v1.LoadBalancerIPModeVIP

			for i, ing := range obj.Status.LoadBalancer.Ingress {
				if ing.IP != "" && ing.IPMode == nil {
					obj.Status.LoadBalancer.Ingress[i].IPMode = &ipMode
				}
			}
		}
	}

}
func SetDefaults_Pod(obj *v1.Pod) {
	// If limits are specified, but requests are not, default requests to limits
	// This is done here rather than a more specific defaulting pass on v1.ResourceRequirements
	// because we only want this defaulting semantic to take place on a v1.Pod and not a v1.PodTemplate
	for i := range obj.Spec.Containers {
		// set requests to limits if requests are not specified, but limits are
		if obj.Spec.Containers[i].Resources.Limits != nil {
			if obj.Spec.Containers[i].Resources.Requests == nil {
				obj.Spec.Containers[i].Resources.Requests = make(v1.ResourceList)
			}
			for key, value := range obj.Spec.Containers[i].Resources.Limits {
				if _, exists := obj.Spec.Containers[i].Resources.Requests[key]; !exists {
					obj.Spec.Containers[i].Resources.Requests[key] = value.DeepCopy()
				}
			}
		}
	}
	for i := range obj.Spec.InitContainers {
		if obj.Spec.InitContainers[i].Resources.Limits != nil {
			if obj.Spec.InitContainers[i].Resources.Requests == nil {
				obj.Spec.InitContainers[i].Resources.Requests = make(v1.ResourceList)
			}
			for key, value := range obj.Spec.InitContainers[i].Resources.Limits {
				if _, exists := obj.Spec.InitContainers[i].Resources.Requests[key]; !exists {
					obj.Spec.InitContainers[i].Resources.Requests[key] = value.DeepCopy()
				}
			}
		}
	}

	// Pod Requests default values must be applied after container-level default values
	// have been populated.
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources) {
		defaultHugePagePodLimits(obj)
		defaultPodRequests(obj)
	}

	if obj.Spec.EnableServiceLinks == nil {
		enableServiceLinks := v1.DefaultEnableServiceLinks
		obj.Spec.EnableServiceLinks = &enableServiceLinks
	}

	if obj.Spec.HostNetwork {
		defaultHostNetworkPorts(&obj.Spec.Containers)
		defaultHostNetworkPorts(&obj.Spec.InitContainers)
	}
}
func SetDefaults_PodSpec(obj *v1.PodSpec) {
	// New fields added here will break upgrade tests:
	// https://github.com/kubernetes/kubernetes/issues/69445
	// In most cases the new defaulted field can added to SetDefaults_Pod instead of here, so
	// that it only materializes in the Pod object and not all objects with a PodSpec field.
	if obj.DNSPolicy == "" {
		obj.DNSPolicy = v1.DNSClusterFirst
	}
	if obj.RestartPolicy == "" {
		obj.RestartPolicy = v1.RestartPolicyAlways
	}
	if obj.SecurityContext == nil {
		obj.SecurityContext = &v1.PodSecurityContext{}
	}
	if obj.TerminationGracePeriodSeconds == nil {
		period := int64(v1.DefaultTerminationGracePeriodSeconds)
		obj.TerminationGracePeriodSeconds = &period
	}
	if obj.SchedulerName == "" {
		obj.SchedulerName = v1.DefaultSchedulerName
	}
}
func SetDefaults_Probe(obj *v1.Probe) {
	if obj.TimeoutSeconds == 0 {
		obj.TimeoutSeconds = 1
	}
	if obj.PeriodSeconds == 0 {
		obj.PeriodSeconds = 10
	}
	if obj.SuccessThreshold == 0 {
		obj.SuccessThreshold = 1
	}
	if obj.FailureThreshold == 0 {
		obj.FailureThreshold = 3
	}
}
func SetDefaults_SecretVolumeSource(obj *v1.SecretVolumeSource) {
	if obj.DefaultMode == nil {
		perm := int32(v1.SecretVolumeSourceDefaultMode)
		obj.DefaultMode = &perm
	}
}
func SetDefaults_ConfigMapVolumeSource(obj *v1.ConfigMapVolumeSource) {
	if obj.DefaultMode == nil {
		perm := int32(v1.ConfigMapVolumeSourceDefaultMode)
		obj.DefaultMode = &perm
	}
}
func SetDefaults_DownwardAPIVolumeSource(obj *v1.DownwardAPIVolumeSource) {
	if obj.DefaultMode == nil {
		perm := int32(v1.DownwardAPIVolumeSourceDefaultMode)
		obj.DefaultMode = &perm
	}
}
func SetDefaults_Secret(obj *v1.Secret) {
	if obj.Type == "" {
		obj.Type = v1.SecretTypeOpaque
	}
}
func SetDefaults_ProjectedVolumeSource(obj *v1.ProjectedVolumeSource) {
	if obj.DefaultMode == nil {
		perm := int32(v1.ProjectedVolumeSourceDefaultMode)
		obj.DefaultMode = &perm
	}
}
func SetDefaults_ServiceAccountTokenProjection(obj *v1.ServiceAccountTokenProjection) {
	hour := int64(time.Hour.Seconds())
	if obj.ExpirationSeconds == nil {
		obj.ExpirationSeconds = &hour
	}
}
func SetDefaults_PersistentVolume(obj *v1.PersistentVolume) {
	if obj.Status.Phase == "" {
		obj.Status.Phase = v1.VolumePending
	}
	if obj.Spec.PersistentVolumeReclaimPolicy == "" {
		obj.Spec.PersistentVolumeReclaimPolicy = v1.PersistentVolumeReclaimRetain
	}
	if obj.Spec.VolumeMode == nil {
		obj.Spec.VolumeMode = new(v1.PersistentVolumeMode)
		*obj.Spec.VolumeMode = v1.PersistentVolumeFilesystem
	}
}
func SetDefaults_PersistentVolumeClaim(obj *v1.PersistentVolumeClaim) {
	if obj.Status.Phase == "" {
		obj.Status.Phase = v1.ClaimPending
	}
}
func SetDefaults_PersistentVolumeClaimSpec(obj *v1.PersistentVolumeClaimSpec) {
	if obj.VolumeMode == nil {
		obj.VolumeMode = new(v1.PersistentVolumeMode)
		*obj.VolumeMode = v1.PersistentVolumeFilesystem
	}
}
func SetDefaults_Endpoints(obj *v1.Endpoints) {
	for i := range obj.Subsets {
		ss := &obj.Subsets[i]
		for i := range ss.Ports {
			ep := &ss.Ports[i]
			if ep.Protocol == "" {
				ep.Protocol = v1.ProtocolTCP
			}
		}
	}
}
func SetDefaults_HTTPGetAction(obj *v1.HTTPGetAction) {
	if obj.Path == "" {
		obj.Path = "/"
	}
	if obj.Scheme == "" {
		obj.Scheme = v1.URISchemeHTTP
	}
}

// SetDefaults_Namespace adds a default label for all namespaces
func SetDefaults_Namespace(obj *v1.Namespace) {
	// we can't SetDefaults for nameless namespaces (generateName).
	// This code needs to be kept in sync with the implementation that exists
	// in Namespace Canonicalize strategy (pkg/registry/core/namespace)

	// note that this can result in many calls to feature enablement in some cases, but
	// we assume that there's no real cost there.
	if len(obj.Name) > 0 {
		if obj.Labels == nil {
			obj.Labels = map[string]string{}
		}
		obj.Labels[v1.LabelMetadataName] = obj.Name
	}
}

func SetDefaults_NamespaceStatus(obj *v1.NamespaceStatus) {
	if obj.Phase == "" {
		obj.Phase = v1.NamespaceActive
	}
}
func SetDefaults_NodeStatus(obj *v1.NodeStatus) {
	if obj.Allocatable == nil && obj.Capacity != nil {
		obj.Allocatable = make(v1.ResourceList, len(obj.Capacity))
		for key, value := range obj.Capacity {
			obj.Allocatable[key] = value.DeepCopy()
		}
		obj.Allocatable = obj.Capacity
	}
}
func SetDefaults_ObjectFieldSelector(obj *v1.ObjectFieldSelector) {
	if obj.APIVersion == "" {
		obj.APIVersion = "v1"
	}
}
func SetDefaults_LimitRangeItem(obj *v1.LimitRangeItem) {
	// for container limits, we apply default values
	if obj.Type == v1.LimitTypeContainer {

		if obj.Default == nil {
			obj.Default = make(v1.ResourceList)
		}
		if obj.DefaultRequest == nil {
			obj.DefaultRequest = make(v1.ResourceList)
		}

		// If a default limit is unspecified, but the max is specified, default the limit to the max
		for key, value := range obj.Max {
			if _, exists := obj.Default[key]; !exists {
				obj.Default[key] = value.DeepCopy()
			}
		}
		// If a default limit is specified, but the default request is not, default request to limit
		for key, value := range obj.Default {
			if _, exists := obj.DefaultRequest[key]; !exists {
				obj.DefaultRequest[key] = value.DeepCopy()
			}
		}
		// If a default request is not specified, but the min is provided, default request to the min
		for key, value := range obj.Min {
			if _, exists := obj.DefaultRequest[key]; !exists {
				obj.DefaultRequest[key] = value.DeepCopy()
			}
		}
	}
}
func SetDefaults_ConfigMap(obj *v1.ConfigMap) {
	if obj.Data == nil {
		obj.Data = make(map[string]string)
	}
}

// With host networking default all container ports to host ports.
func defaultHostNetworkPorts(containers *[]v1.Container) {
	for i := range *containers {
		for j := range (*containers)[i].Ports {
			if (*containers)[i].Ports[j].HostPort == 0 {
				(*containers)[i].Ports[j].HostPort = (*containers)[i].Ports[j].ContainerPort
			}
		}
	}
}

func SetDefaults_HostPathVolumeSource(obj *v1.HostPathVolumeSource) {
	typeVol := v1.HostPathUnset
	if obj.Type == nil {
		obj.Type = &typeVol
	}
}

func SetDefaults_PodLogOptions(obj *v1.PodLogOptions) {
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLogsQuerySplitStreams) {
		if obj.Stream == nil {
			obj.Stream = ptr.To(v1.LogStreamAll)
		}
	}
}

// defaultPodRequests applies default values for pod-level requests, only when
// pod-level limits are set, in following scenarios:
// 1. When at least one container (regular, init or sidecar) has requests set:
// The pod-level requests become equal to the effective requests of all containers
// in the pod.
// 2. When no containers have requests set: The pod-level requests become equal to
// pod-level limits.
// This defaulting behavior ensures consistent resource accounting at the pod-level
// while maintaining compatibility with the container-level specifications, as detailed
// in KEP-2837: https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/2837-pod-level-resource-spec/README.md#proposed-validation--defaulting-rules
func defaultPodRequests(obj *v1.Pod) {
	// We only populate defaults when the pod-level resources are partly specified already.
	if obj.Spec.Resources == nil {
		return
	}

	if len(obj.Spec.Resources.Limits) == 0 {
		return
	}

	var podReqs v1.ResourceList
	podReqs = obj.Spec.Resources.Requests
	if podReqs == nil {
		podReqs = make(v1.ResourceList)
	}

	aggrCtrReqs := resourcehelper.AggregateContainerRequests(obj, resourcehelper.PodResourcesOptions{})

	// When containers specify requests for a resource (supported by
	// PodLevelResources feature) and pod-level requests are not set, the pod-level requests
	// default to the effective requests of all the containers for that resource.
	for key, aggrCtrLim := range aggrCtrReqs {
		// Defaulting for pod level hugepages requests takes them directly from the pod limit,
		// hugepages cannot be overcommited and must have the limit, so we skip them here.
		if _, exists := podReqs[key]; !exists && resourcehelper.IsSupportedPodLevelResource(key) && !corev1helper.IsHugePageResourceName(key) {
			podReqs[key] = aggrCtrLim.DeepCopy()
		}
	}

	// When no containers specify requests for a resource, the pod-level requests
	// will default to match the pod-level limits, if pod-level
	// limits exist for that resource.
	// Defaulting for pod level hugepages requests is dependent on defaultHugePagePodLimits,
	// if defaultHugePagePodLimits defined the limit, the request will be set here.
	for key, podLim := range obj.Spec.Resources.Limits {
		if _, exists := podReqs[key]; !exists && resourcehelper.IsSupportedPodLevelResource(key) {
			podReqs[key] = podLim.DeepCopy()
		}
	}

	// Only set pod-level resource requests in the PodSpec if the requirements map
	// contains entries after collecting container-level requests and pod-level limits.
	if len(podReqs) > 0 {
		obj.Spec.Resources.Requests = podReqs
	}
}

// defaultHugePagePodLimits applies default values for pod-level limits, only when
// container hugepage limits are set, but not at pod level, in following
// scenario:
// 1. When at least one container (regular, init or sidecar) has hugepage
// limits set:
// The pod-level limit becomes equal to the aggregated hugepages limit of all
// the containers in the pod.
func defaultHugePagePodLimits(obj *v1.Pod) {
	// We only populate defaults when the pod-level resources are partly specified already.
	if obj.Spec.Resources == nil {
		return
	}

	if len(obj.Spec.Resources.Limits) == 0 && len(obj.Spec.Resources.Requests) == 0 {
		return
	}

	var podLims v1.ResourceList
	podLims = obj.Spec.Resources.Limits
	if podLims == nil {
		podLims = make(v1.ResourceList)
	}

	aggrCtrLims := resourcehelper.AggregateContainerLimits(obj, resourcehelper.PodResourcesOptions{})

	// When containers specify limits for hugepages and pod-level limits are not
	// set for that resource, the pod-level limit will default to the aggregated
	// hugepages limit of all the containers.
	for key, aggrCtrLim := range aggrCtrLims {
		if _, exists := podLims[key]; !exists && resourcehelper.IsSupportedPodLevelResource(key) && corev1helper.IsHugePageResourceName(key) {
			podLims[key] = aggrCtrLim.DeepCopy()
		}
	}

	// Only set pod-level resource limits in the PodSpec if the requirements map
	// contains entries after collecting container-level limits and pod-level limits for hugepages.
	if len(podLims) > 0 {
		obj.Spec.Resources.Limits = podLims
	}
}
