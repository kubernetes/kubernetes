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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/parsers"
	"k8s.io/utils/pointer"
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
	if obj.Spec.Replicas == nil {
		obj.Spec.Replicas = new(int32)
		*obj.Spec.Replicas = 1
	}
}
func SetDefaults_Volume(obj *v1.Volume) {
	if pointer.AllPtrFieldsNil(&obj.VolumeSource) {
		obj.VolumeSource = v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
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
			obj.Spec.AllocateLoadBalancerNodePorts = pointer.Bool(true)
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
		if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) &&
			obj.Spec.Containers[i].Resources.Requests != nil {
			// For normal containers, set resize restart policy to default value (NotRequired), if not specified.
			resizePolicySpecified := make(map[v1.ResourceName]bool)
			for _, p := range obj.Spec.Containers[i].ResizePolicy {
				resizePolicySpecified[p.ResourceName] = true
			}
			setDefaultResizePolicy := func(resourceName v1.ResourceName) {
				if _, found := resizePolicySpecified[resourceName]; !found {
					obj.Spec.Containers[i].ResizePolicy = append(obj.Spec.Containers[i].ResizePolicy,
						v1.ContainerResizePolicy{
							ResourceName:  resourceName,
							RestartPolicy: v1.NotRequired,
						})
				}
			}
			if _, exists := obj.Spec.Containers[i].Resources.Requests[v1.ResourceCPU]; exists {
				setDefaultResizePolicy(v1.ResourceCPU)
			}
			if _, exists := obj.Spec.Containers[i].Resources.Requests[v1.ResourceMemory]; exists {
				setDefaultResizePolicy(v1.ResourceMemory)
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
func SetDefaults_ISCSIVolumeSource(obj *v1.ISCSIVolumeSource) {
	if obj.ISCSIInterface == "" {
		obj.ISCSIInterface = "default"
	}
}
func SetDefaults_ISCSIPersistentVolumeSource(obj *v1.ISCSIPersistentVolumeSource) {
	if obj.ISCSIInterface == "" {
		obj.ISCSIInterface = "default"
	}
}
func SetDefaults_AzureDiskVolumeSource(obj *v1.AzureDiskVolumeSource) {
	if obj.CachingMode == nil {
		obj.CachingMode = new(v1.AzureDataDiskCachingMode)
		*obj.CachingMode = v1.AzureDataDiskCachingReadWrite
	}
	if obj.Kind == nil {
		obj.Kind = new(v1.AzureDataDiskKind)
		*obj.Kind = v1.AzureSharedBlobDisk
	}
	if obj.FSType == nil {
		obj.FSType = new(string)
		*obj.FSType = "ext4"
	}
	if obj.ReadOnly == nil {
		obj.ReadOnly = new(bool)
		*obj.ReadOnly = false
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

func SetDefaults_ScaleIOVolumeSource(obj *v1.ScaleIOVolumeSource) {
	if obj.StorageMode == "" {
		obj.StorageMode = "ThinProvisioned"
	}
	if obj.FSType == "" {
		obj.FSType = "xfs"
	}
}

func SetDefaults_ScaleIOPersistentVolumeSource(obj *v1.ScaleIOPersistentVolumeSource) {
	if obj.StorageMode == "" {
		obj.StorageMode = "ThinProvisioned"
	}
	if obj.FSType == "" {
		obj.FSType = "xfs"
	}
}

func SetDefaults_HostPathVolumeSource(obj *v1.HostPathVolumeSource) {
	typeVol := v1.HostPathUnset
	if obj.Type == nil {
		obj.Type = &typeVol
	}
}
