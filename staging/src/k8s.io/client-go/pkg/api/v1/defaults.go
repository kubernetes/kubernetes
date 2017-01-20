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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/util"
	"k8s.io/client-go/pkg/util/intstr"
	"k8s.io/client-go/pkg/util/parsers"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	RegisterDefaults(scheme)
	return scheme.AddDefaultingFuncs(
		SetDefaults_PodExecOptions,
		SetDefaults_PodAttachOptions,
		SetDefaults_ReplicationController,
		SetDefaults_Volume,
		SetDefaults_ContainerPort,
		SetDefaults_Container,
		SetDefaults_ServiceSpec,
		SetDefaults_Pod,
		SetDefaults_PodSpec,
		SetDefaults_Probe,
		SetDefaults_SecretVolumeSource,
		SetDefaults_ConfigMapVolumeSource,
		SetDefaults_DownwardAPIVolumeSource,
		SetDefaults_Secret,
		SetDefaults_PersistentVolume,
		SetDefaults_PersistentVolumeClaim,
		SetDefaults_ISCSIVolumeSource,
		SetDefaults_Endpoints,
		SetDefaults_HTTPGetAction,
		SetDefaults_NamespaceStatus,
		SetDefaults_Node,
		SetDefaults_NodeStatus,
		SetDefaults_ObjectFieldSelector,
		SetDefaults_LimitRangeItem,
		SetDefaults_ConfigMap,
		SetDefaults_RBDVolumeSource,
		SetDefaults_ResourceList,
	)
}

func SetDefaults_ResourceList(obj *ResourceList) {
	for key, val := range *obj {
		// TODO(#18538): We round up resource values to milli scale to maintain API compatibility.
		// In the future, we should instead reject values that need rounding.
		const milliScale = -3
		val.RoundUp(milliScale)

		(*obj)[ResourceName(key)] = val
	}
}

func SetDefaults_PodExecOptions(obj *PodExecOptions) {
	obj.Stdout = true
	obj.Stderr = true
}
func SetDefaults_PodAttachOptions(obj *PodAttachOptions) {
	obj.Stdout = true
	obj.Stderr = true
}
func SetDefaults_ReplicationController(obj *ReplicationController) {
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
func SetDefaults_Volume(obj *Volume) {
	if util.AllPtrFieldsNil(&obj.VolumeSource) {
		obj.VolumeSource = VolumeSource{
			EmptyDir: &EmptyDirVolumeSource{},
		}
	}
}
func SetDefaults_ContainerPort(obj *ContainerPort) {
	if obj.Protocol == "" {
		obj.Protocol = ProtocolTCP
	}
}
func SetDefaults_Container(obj *Container) {
	if obj.ImagePullPolicy == "" {
		// Ignore error and assume it has been validated elsewhere
		_, tag, _, _ := parsers.ParseImageName(obj.Image)

		// Check image tag
		if tag == "latest" {
			obj.ImagePullPolicy = PullAlways
		} else {
			obj.ImagePullPolicy = PullIfNotPresent
		}
	}
	if obj.TerminationMessagePath == "" {
		obj.TerminationMessagePath = TerminationMessagePathDefault
	}
}
func SetDefaults_ServiceSpec(obj *ServiceSpec) {
	if obj.SessionAffinity == "" {
		obj.SessionAffinity = ServiceAffinityNone
	}
	if obj.Type == "" {
		obj.Type = ServiceTypeClusterIP
	}
	for i := range obj.Ports {
		sp := &obj.Ports[i]
		if sp.Protocol == "" {
			sp.Protocol = ProtocolTCP
		}
		if sp.TargetPort == intstr.FromInt(0) || sp.TargetPort == intstr.FromString("") {
			sp.TargetPort = intstr.FromInt(int(sp.Port))
		}
	}
}
func SetDefaults_Pod(obj *Pod) {
	// If limits are specified, but requests are not, default requests to limits
	// This is done here rather than a more specific defaulting pass on ResourceRequirements
	// because we only want this defaulting semantic to take place on a Pod and not a PodTemplate
	for i := range obj.Spec.Containers {
		// set requests to limits if requests are not specified, but limits are
		if obj.Spec.Containers[i].Resources.Limits != nil {
			if obj.Spec.Containers[i].Resources.Requests == nil {
				obj.Spec.Containers[i].Resources.Requests = make(ResourceList)
			}
			for key, value := range obj.Spec.Containers[i].Resources.Limits {
				if _, exists := obj.Spec.Containers[i].Resources.Requests[key]; !exists {
					obj.Spec.Containers[i].Resources.Requests[key] = *(value.Copy())
				}
			}
		}
	}
}
func SetDefaults_PodSpec(obj *PodSpec) {
	if obj.DNSPolicy == "" {
		obj.DNSPolicy = DNSClusterFirst
	}
	if obj.RestartPolicy == "" {
		obj.RestartPolicy = RestartPolicyAlways
	}
	if obj.HostNetwork {
		defaultHostNetworkPorts(&obj.Containers)
	}
	if obj.SecurityContext == nil {
		obj.SecurityContext = &PodSecurityContext{}
	}
	if obj.TerminationGracePeriodSeconds == nil {
		period := int64(DefaultTerminationGracePeriodSeconds)
		obj.TerminationGracePeriodSeconds = &period
	}
}
func SetDefaults_Probe(obj *Probe) {
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
func SetDefaults_SecretVolumeSource(obj *SecretVolumeSource) {
	if obj.DefaultMode == nil {
		perm := int32(SecretVolumeSourceDefaultMode)
		obj.DefaultMode = &perm
	}
}
func SetDefaults_ConfigMapVolumeSource(obj *ConfigMapVolumeSource) {
	if obj.DefaultMode == nil {
		perm := int32(ConfigMapVolumeSourceDefaultMode)
		obj.DefaultMode = &perm
	}
}
func SetDefaults_DownwardAPIVolumeSource(obj *DownwardAPIVolumeSource) {
	if obj.DefaultMode == nil {
		perm := int32(DownwardAPIVolumeSourceDefaultMode)
		obj.DefaultMode = &perm
	}
}
func SetDefaults_Secret(obj *Secret) {
	if obj.Type == "" {
		obj.Type = SecretTypeOpaque
	}
}
func SetDefaults_PersistentVolume(obj *PersistentVolume) {
	if obj.Status.Phase == "" {
		obj.Status.Phase = VolumePending
	}
	if obj.Spec.PersistentVolumeReclaimPolicy == "" {
		obj.Spec.PersistentVolumeReclaimPolicy = PersistentVolumeReclaimRetain
	}
}
func SetDefaults_PersistentVolumeClaim(obj *PersistentVolumeClaim) {
	if obj.Status.Phase == "" {
		obj.Status.Phase = ClaimPending
	}
}
func SetDefaults_ISCSIVolumeSource(obj *ISCSIVolumeSource) {
	if obj.ISCSIInterface == "" {
		obj.ISCSIInterface = "default"
	}
}
func SetDefaults_AzureDiskVolumeSource(obj *AzureDiskVolumeSource) {
	if obj.CachingMode == nil {
		obj.CachingMode = new(AzureDataDiskCachingMode)
		*obj.CachingMode = AzureDataDiskCachingNone
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
func SetDefaults_Endpoints(obj *Endpoints) {
	for i := range obj.Subsets {
		ss := &obj.Subsets[i]
		for i := range ss.Ports {
			ep := &ss.Ports[i]
			if ep.Protocol == "" {
				ep.Protocol = ProtocolTCP
			}
		}
	}
}
func SetDefaults_HTTPGetAction(obj *HTTPGetAction) {
	if obj.Path == "" {
		obj.Path = "/"
	}
	if obj.Scheme == "" {
		obj.Scheme = URISchemeHTTP
	}
}
func SetDefaults_NamespaceStatus(obj *NamespaceStatus) {
	if obj.Phase == "" {
		obj.Phase = NamespaceActive
	}
}
func SetDefaults_Node(obj *Node) {
	if obj.Spec.ExternalID == "" {
		obj.Spec.ExternalID = obj.Name
	}
}
func SetDefaults_NodeStatus(obj *NodeStatus) {
	if obj.Allocatable == nil && obj.Capacity != nil {
		obj.Allocatable = make(ResourceList, len(obj.Capacity))
		for key, value := range obj.Capacity {
			obj.Allocatable[key] = *(value.Copy())
		}
		obj.Allocatable = obj.Capacity
	}
}
func SetDefaults_ObjectFieldSelector(obj *ObjectFieldSelector) {
	if obj.APIVersion == "" {
		obj.APIVersion = "v1"
	}
}
func SetDefaults_LimitRangeItem(obj *LimitRangeItem) {
	// for container limits, we apply default values
	if obj.Type == LimitTypeContainer {

		if obj.Default == nil {
			obj.Default = make(ResourceList)
		}
		if obj.DefaultRequest == nil {
			obj.DefaultRequest = make(ResourceList)
		}

		// If a default limit is unspecified, but the max is specified, default the limit to the max
		for key, value := range obj.Max {
			if _, exists := obj.Default[key]; !exists {
				obj.Default[key] = *(value.Copy())
			}
		}
		// If a default limit is specified, but the default request is not, default request to limit
		for key, value := range obj.Default {
			if _, exists := obj.DefaultRequest[key]; !exists {
				obj.DefaultRequest[key] = *(value.Copy())
			}
		}
		// If a default request is not specified, but the min is provided, default request to the min
		for key, value := range obj.Min {
			if _, exists := obj.DefaultRequest[key]; !exists {
				obj.DefaultRequest[key] = *(value.Copy())
			}
		}
	}
}
func SetDefaults_ConfigMap(obj *ConfigMap) {
	if obj.Data == nil {
		obj.Data = make(map[string]string)
	}
}

// With host networking default all container ports to host ports.
func defaultHostNetworkPorts(containers *[]Container) {
	for i := range *containers {
		for j := range (*containers)[i].Ports {
			if (*containers)[i].Ports[j].HostPort == 0 {
				(*containers)[i].Ports[j].HostPort = (*containers)[i].Ports[j].ContainerPort
			}
		}
	}
}

func SetDefaults_RBDVolumeSource(obj *RBDVolumeSource) {
	if obj.RBDPool == "" {
		obj.RBDPool = "rbd"
	}
	if obj.RadosUser == "" {
		obj.RadosUser = "admin"
	}
	if obj.Keyring == "" {
		obj.Keyring = "/etc/ceph/keyring"
	}
}
