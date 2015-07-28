/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// AUTO-GENERATED FUNCTIONS START HERE
import (
	api "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	resource "github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	conversion "github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	runtime "github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	util "github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	inf "speter.net/go/exp/math/dec/inf"
	time "time"
)

func deepCopy_resource_Quantity(in resource.Quantity, out *resource.Quantity, c *conversion.Cloner) error {
	if in.Amount != nil {
		if newVal, err := c.DeepCopy(in.Amount); err != nil {
			return err
		} else {
			out.Amount = newVal.(*inf.Dec)
		}
	} else {
		out.Amount = nil
	}
	out.Format = in.Format
	return nil
}

func deepCopy_v1_AWSElasticBlockStoreVolumeSource(in AWSElasticBlockStoreVolumeSource, out *AWSElasticBlockStoreVolumeSource, c *conversion.Cloner) error {
	out.VolumeID = in.VolumeID
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func deepCopy_v1_Binding(in Binding, out *Binding, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectReference(in.Target, &out.Target, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_Capabilities(in Capabilities, out *Capabilities, c *conversion.Cloner) error {
	if in.Add != nil {
		out.Add = make([]Capability, len(in.Add))
		for i := range in.Add {
			out.Add[i] = in.Add[i]
		}
	} else {
		out.Add = nil
	}
	if in.Drop != nil {
		out.Drop = make([]Capability, len(in.Drop))
		for i := range in.Drop {
			out.Drop[i] = in.Drop[i]
		}
	} else {
		out.Drop = nil
	}
	return nil
}

func deepCopy_v1_ComponentCondition(in ComponentCondition, out *ComponentCondition, c *conversion.Cloner) error {
	out.Type = in.Type
	out.Status = in.Status
	out.Message = in.Message
	out.Error = in.Error
	return nil
}

func deepCopy_v1_ComponentStatus(in ComponentStatus, out *ComponentStatus, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if in.Conditions != nil {
		out.Conditions = make([]ComponentCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := deepCopy_v1_ComponentCondition(in.Conditions[i], &out.Conditions[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	return nil
}

func deepCopy_v1_ComponentStatusList(in ComponentStatusList, out *ComponentStatusList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ComponentStatus, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_ComponentStatus(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_Container(in Container, out *Container, c *conversion.Cloner) error {
	out.Name = in.Name
	out.Image = in.Image
	if in.Command != nil {
		out.Command = make([]string, len(in.Command))
		for i := range in.Command {
			out.Command[i] = in.Command[i]
		}
	} else {
		out.Command = nil
	}
	if in.Args != nil {
		out.Args = make([]string, len(in.Args))
		for i := range in.Args {
			out.Args[i] = in.Args[i]
		}
	} else {
		out.Args = nil
	}
	out.WorkingDir = in.WorkingDir
	if in.Ports != nil {
		out.Ports = make([]ContainerPort, len(in.Ports))
		for i := range in.Ports {
			if err := deepCopy_v1_ContainerPort(in.Ports[i], &out.Ports[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	if in.Env != nil {
		out.Env = make([]EnvVar, len(in.Env))
		for i := range in.Env {
			if err := deepCopy_v1_EnvVar(in.Env[i], &out.Env[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Env = nil
	}
	if err := deepCopy_v1_ResourceRequirements(in.Resources, &out.Resources, c); err != nil {
		return err
	}
	if in.VolumeMounts != nil {
		out.VolumeMounts = make([]VolumeMount, len(in.VolumeMounts))
		for i := range in.VolumeMounts {
			if err := deepCopy_v1_VolumeMount(in.VolumeMounts[i], &out.VolumeMounts[i], c); err != nil {
				return err
			}
		}
	} else {
		out.VolumeMounts = nil
	}
	if in.LivenessProbe != nil {
		out.LivenessProbe = new(Probe)
		if err := deepCopy_v1_Probe(*in.LivenessProbe, out.LivenessProbe, c); err != nil {
			return err
		}
	} else {
		out.LivenessProbe = nil
	}
	if in.ReadinessProbe != nil {
		out.ReadinessProbe = new(Probe)
		if err := deepCopy_v1_Probe(*in.ReadinessProbe, out.ReadinessProbe, c); err != nil {
			return err
		}
	} else {
		out.ReadinessProbe = nil
	}
	if in.Lifecycle != nil {
		out.Lifecycle = new(Lifecycle)
		if err := deepCopy_v1_Lifecycle(*in.Lifecycle, out.Lifecycle, c); err != nil {
			return err
		}
	} else {
		out.Lifecycle = nil
	}
	out.TerminationMessagePath = in.TerminationMessagePath
	out.ImagePullPolicy = in.ImagePullPolicy
	if in.SecurityContext != nil {
		out.SecurityContext = new(SecurityContext)
		if err := deepCopy_v1_SecurityContext(*in.SecurityContext, out.SecurityContext, c); err != nil {
			return err
		}
	} else {
		out.SecurityContext = nil
	}
	return nil
}

func deepCopy_v1_ContainerPort(in ContainerPort, out *ContainerPort, c *conversion.Cloner) error {
	out.Name = in.Name
	out.HostPort = in.HostPort
	out.ContainerPort = in.ContainerPort
	out.Protocol = in.Protocol
	out.HostIP = in.HostIP
	return nil
}

func deepCopy_v1_ContainerState(in ContainerState, out *ContainerState, c *conversion.Cloner) error {
	if in.Waiting != nil {
		out.Waiting = new(ContainerStateWaiting)
		if err := deepCopy_v1_ContainerStateWaiting(*in.Waiting, out.Waiting, c); err != nil {
			return err
		}
	} else {
		out.Waiting = nil
	}
	if in.Running != nil {
		out.Running = new(ContainerStateRunning)
		if err := deepCopy_v1_ContainerStateRunning(*in.Running, out.Running, c); err != nil {
			return err
		}
	} else {
		out.Running = nil
	}
	if in.Terminated != nil {
		out.Terminated = new(ContainerStateTerminated)
		if err := deepCopy_v1_ContainerStateTerminated(*in.Terminated, out.Terminated, c); err != nil {
			return err
		}
	} else {
		out.Terminated = nil
	}
	return nil
}

func deepCopy_v1_ContainerStateRunning(in ContainerStateRunning, out *ContainerStateRunning, c *conversion.Cloner) error {
	if err := deepCopy_util_Time(in.StartedAt, &out.StartedAt, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_ContainerStateTerminated(in ContainerStateTerminated, out *ContainerStateTerminated, c *conversion.Cloner) error {
	out.ExitCode = in.ExitCode
	out.Signal = in.Signal
	out.Reason = in.Reason
	out.Message = in.Message
	if err := deepCopy_util_Time(in.StartedAt, &out.StartedAt, c); err != nil {
		return err
	}
	if err := deepCopy_util_Time(in.FinishedAt, &out.FinishedAt, c); err != nil {
		return err
	}
	out.ContainerID = in.ContainerID
	return nil
}

func deepCopy_v1_ContainerStateWaiting(in ContainerStateWaiting, out *ContainerStateWaiting, c *conversion.Cloner) error {
	out.Reason = in.Reason
	return nil
}

func deepCopy_v1_ContainerStatus(in ContainerStatus, out *ContainerStatus, c *conversion.Cloner) error {
	out.Name = in.Name
	if err := deepCopy_v1_ContainerState(in.State, &out.State, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ContainerState(in.LastTerminationState, &out.LastTerminationState, c); err != nil {
		return err
	}
	out.Ready = in.Ready
	out.RestartCount = in.RestartCount
	out.Image = in.Image
	out.ImageID = in.ImageID
	out.ContainerID = in.ContainerID
	return nil
}

func deepCopy_v1_DeleteOptions(in DeleteOptions, out *DeleteOptions, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if in.GracePeriodSeconds != nil {
		out.GracePeriodSeconds = new(int64)
		*out.GracePeriodSeconds = *in.GracePeriodSeconds
	} else {
		out.GracePeriodSeconds = nil
	}
	return nil
}

func deepCopy_v1_EmptyDirVolumeSource(in EmptyDirVolumeSource, out *EmptyDirVolumeSource, c *conversion.Cloner) error {
	out.Medium = in.Medium
	return nil
}

func deepCopy_v1_EndpointAddress(in EndpointAddress, out *EndpointAddress, c *conversion.Cloner) error {
	out.IP = in.IP
	if in.TargetRef != nil {
		out.TargetRef = new(ObjectReference)
		if err := deepCopy_v1_ObjectReference(*in.TargetRef, out.TargetRef, c); err != nil {
			return err
		}
	} else {
		out.TargetRef = nil
	}
	return nil
}

func deepCopy_v1_EndpointPort(in EndpointPort, out *EndpointPort, c *conversion.Cloner) error {
	out.Name = in.Name
	out.Port = in.Port
	out.Protocol = in.Protocol
	return nil
}

func deepCopy_v1_EndpointSubset(in EndpointSubset, out *EndpointSubset, c *conversion.Cloner) error {
	if in.Addresses != nil {
		out.Addresses = make([]EndpointAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := deepCopy_v1_EndpointAddress(in.Addresses[i], &out.Addresses[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if in.Ports != nil {
		out.Ports = make([]EndpointPort, len(in.Ports))
		for i := range in.Ports {
			if err := deepCopy_v1_EndpointPort(in.Ports[i], &out.Ports[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	return nil
}

func deepCopy_v1_Endpoints(in Endpoints, out *Endpoints, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if in.Subsets != nil {
		out.Subsets = make([]EndpointSubset, len(in.Subsets))
		for i := range in.Subsets {
			if err := deepCopy_v1_EndpointSubset(in.Subsets[i], &out.Subsets[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Subsets = nil
	}
	return nil
}

func deepCopy_v1_EndpointsList(in EndpointsList, out *EndpointsList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Endpoints, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_Endpoints(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_EnvVar(in EnvVar, out *EnvVar, c *conversion.Cloner) error {
	out.Name = in.Name
	out.Value = in.Value
	if in.ValueFrom != nil {
		out.ValueFrom = new(EnvVarSource)
		if err := deepCopy_v1_EnvVarSource(*in.ValueFrom, out.ValueFrom, c); err != nil {
			return err
		}
	} else {
		out.ValueFrom = nil
	}
	return nil
}

func deepCopy_v1_EnvVarSource(in EnvVarSource, out *EnvVarSource, c *conversion.Cloner) error {
	if in.FieldRef != nil {
		out.FieldRef = new(ObjectFieldSelector)
		if err := deepCopy_v1_ObjectFieldSelector(*in.FieldRef, out.FieldRef, c); err != nil {
			return err
		}
	} else {
		out.FieldRef = nil
	}
	return nil
}

func deepCopy_v1_Event(in Event, out *Event, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectReference(in.InvolvedObject, &out.InvolvedObject, c); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	if err := deepCopy_v1_EventSource(in.Source, &out.Source, c); err != nil {
		return err
	}
	if err := deepCopy_util_Time(in.FirstTimestamp, &out.FirstTimestamp, c); err != nil {
		return err
	}
	if err := deepCopy_util_Time(in.LastTimestamp, &out.LastTimestamp, c); err != nil {
		return err
	}
	out.Count = in.Count
	return nil
}

func deepCopy_v1_EventList(in EventList, out *EventList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Event, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_Event(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_EventSource(in EventSource, out *EventSource, c *conversion.Cloner) error {
	out.Component = in.Component
	out.Host = in.Host
	return nil
}

func deepCopy_v1_ExecAction(in ExecAction, out *ExecAction, c *conversion.Cloner) error {
	if in.Command != nil {
		out.Command = make([]string, len(in.Command))
		for i := range in.Command {
			out.Command[i] = in.Command[i]
		}
	} else {
		out.Command = nil
	}
	return nil
}

func deepCopy_v1_GCEPersistentDiskVolumeSource(in GCEPersistentDiskVolumeSource, out *GCEPersistentDiskVolumeSource, c *conversion.Cloner) error {
	out.PDName = in.PDName
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func deepCopy_v1_GitRepoVolumeSource(in GitRepoVolumeSource, out *GitRepoVolumeSource, c *conversion.Cloner) error {
	out.Repository = in.Repository
	out.Revision = in.Revision
	return nil
}

func deepCopy_v1_GlusterfsVolumeSource(in GlusterfsVolumeSource, out *GlusterfsVolumeSource, c *conversion.Cloner) error {
	out.EndpointsName = in.EndpointsName
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func deepCopy_v1_HTTPGetAction(in HTTPGetAction, out *HTTPGetAction, c *conversion.Cloner) error {
	out.Path = in.Path
	if err := deepCopy_util_IntOrString(in.Port, &out.Port, c); err != nil {
		return err
	}
	out.Host = in.Host
	out.Scheme = in.Scheme
	return nil
}

func deepCopy_v1_Handler(in Handler, out *Handler, c *conversion.Cloner) error {
	if in.Exec != nil {
		out.Exec = new(ExecAction)
		if err := deepCopy_v1_ExecAction(*in.Exec, out.Exec, c); err != nil {
			return err
		}
	} else {
		out.Exec = nil
	}
	if in.HTTPGet != nil {
		out.HTTPGet = new(HTTPGetAction)
		if err := deepCopy_v1_HTTPGetAction(*in.HTTPGet, out.HTTPGet, c); err != nil {
			return err
		}
	} else {
		out.HTTPGet = nil
	}
	if in.TCPSocket != nil {
		out.TCPSocket = new(TCPSocketAction)
		if err := deepCopy_v1_TCPSocketAction(*in.TCPSocket, out.TCPSocket, c); err != nil {
			return err
		}
	} else {
		out.TCPSocket = nil
	}
	return nil
}

func deepCopy_v1_HostPathVolumeSource(in HostPathVolumeSource, out *HostPathVolumeSource, c *conversion.Cloner) error {
	out.Path = in.Path
	return nil
}

func deepCopy_v1_ISCSIVolumeSource(in ISCSIVolumeSource, out *ISCSIVolumeSource, c *conversion.Cloner) error {
	out.TargetPortal = in.TargetPortal
	out.IQN = in.IQN
	out.Lun = in.Lun
	out.FSType = in.FSType
	out.ReadOnly = in.ReadOnly
	return nil
}

func deepCopy_v1_Lifecycle(in Lifecycle, out *Lifecycle, c *conversion.Cloner) error {
	if in.PostStart != nil {
		out.PostStart = new(Handler)
		if err := deepCopy_v1_Handler(*in.PostStart, out.PostStart, c); err != nil {
			return err
		}
	} else {
		out.PostStart = nil
	}
	if in.PreStop != nil {
		out.PreStop = new(Handler)
		if err := deepCopy_v1_Handler(*in.PreStop, out.PreStop, c); err != nil {
			return err
		}
	} else {
		out.PreStop = nil
	}
	return nil
}

func deepCopy_v1_LimitRange(in LimitRange, out *LimitRange, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_LimitRangeSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_LimitRangeItem(in LimitRangeItem, out *LimitRangeItem, c *conversion.Cloner) error {
	out.Type = in.Type
	if in.Max != nil {
		out.Max = make(ResourceList)
		for key, val := range in.Max {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Max[key] = *newVal
		}
	} else {
		out.Max = nil
	}
	if in.Min != nil {
		out.Min = make(ResourceList)
		for key, val := range in.Min {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Min[key] = *newVal
		}
	} else {
		out.Min = nil
	}
	if in.Default != nil {
		out.Default = make(ResourceList)
		for key, val := range in.Default {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Default[key] = *newVal
		}
	} else {
		out.Default = nil
	}
	return nil
}

func deepCopy_v1_LimitRangeList(in LimitRangeList, out *LimitRangeList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]LimitRange, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_LimitRange(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_LimitRangeSpec(in LimitRangeSpec, out *LimitRangeSpec, c *conversion.Cloner) error {
	if in.Limits != nil {
		out.Limits = make([]LimitRangeItem, len(in.Limits))
		for i := range in.Limits {
			if err := deepCopy_v1_LimitRangeItem(in.Limits[i], &out.Limits[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Limits = nil
	}
	return nil
}

func deepCopy_v1_List(in List, out *List, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]runtime.RawExtension, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_runtime_RawExtension(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_ListMeta(in ListMeta, out *ListMeta, c *conversion.Cloner) error {
	out.SelfLink = in.SelfLink
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func deepCopy_v1_ListOptions(in ListOptions, out *ListOptions, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	out.LabelSelector = in.LabelSelector
	out.FieldSelector = in.FieldSelector
	out.Watch = in.Watch
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func deepCopy_v1_LoadBalancerIngress(in LoadBalancerIngress, out *LoadBalancerIngress, c *conversion.Cloner) error {
	out.IP = in.IP
	out.Hostname = in.Hostname
	return nil
}

func deepCopy_v1_LoadBalancerStatus(in LoadBalancerStatus, out *LoadBalancerStatus, c *conversion.Cloner) error {
	if in.Ingress != nil {
		out.Ingress = make([]LoadBalancerIngress, len(in.Ingress))
		for i := range in.Ingress {
			if err := deepCopy_v1_LoadBalancerIngress(in.Ingress[i], &out.Ingress[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Ingress = nil
	}
	return nil
}

func deepCopy_v1_LocalObjectReference(in LocalObjectReference, out *LocalObjectReference, c *conversion.Cloner) error {
	out.Name = in.Name
	return nil
}

func deepCopy_v1_NFSVolumeSource(in NFSVolumeSource, out *NFSVolumeSource, c *conversion.Cloner) error {
	out.Server = in.Server
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func deepCopy_v1_Namespace(in Namespace, out *Namespace, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_NamespaceSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	if err := deepCopy_v1_NamespaceStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_NamespaceList(in NamespaceList, out *NamespaceList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Namespace, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_Namespace(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_NamespaceSpec(in NamespaceSpec, out *NamespaceSpec, c *conversion.Cloner) error {
	if in.Finalizers != nil {
		out.Finalizers = make([]FinalizerName, len(in.Finalizers))
		for i := range in.Finalizers {
			out.Finalizers[i] = in.Finalizers[i]
		}
	} else {
		out.Finalizers = nil
	}
	return nil
}

func deepCopy_v1_NamespaceStatus(in NamespaceStatus, out *NamespaceStatus, c *conversion.Cloner) error {
	out.Phase = in.Phase
	return nil
}

func deepCopy_v1_Node(in Node, out *Node, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_NodeSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	if err := deepCopy_v1_NodeStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_NodeAddress(in NodeAddress, out *NodeAddress, c *conversion.Cloner) error {
	out.Type = in.Type
	out.Address = in.Address
	return nil
}

func deepCopy_v1_NodeCondition(in NodeCondition, out *NodeCondition, c *conversion.Cloner) error {
	out.Type = in.Type
	out.Status = in.Status
	if err := deepCopy_util_Time(in.LastHeartbeatTime, &out.LastHeartbeatTime, c); err != nil {
		return err
	}
	if err := deepCopy_util_Time(in.LastTransitionTime, &out.LastTransitionTime, c); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	return nil
}

func deepCopy_v1_NodeList(in NodeList, out *NodeList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Node, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_Node(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_NodeSpec(in NodeSpec, out *NodeSpec, c *conversion.Cloner) error {
	out.PodCIDR = in.PodCIDR
	out.ExternalID = in.ExternalID
	out.ProviderID = in.ProviderID
	out.Unschedulable = in.Unschedulable
	return nil
}

func deepCopy_v1_NodeStatus(in NodeStatus, out *NodeStatus, c *conversion.Cloner) error {
	if in.Capacity != nil {
		out.Capacity = make(ResourceList)
		for key, val := range in.Capacity {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Capacity[key] = *newVal
		}
	} else {
		out.Capacity = nil
	}
	out.Phase = in.Phase
	if in.Conditions != nil {
		out.Conditions = make([]NodeCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := deepCopy_v1_NodeCondition(in.Conditions[i], &out.Conditions[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	if in.Addresses != nil {
		out.Addresses = make([]NodeAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := deepCopy_v1_NodeAddress(in.Addresses[i], &out.Addresses[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if err := deepCopy_v1_NodeSystemInfo(in.NodeInfo, &out.NodeInfo, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_NodeSystemInfo(in NodeSystemInfo, out *NodeSystemInfo, c *conversion.Cloner) error {
	out.MachineID = in.MachineID
	out.SystemUUID = in.SystemUUID
	out.BootID = in.BootID
	out.KernelVersion = in.KernelVersion
	out.OsImage = in.OsImage
	out.ContainerRuntimeVersion = in.ContainerRuntimeVersion
	out.KubeletVersion = in.KubeletVersion
	out.KubeProxyVersion = in.KubeProxyVersion
	return nil
}

func deepCopy_v1_ObjectFieldSelector(in ObjectFieldSelector, out *ObjectFieldSelector, c *conversion.Cloner) error {
	out.APIVersion = in.APIVersion
	out.FieldPath = in.FieldPath
	return nil
}

func deepCopy_v1_ObjectMeta(in ObjectMeta, out *ObjectMeta, c *conversion.Cloner) error {
	out.Name = in.Name
	out.GenerateName = in.GenerateName
	out.Namespace = in.Namespace
	out.SelfLink = in.SelfLink
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
	out.Generation = in.Generation
	if err := deepCopy_util_Time(in.CreationTimestamp, &out.CreationTimestamp, c); err != nil {
		return err
	}
	if in.DeletionTimestamp != nil {
		out.DeletionTimestamp = new(util.Time)
		if err := deepCopy_util_Time(*in.DeletionTimestamp, out.DeletionTimestamp, c); err != nil {
			return err
		}
	} else {
		out.DeletionTimestamp = nil
	}
	if in.Labels != nil {
		out.Labels = make(map[string]string)
		for key, val := range in.Labels {
			out.Labels[key] = val
		}
	} else {
		out.Labels = nil
	}
	if in.Annotations != nil {
		out.Annotations = make(map[string]string)
		for key, val := range in.Annotations {
			out.Annotations[key] = val
		}
	} else {
		out.Annotations = nil
	}
	return nil
}

func deepCopy_v1_ObjectReference(in ObjectReference, out *ObjectReference, c *conversion.Cloner) error {
	out.Kind = in.Kind
	out.Namespace = in.Namespace
	out.Name = in.Name
	out.UID = in.UID
	out.APIVersion = in.APIVersion
	out.ResourceVersion = in.ResourceVersion
	out.FieldPath = in.FieldPath
	return nil
}

func deepCopy_v1_PersistentVolume(in PersistentVolume, out *PersistentVolume, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PersistentVolumeSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PersistentVolumeStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_PersistentVolumeClaim(in PersistentVolumeClaim, out *PersistentVolumeClaim, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PersistentVolumeClaimSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PersistentVolumeClaimStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_PersistentVolumeClaimList(in PersistentVolumeClaimList, out *PersistentVolumeClaimList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PersistentVolumeClaim, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_PersistentVolumeClaim(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_PersistentVolumeClaimSpec(in PersistentVolumeClaimSpec, out *PersistentVolumeClaimSpec, c *conversion.Cloner) error {
	if in.AccessModes != nil {
		out.AccessModes = make([]PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = in.AccessModes[i]
		}
	} else {
		out.AccessModes = nil
	}
	if err := deepCopy_v1_ResourceRequirements(in.Resources, &out.Resources, c); err != nil {
		return err
	}
	out.VolumeName = in.VolumeName
	return nil
}

func deepCopy_v1_PersistentVolumeClaimStatus(in PersistentVolumeClaimStatus, out *PersistentVolumeClaimStatus, c *conversion.Cloner) error {
	out.Phase = in.Phase
	if in.AccessModes != nil {
		out.AccessModes = make([]PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = in.AccessModes[i]
		}
	} else {
		out.AccessModes = nil
	}
	if in.Capacity != nil {
		out.Capacity = make(ResourceList)
		for key, val := range in.Capacity {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Capacity[key] = *newVal
		}
	} else {
		out.Capacity = nil
	}
	return nil
}

func deepCopy_v1_PersistentVolumeClaimVolumeSource(in PersistentVolumeClaimVolumeSource, out *PersistentVolumeClaimVolumeSource, c *conversion.Cloner) error {
	out.ClaimName = in.ClaimName
	out.ReadOnly = in.ReadOnly
	return nil
}

func deepCopy_v1_PersistentVolumeList(in PersistentVolumeList, out *PersistentVolumeList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PersistentVolume, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_PersistentVolume(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_PersistentVolumeSource(in PersistentVolumeSource, out *PersistentVolumeSource, c *conversion.Cloner) error {
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(GCEPersistentDiskVolumeSource)
		if err := deepCopy_v1_GCEPersistentDiskVolumeSource(*in.GCEPersistentDisk, out.GCEPersistentDisk, c); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(AWSElasticBlockStoreVolumeSource)
		if err := deepCopy_v1_AWSElasticBlockStoreVolumeSource(*in.AWSElasticBlockStore, out.AWSElasticBlockStore, c); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.HostPath != nil {
		out.HostPath = new(HostPathVolumeSource)
		if err := deepCopy_v1_HostPathVolumeSource(*in.HostPath, out.HostPath, c); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(GlusterfsVolumeSource)
		if err := deepCopy_v1_GlusterfsVolumeSource(*in.Glusterfs, out.Glusterfs, c); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.NFS != nil {
		out.NFS = new(NFSVolumeSource)
		if err := deepCopy_v1_NFSVolumeSource(*in.NFS, out.NFS, c); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	if in.RBD != nil {
		out.RBD = new(RBDVolumeSource)
		if err := deepCopy_v1_RBDVolumeSource(*in.RBD, out.RBD, c); err != nil {
			return err
		}
	} else {
		out.RBD = nil
	}
	if in.ISCSI != nil {
		out.ISCSI = new(ISCSIVolumeSource)
		if err := deepCopy_v1_ISCSIVolumeSource(*in.ISCSI, out.ISCSI, c); err != nil {
			return err
		}
	} else {
		out.ISCSI = nil
	}
	return nil
}

func deepCopy_v1_PersistentVolumeSpec(in PersistentVolumeSpec, out *PersistentVolumeSpec, c *conversion.Cloner) error {
	if in.Capacity != nil {
		out.Capacity = make(ResourceList)
		for key, val := range in.Capacity {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Capacity[key] = *newVal
		}
	} else {
		out.Capacity = nil
	}
	if err := deepCopy_v1_PersistentVolumeSource(in.PersistentVolumeSource, &out.PersistentVolumeSource, c); err != nil {
		return err
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = in.AccessModes[i]
		}
	} else {
		out.AccessModes = nil
	}
	if in.ClaimRef != nil {
		out.ClaimRef = new(ObjectReference)
		if err := deepCopy_v1_ObjectReference(*in.ClaimRef, out.ClaimRef, c); err != nil {
			return err
		}
	} else {
		out.ClaimRef = nil
	}
	out.PersistentVolumeReclaimPolicy = in.PersistentVolumeReclaimPolicy
	return nil
}

func deepCopy_v1_PersistentVolumeStatus(in PersistentVolumeStatus, out *PersistentVolumeStatus, c *conversion.Cloner) error {
	out.Phase = in.Phase
	out.Message = in.Message
	out.Reason = in.Reason
	return nil
}

func deepCopy_v1_Pod(in Pod, out *Pod, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PodSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PodStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_PodAttachOptions(in PodAttachOptions, out *PodAttachOptions, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	out.Stdin = in.Stdin
	out.Stdout = in.Stdout
	out.Stderr = in.Stderr
	out.TTY = in.TTY
	out.Container = in.Container
	return nil
}

func deepCopy_v1_PodCondition(in PodCondition, out *PodCondition, c *conversion.Cloner) error {
	out.Type = in.Type
	out.Status = in.Status
	return nil
}

func deepCopy_v1_PodExecOptions(in PodExecOptions, out *PodExecOptions, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	out.Stdin = in.Stdin
	out.Stdout = in.Stdout
	out.Stderr = in.Stderr
	out.TTY = in.TTY
	out.Container = in.Container
	if in.Command != nil {
		out.Command = make([]string, len(in.Command))
		for i := range in.Command {
			out.Command[i] = in.Command[i]
		}
	} else {
		out.Command = nil
	}
	return nil
}

func deepCopy_v1_PodList(in PodList, out *PodList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Pod, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_Pod(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_PodLogOptions(in PodLogOptions, out *PodLogOptions, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	out.Container = in.Container
	out.Follow = in.Follow
	out.Previous = in.Previous
	return nil
}

func deepCopy_v1_PodProxyOptions(in PodProxyOptions, out *PodProxyOptions, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	out.Path = in.Path
	return nil
}

func deepCopy_v1_PodSpec(in PodSpec, out *PodSpec, c *conversion.Cloner) error {
	if in.Volumes != nil {
		out.Volumes = make([]Volume, len(in.Volumes))
		for i := range in.Volumes {
			if err := deepCopy_v1_Volume(in.Volumes[i], &out.Volumes[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Volumes = nil
	}
	if in.Containers != nil {
		out.Containers = make([]Container, len(in.Containers))
		for i := range in.Containers {
			if err := deepCopy_v1_Container(in.Containers[i], &out.Containers[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Containers = nil
	}
	out.RestartPolicy = in.RestartPolicy
	if in.TerminationGracePeriodSeconds != nil {
		out.TerminationGracePeriodSeconds = new(int64)
		*out.TerminationGracePeriodSeconds = *in.TerminationGracePeriodSeconds
	} else {
		out.TerminationGracePeriodSeconds = nil
	}
	if in.ActiveDeadlineSeconds != nil {
		out.ActiveDeadlineSeconds = new(int64)
		*out.ActiveDeadlineSeconds = *in.ActiveDeadlineSeconds
	} else {
		out.ActiveDeadlineSeconds = nil
	}
	out.DNSPolicy = in.DNSPolicy
	if in.NodeSelector != nil {
		out.NodeSelector = make(map[string]string)
		for key, val := range in.NodeSelector {
			out.NodeSelector[key] = val
		}
	} else {
		out.NodeSelector = nil
	}
	out.ServiceAccountName = in.ServiceAccountName
	out.DeprecatedServiceAccount = in.DeprecatedServiceAccount
	out.NodeName = in.NodeName
	out.HostNetwork = in.HostNetwork
	if in.ImagePullSecrets != nil {
		out.ImagePullSecrets = make([]LocalObjectReference, len(in.ImagePullSecrets))
		for i := range in.ImagePullSecrets {
			if err := deepCopy_v1_LocalObjectReference(in.ImagePullSecrets[i], &out.ImagePullSecrets[i], c); err != nil {
				return err
			}
		}
	} else {
		out.ImagePullSecrets = nil
	}
	return nil
}

func deepCopy_v1_PodStatus(in PodStatus, out *PodStatus, c *conversion.Cloner) error {
	out.Phase = in.Phase
	if in.Conditions != nil {
		out.Conditions = make([]PodCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := deepCopy_v1_PodCondition(in.Conditions[i], &out.Conditions[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	out.Message = in.Message
	out.Reason = in.Reason
	out.HostIP = in.HostIP
	out.PodIP = in.PodIP
	if in.StartTime != nil {
		out.StartTime = new(util.Time)
		if err := deepCopy_util_Time(*in.StartTime, out.StartTime, c); err != nil {
			return err
		}
	} else {
		out.StartTime = nil
	}
	if in.ContainerStatuses != nil {
		out.ContainerStatuses = make([]ContainerStatus, len(in.ContainerStatuses))
		for i := range in.ContainerStatuses {
			if err := deepCopy_v1_ContainerStatus(in.ContainerStatuses[i], &out.ContainerStatuses[i], c); err != nil {
				return err
			}
		}
	} else {
		out.ContainerStatuses = nil
	}
	return nil
}

func deepCopy_v1_PodStatusResult(in PodStatusResult, out *PodStatusResult, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PodStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_PodTemplate(in PodTemplate, out *PodTemplate, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PodTemplateSpec(in.Template, &out.Template, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_PodTemplateList(in PodTemplateList, out *PodTemplateList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PodTemplate, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_PodTemplate(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_PodTemplateSpec(in PodTemplateSpec, out *PodTemplateSpec, c *conversion.Cloner) error {
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_PodSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_Probe(in Probe, out *Probe, c *conversion.Cloner) error {
	if err := deepCopy_v1_Handler(in.Handler, &out.Handler, c); err != nil {
		return err
	}
	out.InitialDelaySeconds = in.InitialDelaySeconds
	out.TimeoutSeconds = in.TimeoutSeconds
	return nil
}

func deepCopy_v1_RBDVolumeSource(in RBDVolumeSource, out *RBDVolumeSource, c *conversion.Cloner) error {
	if in.CephMonitors != nil {
		out.CephMonitors = make([]string, len(in.CephMonitors))
		for i := range in.CephMonitors {
			out.CephMonitors[i] = in.CephMonitors[i]
		}
	} else {
		out.CephMonitors = nil
	}
	out.RBDImage = in.RBDImage
	out.FSType = in.FSType
	out.RBDPool = in.RBDPool
	out.RadosUser = in.RadosUser
	out.Keyring = in.Keyring
	if in.SecretRef != nil {
		out.SecretRef = new(LocalObjectReference)
		if err := deepCopy_v1_LocalObjectReference(*in.SecretRef, out.SecretRef, c); err != nil {
			return err
		}
	} else {
		out.SecretRef = nil
	}
	out.ReadOnly = in.ReadOnly
	return nil
}

func deepCopy_v1_RangeAllocation(in RangeAllocation, out *RangeAllocation, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	out.Range = in.Range
	if in.Data != nil {
		out.Data = make([]uint8, len(in.Data))
		for i := range in.Data {
			out.Data[i] = in.Data[i]
		}
	} else {
		out.Data = nil
	}
	return nil
}

func deepCopy_v1_ReplicationController(in ReplicationController, out *ReplicationController, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ReplicationControllerSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ReplicationControllerStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_ReplicationControllerList(in ReplicationControllerList, out *ReplicationControllerList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ReplicationController, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_ReplicationController(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_ReplicationControllerSpec(in ReplicationControllerSpec, out *ReplicationControllerSpec, c *conversion.Cloner) error {
	if in.Replicas != nil {
		out.Replicas = new(int)
		*out.Replicas = *in.Replicas
	} else {
		out.Replicas = nil
	}
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	} else {
		out.Selector = nil
	}
	if in.Template != nil {
		out.Template = new(PodTemplateSpec)
		if err := deepCopy_v1_PodTemplateSpec(*in.Template, out.Template, c); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}

func deepCopy_v1_ReplicationControllerStatus(in ReplicationControllerStatus, out *ReplicationControllerStatus, c *conversion.Cloner) error {
	out.Replicas = in.Replicas
	out.ObservedGeneration = in.ObservedGeneration
	return nil
}

func deepCopy_v1_ResourceQuota(in ResourceQuota, out *ResourceQuota, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ResourceQuotaSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ResourceQuotaStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_ResourceQuotaList(in ResourceQuotaList, out *ResourceQuotaList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ResourceQuota, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_ResourceQuota(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_ResourceQuotaSpec(in ResourceQuotaSpec, out *ResourceQuotaSpec, c *conversion.Cloner) error {
	if in.Hard != nil {
		out.Hard = make(ResourceList)
		for key, val := range in.Hard {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Hard[key] = *newVal
		}
	} else {
		out.Hard = nil
	}
	return nil
}

func deepCopy_v1_ResourceQuotaStatus(in ResourceQuotaStatus, out *ResourceQuotaStatus, c *conversion.Cloner) error {
	if in.Hard != nil {
		out.Hard = make(ResourceList)
		for key, val := range in.Hard {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Hard[key] = *newVal
		}
	} else {
		out.Hard = nil
	}
	if in.Used != nil {
		out.Used = make(ResourceList)
		for key, val := range in.Used {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Used[key] = *newVal
		}
	} else {
		out.Used = nil
	}
	return nil
}

func deepCopy_v1_ResourceRequirements(in ResourceRequirements, out *ResourceRequirements, c *conversion.Cloner) error {
	if in.Limits != nil {
		out.Limits = make(ResourceList)
		for key, val := range in.Limits {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Limits[key] = *newVal
		}
	} else {
		out.Limits = nil
	}
	if in.Requests != nil {
		out.Requests = make(ResourceList)
		for key, val := range in.Requests {
			newVal := new(resource.Quantity)
			if err := deepCopy_resource_Quantity(val, newVal, c); err != nil {
				return err
			}
			out.Requests[key] = *newVal
		}
	} else {
		out.Requests = nil
	}
	return nil
}

func deepCopy_v1_SELinuxOptions(in SELinuxOptions, out *SELinuxOptions, c *conversion.Cloner) error {
	out.User = in.User
	out.Role = in.Role
	out.Type = in.Type
	out.Level = in.Level
	return nil
}

func deepCopy_v1_Secret(in Secret, out *Secret, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if in.Data != nil {
		out.Data = make(map[string][]uint8)
		for key, val := range in.Data {
			if newVal, err := c.DeepCopy(val); err != nil {
				return err
			} else {
				out.Data[key] = newVal.([]uint8)
			}
		}
	} else {
		out.Data = nil
	}
	out.Type = in.Type
	return nil
}

func deepCopy_v1_SecretList(in SecretList, out *SecretList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Secret, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_Secret(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_SecretVolumeSource(in SecretVolumeSource, out *SecretVolumeSource, c *conversion.Cloner) error {
	out.SecretName = in.SecretName
	return nil
}

func deepCopy_v1_SecurityContext(in SecurityContext, out *SecurityContext, c *conversion.Cloner) error {
	if in.Capabilities != nil {
		out.Capabilities = new(Capabilities)
		if err := deepCopy_v1_Capabilities(*in.Capabilities, out.Capabilities, c); err != nil {
			return err
		}
	} else {
		out.Capabilities = nil
	}
	if in.Privileged != nil {
		out.Privileged = new(bool)
		*out.Privileged = *in.Privileged
	} else {
		out.Privileged = nil
	}
	if in.SELinuxOptions != nil {
		out.SELinuxOptions = new(SELinuxOptions)
		if err := deepCopy_v1_SELinuxOptions(*in.SELinuxOptions, out.SELinuxOptions, c); err != nil {
			return err
		}
	} else {
		out.SELinuxOptions = nil
	}
	if in.RunAsUser != nil {
		out.RunAsUser = new(int64)
		*out.RunAsUser = *in.RunAsUser
	} else {
		out.RunAsUser = nil
	}
	return nil
}

func deepCopy_v1_SerializedReference(in SerializedReference, out *SerializedReference, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectReference(in.Reference, &out.Reference, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_Service(in Service, out *Service, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ServiceSpec(in.Spec, &out.Spec, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ServiceStatus(in.Status, &out.Status, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_ServiceAccount(in ServiceAccount, out *ServiceAccount, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ObjectMeta(in.ObjectMeta, &out.ObjectMeta, c); err != nil {
		return err
	}
	if in.Secrets != nil {
		out.Secrets = make([]ObjectReference, len(in.Secrets))
		for i := range in.Secrets {
			if err := deepCopy_v1_ObjectReference(in.Secrets[i], &out.Secrets[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Secrets = nil
	}
	if in.ImagePullSecrets != nil {
		out.ImagePullSecrets = make([]LocalObjectReference, len(in.ImagePullSecrets))
		for i := range in.ImagePullSecrets {
			if err := deepCopy_v1_LocalObjectReference(in.ImagePullSecrets[i], &out.ImagePullSecrets[i], c); err != nil {
				return err
			}
		}
	} else {
		out.ImagePullSecrets = nil
	}
	return nil
}

func deepCopy_v1_ServiceAccountList(in ServiceAccountList, out *ServiceAccountList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ServiceAccount, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_ServiceAccount(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_ServiceList(in ServiceList, out *ServiceList, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Service, len(in.Items))
		for i := range in.Items {
			if err := deepCopy_v1_Service(in.Items[i], &out.Items[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func deepCopy_v1_ServicePort(in ServicePort, out *ServicePort, c *conversion.Cloner) error {
	out.Name = in.Name
	out.Protocol = in.Protocol
	out.Port = in.Port
	if err := deepCopy_util_IntOrString(in.TargetPort, &out.TargetPort, c); err != nil {
		return err
	}
	out.NodePort = in.NodePort
	return nil
}

func deepCopy_v1_ServiceSpec(in ServiceSpec, out *ServiceSpec, c *conversion.Cloner) error {
	if in.Ports != nil {
		out.Ports = make([]ServicePort, len(in.Ports))
		for i := range in.Ports {
			if err := deepCopy_v1_ServicePort(in.Ports[i], &out.Ports[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	} else {
		out.Selector = nil
	}
	out.ClusterIP = in.ClusterIP
	out.Type = in.Type
	if in.DeprecatedPublicIPs != nil {
		out.DeprecatedPublicIPs = make([]string, len(in.DeprecatedPublicIPs))
		for i := range in.DeprecatedPublicIPs {
			out.DeprecatedPublicIPs[i] = in.DeprecatedPublicIPs[i]
		}
	} else {
		out.DeprecatedPublicIPs = nil
	}
	out.SessionAffinity = in.SessionAffinity
	return nil
}

func deepCopy_v1_ServiceStatus(in ServiceStatus, out *ServiceStatus, c *conversion.Cloner) error {
	if err := deepCopy_v1_LoadBalancerStatus(in.LoadBalancer, &out.LoadBalancer, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_Status(in Status, out *Status, c *conversion.Cloner) error {
	if err := deepCopy_v1_TypeMeta(in.TypeMeta, &out.TypeMeta, c); err != nil {
		return err
	}
	if err := deepCopy_v1_ListMeta(in.ListMeta, &out.ListMeta, c); err != nil {
		return err
	}
	out.Status = in.Status
	out.Message = in.Message
	out.Reason = in.Reason
	if in.Details != nil {
		out.Details = new(StatusDetails)
		if err := deepCopy_v1_StatusDetails(*in.Details, out.Details, c); err != nil {
			return err
		}
	} else {
		out.Details = nil
	}
	out.Code = in.Code
	return nil
}

func deepCopy_v1_StatusCause(in StatusCause, out *StatusCause, c *conversion.Cloner) error {
	out.Type = in.Type
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func deepCopy_v1_StatusDetails(in StatusDetails, out *StatusDetails, c *conversion.Cloner) error {
	out.Name = in.Name
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := deepCopy_v1_StatusCause(in.Causes[i], &out.Causes[i], c); err != nil {
				return err
			}
		}
	} else {
		out.Causes = nil
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func deepCopy_v1_TCPSocketAction(in TCPSocketAction, out *TCPSocketAction, c *conversion.Cloner) error {
	if err := deepCopy_util_IntOrString(in.Port, &out.Port, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_TypeMeta(in TypeMeta, out *TypeMeta, c *conversion.Cloner) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	return nil
}

func deepCopy_v1_Volume(in Volume, out *Volume, c *conversion.Cloner) error {
	out.Name = in.Name
	if err := deepCopy_v1_VolumeSource(in.VolumeSource, &out.VolumeSource, c); err != nil {
		return err
	}
	return nil
}

func deepCopy_v1_VolumeMount(in VolumeMount, out *VolumeMount, c *conversion.Cloner) error {
	out.Name = in.Name
	out.ReadOnly = in.ReadOnly
	out.MountPath = in.MountPath
	return nil
}

func deepCopy_v1_VolumeSource(in VolumeSource, out *VolumeSource, c *conversion.Cloner) error {
	if in.HostPath != nil {
		out.HostPath = new(HostPathVolumeSource)
		if err := deepCopy_v1_HostPathVolumeSource(*in.HostPath, out.HostPath, c); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.EmptyDir != nil {
		out.EmptyDir = new(EmptyDirVolumeSource)
		if err := deepCopy_v1_EmptyDirVolumeSource(*in.EmptyDir, out.EmptyDir, c); err != nil {
			return err
		}
	} else {
		out.EmptyDir = nil
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(GCEPersistentDiskVolumeSource)
		if err := deepCopy_v1_GCEPersistentDiskVolumeSource(*in.GCEPersistentDisk, out.GCEPersistentDisk, c); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(AWSElasticBlockStoreVolumeSource)
		if err := deepCopy_v1_AWSElasticBlockStoreVolumeSource(*in.AWSElasticBlockStore, out.AWSElasticBlockStore, c); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.GitRepo != nil {
		out.GitRepo = new(GitRepoVolumeSource)
		if err := deepCopy_v1_GitRepoVolumeSource(*in.GitRepo, out.GitRepo, c); err != nil {
			return err
		}
	} else {
		out.GitRepo = nil
	}
	if in.Secret != nil {
		out.Secret = new(SecretVolumeSource)
		if err := deepCopy_v1_SecretVolumeSource(*in.Secret, out.Secret, c); err != nil {
			return err
		}
	} else {
		out.Secret = nil
	}
	if in.NFS != nil {
		out.NFS = new(NFSVolumeSource)
		if err := deepCopy_v1_NFSVolumeSource(*in.NFS, out.NFS, c); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	if in.ISCSI != nil {
		out.ISCSI = new(ISCSIVolumeSource)
		if err := deepCopy_v1_ISCSIVolumeSource(*in.ISCSI, out.ISCSI, c); err != nil {
			return err
		}
	} else {
		out.ISCSI = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(GlusterfsVolumeSource)
		if err := deepCopy_v1_GlusterfsVolumeSource(*in.Glusterfs, out.Glusterfs, c); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.PersistentVolumeClaim != nil {
		out.PersistentVolumeClaim = new(PersistentVolumeClaimVolumeSource)
		if err := deepCopy_v1_PersistentVolumeClaimVolumeSource(*in.PersistentVolumeClaim, out.PersistentVolumeClaim, c); err != nil {
			return err
		}
	} else {
		out.PersistentVolumeClaim = nil
	}
	if in.RBD != nil {
		out.RBD = new(RBDVolumeSource)
		if err := deepCopy_v1_RBDVolumeSource(*in.RBD, out.RBD, c); err != nil {
			return err
		}
	} else {
		out.RBD = nil
	}
	return nil
}

func deepCopy_runtime_RawExtension(in runtime.RawExtension, out *runtime.RawExtension, c *conversion.Cloner) error {
	if in.RawJSON != nil {
		out.RawJSON = make([]uint8, len(in.RawJSON))
		for i := range in.RawJSON {
			out.RawJSON[i] = in.RawJSON[i]
		}
	} else {
		out.RawJSON = nil
	}
	return nil
}

func deepCopy_util_IntOrString(in util.IntOrString, out *util.IntOrString, c *conversion.Cloner) error {
	out.Kind = in.Kind
	out.IntVal = in.IntVal
	out.StrVal = in.StrVal
	return nil
}

func deepCopy_util_Time(in util.Time, out *util.Time, c *conversion.Cloner) error {
	if newVal, err := c.DeepCopy(in.Time); err != nil {
		return err
	} else {
		out.Time = newVal.(time.Time)
	}
	return nil
}

func init() {
	err := api.Scheme.AddGeneratedDeepCopyFuncs(
		deepCopy_resource_Quantity,
		deepCopy_v1_AWSElasticBlockStoreVolumeSource,
		deepCopy_v1_Binding,
		deepCopy_v1_Capabilities,
		deepCopy_v1_ComponentCondition,
		deepCopy_v1_ComponentStatus,
		deepCopy_v1_ComponentStatusList,
		deepCopy_v1_Container,
		deepCopy_v1_ContainerPort,
		deepCopy_v1_ContainerState,
		deepCopy_v1_ContainerStateRunning,
		deepCopy_v1_ContainerStateTerminated,
		deepCopy_v1_ContainerStateWaiting,
		deepCopy_v1_ContainerStatus,
		deepCopy_v1_DeleteOptions,
		deepCopy_v1_EmptyDirVolumeSource,
		deepCopy_v1_EndpointAddress,
		deepCopy_v1_EndpointPort,
		deepCopy_v1_EndpointSubset,
		deepCopy_v1_Endpoints,
		deepCopy_v1_EndpointsList,
		deepCopy_v1_EnvVar,
		deepCopy_v1_EnvVarSource,
		deepCopy_v1_Event,
		deepCopy_v1_EventList,
		deepCopy_v1_EventSource,
		deepCopy_v1_ExecAction,
		deepCopy_v1_GCEPersistentDiskVolumeSource,
		deepCopy_v1_GitRepoVolumeSource,
		deepCopy_v1_GlusterfsVolumeSource,
		deepCopy_v1_HTTPGetAction,
		deepCopy_v1_Handler,
		deepCopy_v1_HostPathVolumeSource,
		deepCopy_v1_ISCSIVolumeSource,
		deepCopy_v1_Lifecycle,
		deepCopy_v1_LimitRange,
		deepCopy_v1_LimitRangeItem,
		deepCopy_v1_LimitRangeList,
		deepCopy_v1_LimitRangeSpec,
		deepCopy_v1_List,
		deepCopy_v1_ListMeta,
		deepCopy_v1_ListOptions,
		deepCopy_v1_LoadBalancerIngress,
		deepCopy_v1_LoadBalancerStatus,
		deepCopy_v1_LocalObjectReference,
		deepCopy_v1_NFSVolumeSource,
		deepCopy_v1_Namespace,
		deepCopy_v1_NamespaceList,
		deepCopy_v1_NamespaceSpec,
		deepCopy_v1_NamespaceStatus,
		deepCopy_v1_Node,
		deepCopy_v1_NodeAddress,
		deepCopy_v1_NodeCondition,
		deepCopy_v1_NodeList,
		deepCopy_v1_NodeSpec,
		deepCopy_v1_NodeStatus,
		deepCopy_v1_NodeSystemInfo,
		deepCopy_v1_ObjectFieldSelector,
		deepCopy_v1_ObjectMeta,
		deepCopy_v1_ObjectReference,
		deepCopy_v1_PersistentVolume,
		deepCopy_v1_PersistentVolumeClaim,
		deepCopy_v1_PersistentVolumeClaimList,
		deepCopy_v1_PersistentVolumeClaimSpec,
		deepCopy_v1_PersistentVolumeClaimStatus,
		deepCopy_v1_PersistentVolumeClaimVolumeSource,
		deepCopy_v1_PersistentVolumeList,
		deepCopy_v1_PersistentVolumeSource,
		deepCopy_v1_PersistentVolumeSpec,
		deepCopy_v1_PersistentVolumeStatus,
		deepCopy_v1_Pod,
		deepCopy_v1_PodAttachOptions,
		deepCopy_v1_PodCondition,
		deepCopy_v1_PodExecOptions,
		deepCopy_v1_PodList,
		deepCopy_v1_PodLogOptions,
		deepCopy_v1_PodProxyOptions,
		deepCopy_v1_PodSpec,
		deepCopy_v1_PodStatus,
		deepCopy_v1_PodStatusResult,
		deepCopy_v1_PodTemplate,
		deepCopy_v1_PodTemplateList,
		deepCopy_v1_PodTemplateSpec,
		deepCopy_v1_Probe,
		deepCopy_v1_RBDVolumeSource,
		deepCopy_v1_RangeAllocation,
		deepCopy_v1_ReplicationController,
		deepCopy_v1_ReplicationControllerList,
		deepCopy_v1_ReplicationControllerSpec,
		deepCopy_v1_ReplicationControllerStatus,
		deepCopy_v1_ResourceQuota,
		deepCopy_v1_ResourceQuotaList,
		deepCopy_v1_ResourceQuotaSpec,
		deepCopy_v1_ResourceQuotaStatus,
		deepCopy_v1_ResourceRequirements,
		deepCopy_v1_SELinuxOptions,
		deepCopy_v1_Secret,
		deepCopy_v1_SecretList,
		deepCopy_v1_SecretVolumeSource,
		deepCopy_v1_SecurityContext,
		deepCopy_v1_SerializedReference,
		deepCopy_v1_Service,
		deepCopy_v1_ServiceAccount,
		deepCopy_v1_ServiceAccountList,
		deepCopy_v1_ServiceList,
		deepCopy_v1_ServicePort,
		deepCopy_v1_ServiceSpec,
		deepCopy_v1_ServiceStatus,
		deepCopy_v1_Status,
		deepCopy_v1_StatusCause,
		deepCopy_v1_StatusDetails,
		deepCopy_v1_TCPSocketAction,
		deepCopy_v1_TypeMeta,
		deepCopy_v1_Volume,
		deepCopy_v1_VolumeMount,
		deepCopy_v1_VolumeSource,
		deepCopy_runtime_RawExtension,
		deepCopy_util_IntOrString,
		deepCopy_util_Time,
	)
	if err != nil {
		// if one of the deep copy functions is malformed, detect it immediately.
		panic(err)
	}
}

// AUTO-GENERATED FUNCTIONS END HERE
