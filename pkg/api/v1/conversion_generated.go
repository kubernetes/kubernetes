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
	reflect "reflect"
)

func convert_api_AWSElasticBlockStoreVolumeSource_To_v1_AWSElasticBlockStoreVolumeSource(in *api.AWSElasticBlockStoreVolumeSource, out *AWSElasticBlockStoreVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.AWSElasticBlockStoreVolumeSource))(in)
	}
	out.VolumeID = in.VolumeID
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_Binding_To_v1_Binding(in *api.Binding, out *Binding, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Binding))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectReference_To_v1_ObjectReference(&in.Target, &out.Target, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Capabilities_To_v1_Capabilities(in *api.Capabilities, out *Capabilities, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Capabilities))(in)
	}
	if in.Add != nil {
		out.Add = make([]Capability, len(in.Add))
		for i := range in.Add {
			out.Add[i] = Capability(in.Add[i])
		}
	} else {
		out.Add = nil
	}
	if in.Drop != nil {
		out.Drop = make([]Capability, len(in.Drop))
		for i := range in.Drop {
			out.Drop[i] = Capability(in.Drop[i])
		}
	} else {
		out.Drop = nil
	}
	return nil
}

func convert_api_ComponentCondition_To_v1_ComponentCondition(in *api.ComponentCondition, out *ComponentCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ComponentCondition))(in)
	}
	out.Type = ComponentConditionType(in.Type)
	out.Status = ConditionStatus(in.Status)
	out.Message = in.Message
	out.Error = in.Error
	return nil
}

func convert_api_ComponentStatus_To_v1_ComponentStatus(in *api.ComponentStatus, out *ComponentStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ComponentStatus))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Conditions != nil {
		out.Conditions = make([]ComponentCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_api_ComponentCondition_To_v1_ComponentCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	return nil
}

func convert_api_ComponentStatusList_To_v1_ComponentStatusList(in *api.ComponentStatusList, out *ComponentStatusList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ComponentStatusList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ComponentStatus, len(in.Items))
		for i := range in.Items {
			if err := convert_api_ComponentStatus_To_v1_ComponentStatus(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_Container_To_v1_Container(in *api.Container, out *Container, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Container))(in)
	}
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
			if err := convert_api_ContainerPort_To_v1_ContainerPort(&in.Ports[i], &out.Ports[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	if in.Env != nil {
		out.Env = make([]EnvVar, len(in.Env))
		for i := range in.Env {
			if err := convert_api_EnvVar_To_v1_EnvVar(&in.Env[i], &out.Env[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Env = nil
	}
	if err := convert_api_ResourceRequirements_To_v1_ResourceRequirements(&in.Resources, &out.Resources, s); err != nil {
		return err
	}
	if in.VolumeMounts != nil {
		out.VolumeMounts = make([]VolumeMount, len(in.VolumeMounts))
		for i := range in.VolumeMounts {
			if err := convert_api_VolumeMount_To_v1_VolumeMount(&in.VolumeMounts[i], &out.VolumeMounts[i], s); err != nil {
				return err
			}
		}
	} else {
		out.VolumeMounts = nil
	}
	if in.LivenessProbe != nil {
		out.LivenessProbe = new(Probe)
		if err := convert_api_Probe_To_v1_Probe(in.LivenessProbe, out.LivenessProbe, s); err != nil {
			return err
		}
	} else {
		out.LivenessProbe = nil
	}
	if in.ReadinessProbe != nil {
		out.ReadinessProbe = new(Probe)
		if err := convert_api_Probe_To_v1_Probe(in.ReadinessProbe, out.ReadinessProbe, s); err != nil {
			return err
		}
	} else {
		out.ReadinessProbe = nil
	}
	if in.Lifecycle != nil {
		out.Lifecycle = new(Lifecycle)
		if err := convert_api_Lifecycle_To_v1_Lifecycle(in.Lifecycle, out.Lifecycle, s); err != nil {
			return err
		}
	} else {
		out.Lifecycle = nil
	}
	out.TerminationMessagePath = in.TerminationMessagePath
	out.ImagePullPolicy = PullPolicy(in.ImagePullPolicy)
	if in.SecurityContext != nil {
		out.SecurityContext = new(SecurityContext)
		if err := convert_api_SecurityContext_To_v1_SecurityContext(in.SecurityContext, out.SecurityContext, s); err != nil {
			return err
		}
	} else {
		out.SecurityContext = nil
	}
	return nil
}

func convert_api_ContainerPort_To_v1_ContainerPort(in *api.ContainerPort, out *ContainerPort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ContainerPort))(in)
	}
	out.Name = in.Name
	out.HostPort = in.HostPort
	out.ContainerPort = in.ContainerPort
	out.Protocol = Protocol(in.Protocol)
	out.HostIP = in.HostIP
	return nil
}

func convert_api_ContainerState_To_v1_ContainerState(in *api.ContainerState, out *ContainerState, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ContainerState))(in)
	}
	if in.Waiting != nil {
		out.Waiting = new(ContainerStateWaiting)
		if err := convert_api_ContainerStateWaiting_To_v1_ContainerStateWaiting(in.Waiting, out.Waiting, s); err != nil {
			return err
		}
	} else {
		out.Waiting = nil
	}
	if in.Running != nil {
		out.Running = new(ContainerStateRunning)
		if err := convert_api_ContainerStateRunning_To_v1_ContainerStateRunning(in.Running, out.Running, s); err != nil {
			return err
		}
	} else {
		out.Running = nil
	}
	if in.Terminated != nil {
		out.Terminated = new(ContainerStateTerminated)
		if err := convert_api_ContainerStateTerminated_To_v1_ContainerStateTerminated(in.Terminated, out.Terminated, s); err != nil {
			return err
		}
	} else {
		out.Terminated = nil
	}
	return nil
}

func convert_api_ContainerStateRunning_To_v1_ContainerStateRunning(in *api.ContainerStateRunning, out *ContainerStateRunning, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ContainerStateRunning))(in)
	}
	if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ContainerStateTerminated_To_v1_ContainerStateTerminated(in *api.ContainerStateTerminated, out *ContainerStateTerminated, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ContainerStateTerminated))(in)
	}
	out.ExitCode = in.ExitCode
	out.Signal = in.Signal
	out.Reason = in.Reason
	out.Message = in.Message
	if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.FinishedAt, &out.FinishedAt, 0); err != nil {
		return err
	}
	out.ContainerID = in.ContainerID
	return nil
}

func convert_api_ContainerStateWaiting_To_v1_ContainerStateWaiting(in *api.ContainerStateWaiting, out *ContainerStateWaiting, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ContainerStateWaiting))(in)
	}
	out.Reason = in.Reason
	return nil
}

func convert_api_ContainerStatus_To_v1_ContainerStatus(in *api.ContainerStatus, out *ContainerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ContainerStatus))(in)
	}
	out.Name = in.Name
	if err := convert_api_ContainerState_To_v1_ContainerState(&in.State, &out.State, s); err != nil {
		return err
	}
	if err := convert_api_ContainerState_To_v1_ContainerState(&in.LastTerminationState, &out.LastTerminationState, s); err != nil {
		return err
	}
	out.Ready = in.Ready
	out.RestartCount = in.RestartCount
	out.Image = in.Image
	out.ImageID = in.ImageID
	out.ContainerID = in.ContainerID
	return nil
}

func convert_api_DeleteOptions_To_v1_DeleteOptions(in *api.DeleteOptions, out *DeleteOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.DeleteOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
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

func convert_api_EmptyDirVolumeSource_To_v1_EmptyDirVolumeSource(in *api.EmptyDirVolumeSource, out *EmptyDirVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EmptyDirVolumeSource))(in)
	}
	out.Medium = StorageMedium(in.Medium)
	return nil
}

func convert_api_EndpointAddress_To_v1_EndpointAddress(in *api.EndpointAddress, out *EndpointAddress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EndpointAddress))(in)
	}
	out.IP = in.IP
	if in.TargetRef != nil {
		out.TargetRef = new(ObjectReference)
		if err := convert_api_ObjectReference_To_v1_ObjectReference(in.TargetRef, out.TargetRef, s); err != nil {
			return err
		}
	} else {
		out.TargetRef = nil
	}
	return nil
}

func convert_api_EndpointPort_To_v1_EndpointPort(in *api.EndpointPort, out *EndpointPort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EndpointPort))(in)
	}
	out.Name = in.Name
	out.Port = in.Port
	out.Protocol = Protocol(in.Protocol)
	return nil
}

func convert_api_EndpointSubset_To_v1_EndpointSubset(in *api.EndpointSubset, out *EndpointSubset, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EndpointSubset))(in)
	}
	if in.Addresses != nil {
		out.Addresses = make([]EndpointAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := convert_api_EndpointAddress_To_v1_EndpointAddress(&in.Addresses[i], &out.Addresses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if in.Ports != nil {
		out.Ports = make([]EndpointPort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_api_EndpointPort_To_v1_EndpointPort(&in.Ports[i], &out.Ports[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	return nil
}

func convert_api_Endpoints_To_v1_Endpoints(in *api.Endpoints, out *Endpoints, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Endpoints))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Subsets != nil {
		out.Subsets = make([]EndpointSubset, len(in.Subsets))
		for i := range in.Subsets {
			if err := convert_api_EndpointSubset_To_v1_EndpointSubset(&in.Subsets[i], &out.Subsets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Subsets = nil
	}
	return nil
}

func convert_api_EndpointsList_To_v1_EndpointsList(in *api.EndpointsList, out *EndpointsList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EndpointsList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Endpoints, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Endpoints_To_v1_Endpoints(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_EnvVar_To_v1_EnvVar(in *api.EnvVar, out *EnvVar, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EnvVar))(in)
	}
	out.Name = in.Name
	out.Value = in.Value
	if in.ValueFrom != nil {
		out.ValueFrom = new(EnvVarSource)
		if err := convert_api_EnvVarSource_To_v1_EnvVarSource(in.ValueFrom, out.ValueFrom, s); err != nil {
			return err
		}
	} else {
		out.ValueFrom = nil
	}
	return nil
}

func convert_api_EnvVarSource_To_v1_EnvVarSource(in *api.EnvVarSource, out *EnvVarSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EnvVarSource))(in)
	}
	if in.FieldRef != nil {
		out.FieldRef = new(ObjectFieldSelector)
		if err := convert_api_ObjectFieldSelector_To_v1_ObjectFieldSelector(in.FieldRef, out.FieldRef, s); err != nil {
			return err
		}
	} else {
		out.FieldRef = nil
	}
	return nil
}

func convert_api_Event_To_v1_Event(in *api.Event, out *Event, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Event))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectReference_To_v1_ObjectReference(&in.InvolvedObject, &out.InvolvedObject, s); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	if err := convert_api_EventSource_To_v1_EventSource(&in.Source, &out.Source, s); err != nil {
		return err
	}
	if err := s.Convert(&in.FirstTimestamp, &out.FirstTimestamp, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.LastTimestamp, &out.LastTimestamp, 0); err != nil {
		return err
	}
	out.Count = in.Count
	return nil
}

func convert_api_EventList_To_v1_EventList(in *api.EventList, out *EventList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EventList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Event, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Event_To_v1_Event(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_EventSource_To_v1_EventSource(in *api.EventSource, out *EventSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.EventSource))(in)
	}
	out.Component = in.Component
	out.Host = in.Host
	return nil
}

func convert_api_ExecAction_To_v1_ExecAction(in *api.ExecAction, out *ExecAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ExecAction))(in)
	}
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

func convert_api_GCEPersistentDiskVolumeSource_To_v1_GCEPersistentDiskVolumeSource(in *api.GCEPersistentDiskVolumeSource, out *GCEPersistentDiskVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.GCEPersistentDiskVolumeSource))(in)
	}
	out.PDName = in.PDName
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_GitRepoVolumeSource_To_v1_GitRepoVolumeSource(in *api.GitRepoVolumeSource, out *GitRepoVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.GitRepoVolumeSource))(in)
	}
	out.Repository = in.Repository
	out.Revision = in.Revision
	return nil
}

func convert_api_GlusterfsVolumeSource_To_v1_GlusterfsVolumeSource(in *api.GlusterfsVolumeSource, out *GlusterfsVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.GlusterfsVolumeSource))(in)
	}
	out.EndpointsName = in.EndpointsName
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_HTTPGetAction_To_v1_HTTPGetAction(in *api.HTTPGetAction, out *HTTPGetAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.HTTPGetAction))(in)
	}
	out.Path = in.Path
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	out.Host = in.Host
	out.Scheme = URIScheme(in.Scheme)
	return nil
}

func convert_api_Handler_To_v1_Handler(in *api.Handler, out *Handler, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Handler))(in)
	}
	if in.Exec != nil {
		out.Exec = new(ExecAction)
		if err := convert_api_ExecAction_To_v1_ExecAction(in.Exec, out.Exec, s); err != nil {
			return err
		}
	} else {
		out.Exec = nil
	}
	if in.HTTPGet != nil {
		out.HTTPGet = new(HTTPGetAction)
		if err := convert_api_HTTPGetAction_To_v1_HTTPGetAction(in.HTTPGet, out.HTTPGet, s); err != nil {
			return err
		}
	} else {
		out.HTTPGet = nil
	}
	if in.TCPSocket != nil {
		out.TCPSocket = new(TCPSocketAction)
		if err := convert_api_TCPSocketAction_To_v1_TCPSocketAction(in.TCPSocket, out.TCPSocket, s); err != nil {
			return err
		}
	} else {
		out.TCPSocket = nil
	}
	return nil
}

func convert_api_HostPathVolumeSource_To_v1_HostPathVolumeSource(in *api.HostPathVolumeSource, out *HostPathVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.HostPathVolumeSource))(in)
	}
	out.Path = in.Path
	return nil
}

func convert_api_ISCSIVolumeSource_To_v1_ISCSIVolumeSource(in *api.ISCSIVolumeSource, out *ISCSIVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ISCSIVolumeSource))(in)
	}
	out.TargetPortal = in.TargetPortal
	out.IQN = in.IQN
	out.Lun = in.Lun
	out.FSType = in.FSType
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_Lifecycle_To_v1_Lifecycle(in *api.Lifecycle, out *Lifecycle, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Lifecycle))(in)
	}
	if in.PostStart != nil {
		out.PostStart = new(Handler)
		if err := convert_api_Handler_To_v1_Handler(in.PostStart, out.PostStart, s); err != nil {
			return err
		}
	} else {
		out.PostStart = nil
	}
	if in.PreStop != nil {
		out.PreStop = new(Handler)
		if err := convert_api_Handler_To_v1_Handler(in.PreStop, out.PreStop, s); err != nil {
			return err
		}
	} else {
		out.PreStop = nil
	}
	return nil
}

func convert_api_LimitRange_To_v1_LimitRange(in *api.LimitRange, out *LimitRange, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.LimitRange))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_LimitRangeSpec_To_v1_LimitRangeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	return nil
}

func convert_api_LimitRangeItem_To_v1_LimitRangeItem(in *api.LimitRangeItem, out *LimitRangeItem, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.LimitRangeItem))(in)
	}
	out.Type = LimitType(in.Type)
	if in.Max != nil {
		out.Max = make(ResourceList)
		for key, val := range in.Max {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Max[ResourceName(key)] = newVal
		}
	} else {
		out.Max = nil
	}
	if in.Min != nil {
		out.Min = make(ResourceList)
		for key, val := range in.Min {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Min[ResourceName(key)] = newVal
		}
	} else {
		out.Min = nil
	}
	if in.Default != nil {
		out.Default = make(ResourceList)
		for key, val := range in.Default {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Default[ResourceName(key)] = newVal
		}
	} else {
		out.Default = nil
	}
	return nil
}

func convert_api_LimitRangeList_To_v1_LimitRangeList(in *api.LimitRangeList, out *LimitRangeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.LimitRangeList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]LimitRange, len(in.Items))
		for i := range in.Items {
			if err := convert_api_LimitRange_To_v1_LimitRange(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_LimitRangeSpec_To_v1_LimitRangeSpec(in *api.LimitRangeSpec, out *LimitRangeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.LimitRangeSpec))(in)
	}
	if in.Limits != nil {
		out.Limits = make([]LimitRangeItem, len(in.Limits))
		for i := range in.Limits {
			if err := convert_api_LimitRangeItem_To_v1_LimitRangeItem(&in.Limits[i], &out.Limits[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Limits = nil
	}
	return nil
}

func convert_api_List_To_v1_List(in *api.List, out *List, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.List))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ListMeta_To_v1_ListMeta(in *api.ListMeta, out *ListMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ListMeta))(in)
	}
	out.SelfLink = in.SelfLink
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func convert_api_ListOptions_To_v1_ListOptions(in *api.ListOptions, out *ListOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ListOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := s.Convert(&in.LabelSelector, &out.LabelSelector, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.FieldSelector, &out.FieldSelector, 0); err != nil {
		return err
	}
	out.Watch = in.Watch
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func convert_api_LoadBalancerIngress_To_v1_LoadBalancerIngress(in *api.LoadBalancerIngress, out *LoadBalancerIngress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.LoadBalancerIngress))(in)
	}
	out.IP = in.IP
	out.Hostname = in.Hostname
	return nil
}

func convert_api_LoadBalancerStatus_To_v1_LoadBalancerStatus(in *api.LoadBalancerStatus, out *LoadBalancerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.LoadBalancerStatus))(in)
	}
	if in.Ingress != nil {
		out.Ingress = make([]LoadBalancerIngress, len(in.Ingress))
		for i := range in.Ingress {
			if err := convert_api_LoadBalancerIngress_To_v1_LoadBalancerIngress(&in.Ingress[i], &out.Ingress[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Ingress = nil
	}
	return nil
}

func convert_api_LocalObjectReference_To_v1_LocalObjectReference(in *api.LocalObjectReference, out *LocalObjectReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.LocalObjectReference))(in)
	}
	out.Name = in.Name
	return nil
}

func convert_api_NFSVolumeSource_To_v1_NFSVolumeSource(in *api.NFSVolumeSource, out *NFSVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NFSVolumeSource))(in)
	}
	out.Server = in.Server
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_Namespace_To_v1_Namespace(in *api.Namespace, out *Namespace, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Namespace))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_NamespaceSpec_To_v1_NamespaceSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_NamespaceStatus_To_v1_NamespaceStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_NamespaceList_To_v1_NamespaceList(in *api.NamespaceList, out *NamespaceList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NamespaceList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Namespace, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Namespace_To_v1_Namespace(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_NamespaceSpec_To_v1_NamespaceSpec(in *api.NamespaceSpec, out *NamespaceSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NamespaceSpec))(in)
	}
	if in.Finalizers != nil {
		out.Finalizers = make([]FinalizerName, len(in.Finalizers))
		for i := range in.Finalizers {
			out.Finalizers[i] = FinalizerName(in.Finalizers[i])
		}
	} else {
		out.Finalizers = nil
	}
	return nil
}

func convert_api_NamespaceStatus_To_v1_NamespaceStatus(in *api.NamespaceStatus, out *NamespaceStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NamespaceStatus))(in)
	}
	out.Phase = NamespacePhase(in.Phase)
	return nil
}

func convert_api_Node_To_v1_Node(in *api.Node, out *Node, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Node))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_NodeSpec_To_v1_NodeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_NodeStatus_To_v1_NodeStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_NodeAddress_To_v1_NodeAddress(in *api.NodeAddress, out *NodeAddress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NodeAddress))(in)
	}
	out.Type = NodeAddressType(in.Type)
	out.Address = in.Address
	return nil
}

func convert_api_NodeCondition_To_v1_NodeCondition(in *api.NodeCondition, out *NodeCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NodeCondition))(in)
	}
	out.Type = NodeConditionType(in.Type)
	out.Status = ConditionStatus(in.Status)
	if err := s.Convert(&in.LastHeartbeatTime, &out.LastHeartbeatTime, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.LastTransitionTime, &out.LastTransitionTime, 0); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	return nil
}

func convert_api_NodeList_To_v1_NodeList(in *api.NodeList, out *NodeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NodeList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Node, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Node_To_v1_Node(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_NodeSpec_To_v1_NodeSpec(in *api.NodeSpec, out *NodeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NodeSpec))(in)
	}
	out.PodCIDR = in.PodCIDR
	out.ExternalID = in.ExternalID
	out.ProviderID = in.ProviderID
	out.Unschedulable = in.Unschedulable
	return nil
}

func convert_api_NodeStatus_To_v1_NodeStatus(in *api.NodeStatus, out *NodeStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NodeStatus))(in)
	}
	if in.Capacity != nil {
		out.Capacity = make(ResourceList)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[ResourceName(key)] = newVal
		}
	} else {
		out.Capacity = nil
	}
	out.Phase = NodePhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]NodeCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_api_NodeCondition_To_v1_NodeCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	if in.Addresses != nil {
		out.Addresses = make([]NodeAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := convert_api_NodeAddress_To_v1_NodeAddress(&in.Addresses[i], &out.Addresses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if err := convert_api_NodeSystemInfo_To_v1_NodeSystemInfo(&in.NodeInfo, &out.NodeInfo, s); err != nil {
		return err
	}
	return nil
}

func convert_api_NodeSystemInfo_To_v1_NodeSystemInfo(in *api.NodeSystemInfo, out *NodeSystemInfo, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.NodeSystemInfo))(in)
	}
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

func convert_api_ObjectFieldSelector_To_v1_ObjectFieldSelector(in *api.ObjectFieldSelector, out *ObjectFieldSelector, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ObjectFieldSelector))(in)
	}
	out.APIVersion = in.APIVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_api_ObjectMeta_To_v1_ObjectMeta(in *api.ObjectMeta, out *ObjectMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ObjectMeta))(in)
	}
	out.Name = in.Name
	out.GenerateName = in.GenerateName
	out.Namespace = in.Namespace
	out.SelfLink = in.SelfLink
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
	out.Generation = in.Generation
	if err := s.Convert(&in.CreationTimestamp, &out.CreationTimestamp, 0); err != nil {
		return err
	}
	if in.DeletionTimestamp != nil {
		if err := s.Convert(&in.DeletionTimestamp, &out.DeletionTimestamp, 0); err != nil {
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

func convert_api_ObjectReference_To_v1_ObjectReference(in *api.ObjectReference, out *ObjectReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ObjectReference))(in)
	}
	out.Kind = in.Kind
	out.Namespace = in.Namespace
	out.Name = in.Name
	out.UID = in.UID
	out.APIVersion = in.APIVersion
	out.ResourceVersion = in.ResourceVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_api_PersistentVolume_To_v1_PersistentVolume(in *api.PersistentVolume, out *PersistentVolume, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolume))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PersistentVolumeSpec_To_v1_PersistentVolumeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_PersistentVolumeStatus_To_v1_PersistentVolumeStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeClaim_To_v1_PersistentVolumeClaim(in *api.PersistentVolumeClaim, out *PersistentVolumeClaim, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeClaim))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PersistentVolumeClaimSpec_To_v1_PersistentVolumeClaimSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_PersistentVolumeClaimStatus_To_v1_PersistentVolumeClaimStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeClaimList_To_v1_PersistentVolumeClaimList(in *api.PersistentVolumeClaimList, out *PersistentVolumeClaimList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeClaimList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PersistentVolumeClaim, len(in.Items))
		for i := range in.Items {
			if err := convert_api_PersistentVolumeClaim_To_v1_PersistentVolumeClaim(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_PersistentVolumeClaimSpec_To_v1_PersistentVolumeClaimSpec(in *api.PersistentVolumeClaimSpec, out *PersistentVolumeClaimSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeClaimSpec))(in)
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = PersistentVolumeAccessMode(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if err := convert_api_ResourceRequirements_To_v1_ResourceRequirements(&in.Resources, &out.Resources, s); err != nil {
		return err
	}
	out.VolumeName = in.VolumeName
	return nil
}

func convert_api_PersistentVolumeClaimStatus_To_v1_PersistentVolumeClaimStatus(in *api.PersistentVolumeClaimStatus, out *PersistentVolumeClaimStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeClaimStatus))(in)
	}
	out.Phase = PersistentVolumeClaimPhase(in.Phase)
	if in.AccessModes != nil {
		out.AccessModes = make([]PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = PersistentVolumeAccessMode(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if in.Capacity != nil {
		out.Capacity = make(ResourceList)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[ResourceName(key)] = newVal
		}
	} else {
		out.Capacity = nil
	}
	return nil
}

func convert_api_PersistentVolumeClaimVolumeSource_To_v1_PersistentVolumeClaimVolumeSource(in *api.PersistentVolumeClaimVolumeSource, out *PersistentVolumeClaimVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeClaimVolumeSource))(in)
	}
	out.ClaimName = in.ClaimName
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_PersistentVolumeList_To_v1_PersistentVolumeList(in *api.PersistentVolumeList, out *PersistentVolumeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PersistentVolume, len(in.Items))
		for i := range in.Items {
			if err := convert_api_PersistentVolume_To_v1_PersistentVolume(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_PersistentVolumeSource_To_v1_PersistentVolumeSource(in *api.PersistentVolumeSource, out *PersistentVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeSource))(in)
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(GCEPersistentDiskVolumeSource)
		if err := convert_api_GCEPersistentDiskVolumeSource_To_v1_GCEPersistentDiskVolumeSource(in.GCEPersistentDisk, out.GCEPersistentDisk, s); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(AWSElasticBlockStoreVolumeSource)
		if err := convert_api_AWSElasticBlockStoreVolumeSource_To_v1_AWSElasticBlockStoreVolumeSource(in.AWSElasticBlockStore, out.AWSElasticBlockStore, s); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.HostPath != nil {
		out.HostPath = new(HostPathVolumeSource)
		if err := convert_api_HostPathVolumeSource_To_v1_HostPathVolumeSource(in.HostPath, out.HostPath, s); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(GlusterfsVolumeSource)
		if err := convert_api_GlusterfsVolumeSource_To_v1_GlusterfsVolumeSource(in.Glusterfs, out.Glusterfs, s); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.NFS != nil {
		out.NFS = new(NFSVolumeSource)
		if err := convert_api_NFSVolumeSource_To_v1_NFSVolumeSource(in.NFS, out.NFS, s); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	if in.RBD != nil {
		out.RBD = new(RBDVolumeSource)
		if err := convert_api_RBDVolumeSource_To_v1_RBDVolumeSource(in.RBD, out.RBD, s); err != nil {
			return err
		}
	} else {
		out.RBD = nil
	}
	if in.ISCSI != nil {
		out.ISCSI = new(ISCSIVolumeSource)
		if err := convert_api_ISCSIVolumeSource_To_v1_ISCSIVolumeSource(in.ISCSI, out.ISCSI, s); err != nil {
			return err
		}
	} else {
		out.ISCSI = nil
	}
	return nil
}

func convert_api_PersistentVolumeSpec_To_v1_PersistentVolumeSpec(in *api.PersistentVolumeSpec, out *PersistentVolumeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeSpec))(in)
	}
	if in.Capacity != nil {
		out.Capacity = make(ResourceList)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[ResourceName(key)] = newVal
		}
	} else {
		out.Capacity = nil
	}
	if err := convert_api_PersistentVolumeSource_To_v1_PersistentVolumeSource(&in.PersistentVolumeSource, &out.PersistentVolumeSource, s); err != nil {
		return err
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = PersistentVolumeAccessMode(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if in.ClaimRef != nil {
		out.ClaimRef = new(ObjectReference)
		if err := convert_api_ObjectReference_To_v1_ObjectReference(in.ClaimRef, out.ClaimRef, s); err != nil {
			return err
		}
	} else {
		out.ClaimRef = nil
	}
	out.PersistentVolumeReclaimPolicy = PersistentVolumeReclaimPolicy(in.PersistentVolumeReclaimPolicy)
	return nil
}

func convert_api_PersistentVolumeStatus_To_v1_PersistentVolumeStatus(in *api.PersistentVolumeStatus, out *PersistentVolumeStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PersistentVolumeStatus))(in)
	}
	out.Phase = PersistentVolumePhase(in.Phase)
	out.Message = in.Message
	out.Reason = in.Reason
	return nil
}

func convert_api_Pod_To_v1_Pod(in *api.Pod, out *Pod, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Pod))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PodSpec_To_v1_PodSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_PodStatus_To_v1_PodStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PodAttachOptions_To_v1_PodAttachOptions(in *api.PodAttachOptions, out *PodAttachOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodAttachOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Stdin = in.Stdin
	out.Stdout = in.Stdout
	out.Stderr = in.Stderr
	out.TTY = in.TTY
	out.Container = in.Container
	return nil
}

func convert_api_PodCondition_To_v1_PodCondition(in *api.PodCondition, out *PodCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodCondition))(in)
	}
	out.Type = PodConditionType(in.Type)
	out.Status = ConditionStatus(in.Status)
	return nil
}

func convert_api_PodExecOptions_To_v1_PodExecOptions(in *api.PodExecOptions, out *PodExecOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodExecOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
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

func convert_api_PodList_To_v1_PodList(in *api.PodList, out *PodList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Pod, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Pod_To_v1_Pod(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_PodLogOptions_To_v1_PodLogOptions(in *api.PodLogOptions, out *PodLogOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodLogOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Container = in.Container
	out.Follow = in.Follow
	out.Previous = in.Previous
	return nil
}

func convert_api_PodProxyOptions_To_v1_PodProxyOptions(in *api.PodProxyOptions, out *PodProxyOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodProxyOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Path = in.Path
	return nil
}

func convert_api_PodStatus_To_v1_PodStatus(in *api.PodStatus, out *PodStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodStatus))(in)
	}
	out.Phase = PodPhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]PodCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_api_PodCondition_To_v1_PodCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
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
		if err := s.Convert(&in.StartTime, &out.StartTime, 0); err != nil {
			return err
		}
	} else {
		out.StartTime = nil
	}
	if in.ContainerStatuses != nil {
		out.ContainerStatuses = make([]ContainerStatus, len(in.ContainerStatuses))
		for i := range in.ContainerStatuses {
			if err := convert_api_ContainerStatus_To_v1_ContainerStatus(&in.ContainerStatuses[i], &out.ContainerStatuses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.ContainerStatuses = nil
	}
	return nil
}

func convert_api_PodStatusResult_To_v1_PodStatusResult(in *api.PodStatusResult, out *PodStatusResult, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodStatusResult))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PodStatus_To_v1_PodStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PodTemplate_To_v1_PodTemplate(in *api.PodTemplate, out *PodTemplate, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodTemplate))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PodTemplateList_To_v1_PodTemplateList(in *api.PodTemplateList, out *PodTemplateList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodTemplateList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PodTemplate, len(in.Items))
		for i := range in.Items {
			if err := convert_api_PodTemplate_To_v1_PodTemplate(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(in *api.PodTemplateSpec, out *PodTemplateSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.PodTemplateSpec))(in)
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PodSpec_To_v1_PodSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Probe_To_v1_Probe(in *api.Probe, out *Probe, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Probe))(in)
	}
	if err := convert_api_Handler_To_v1_Handler(&in.Handler, &out.Handler, s); err != nil {
		return err
	}
	out.InitialDelaySeconds = in.InitialDelaySeconds
	out.TimeoutSeconds = in.TimeoutSeconds
	return nil
}

func convert_api_RBDVolumeSource_To_v1_RBDVolumeSource(in *api.RBDVolumeSource, out *RBDVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.RBDVolumeSource))(in)
	}
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
		if err := convert_api_LocalObjectReference_To_v1_LocalObjectReference(in.SecretRef, out.SecretRef, s); err != nil {
			return err
		}
	} else {
		out.SecretRef = nil
	}
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_RangeAllocation_To_v1_RangeAllocation(in *api.RangeAllocation, out *RangeAllocation, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.RangeAllocation))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	out.Range = in.Range
	if err := s.Convert(&in.Data, &out.Data, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ReplicationController_To_v1_ReplicationController(in *api.ReplicationController, out *ReplicationController, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ReplicationController))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ReplicationControllerSpec_To_v1_ReplicationControllerSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_ReplicationControllerStatus_To_v1_ReplicationControllerStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_ReplicationControllerList_To_v1_ReplicationControllerList(in *api.ReplicationControllerList, out *ReplicationControllerList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ReplicationControllerList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ReplicationController, len(in.Items))
		for i := range in.Items {
			if err := convert_api_ReplicationController_To_v1_ReplicationController(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ReplicationControllerStatus_To_v1_ReplicationControllerStatus(in *api.ReplicationControllerStatus, out *ReplicationControllerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ReplicationControllerStatus))(in)
	}
	out.Replicas = in.Replicas
	out.ObservedGeneration = in.ObservedGeneration
	return nil
}

func convert_api_ResourceQuota_To_v1_ResourceQuota(in *api.ResourceQuota, out *ResourceQuota, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ResourceQuota))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ResourceQuotaSpec_To_v1_ResourceQuotaSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_ResourceQuotaStatus_To_v1_ResourceQuotaStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_ResourceQuotaList_To_v1_ResourceQuotaList(in *api.ResourceQuotaList, out *ResourceQuotaList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ResourceQuotaList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ResourceQuota, len(in.Items))
		for i := range in.Items {
			if err := convert_api_ResourceQuota_To_v1_ResourceQuota(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ResourceQuotaSpec_To_v1_ResourceQuotaSpec(in *api.ResourceQuotaSpec, out *ResourceQuotaSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ResourceQuotaSpec))(in)
	}
	if in.Hard != nil {
		out.Hard = make(ResourceList)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[ResourceName(key)] = newVal
		}
	} else {
		out.Hard = nil
	}
	return nil
}

func convert_api_ResourceQuotaStatus_To_v1_ResourceQuotaStatus(in *api.ResourceQuotaStatus, out *ResourceQuotaStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ResourceQuotaStatus))(in)
	}
	if in.Hard != nil {
		out.Hard = make(ResourceList)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[ResourceName(key)] = newVal
		}
	} else {
		out.Hard = nil
	}
	if in.Used != nil {
		out.Used = make(ResourceList)
		for key, val := range in.Used {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Used[ResourceName(key)] = newVal
		}
	} else {
		out.Used = nil
	}
	return nil
}

func convert_api_ResourceRequirements_To_v1_ResourceRequirements(in *api.ResourceRequirements, out *ResourceRequirements, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ResourceRequirements))(in)
	}
	if in.Limits != nil {
		out.Limits = make(ResourceList)
		for key, val := range in.Limits {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Limits[ResourceName(key)] = newVal
		}
	} else {
		out.Limits = nil
	}
	if in.Requests != nil {
		out.Requests = make(ResourceList)
		for key, val := range in.Requests {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Requests[ResourceName(key)] = newVal
		}
	} else {
		out.Requests = nil
	}
	return nil
}

func convert_api_SELinuxOptions_To_v1_SELinuxOptions(in *api.SELinuxOptions, out *SELinuxOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.SELinuxOptions))(in)
	}
	out.User = in.User
	out.Role = in.Role
	out.Type = in.Type
	out.Level = in.Level
	return nil
}

func convert_api_Secret_To_v1_Secret(in *api.Secret, out *Secret, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Secret))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Data != nil {
		out.Data = make(map[string][]uint8)
		for key, val := range in.Data {
			newVal := []uint8{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Data[key] = newVal
		}
	} else {
		out.Data = nil
	}
	out.Type = SecretType(in.Type)
	return nil
}

func convert_api_SecretList_To_v1_SecretList(in *api.SecretList, out *SecretList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.SecretList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Secret, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Secret_To_v1_Secret(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_SecretVolumeSource_To_v1_SecretVolumeSource(in *api.SecretVolumeSource, out *SecretVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.SecretVolumeSource))(in)
	}
	out.SecretName = in.SecretName
	return nil
}

func convert_api_SecurityContext_To_v1_SecurityContext(in *api.SecurityContext, out *SecurityContext, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.SecurityContext))(in)
	}
	if in.Capabilities != nil {
		out.Capabilities = new(Capabilities)
		if err := convert_api_Capabilities_To_v1_Capabilities(in.Capabilities, out.Capabilities, s); err != nil {
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
		if err := convert_api_SELinuxOptions_To_v1_SELinuxOptions(in.SELinuxOptions, out.SELinuxOptions, s); err != nil {
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

func convert_api_SerializedReference_To_v1_SerializedReference(in *api.SerializedReference, out *SerializedReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.SerializedReference))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectReference_To_v1_ObjectReference(&in.Reference, &out.Reference, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Service_To_v1_Service(in *api.Service, out *Service, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Service))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ServiceSpec_To_v1_ServiceSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_ServiceStatus_To_v1_ServiceStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_ServiceAccount_To_v1_ServiceAccount(in *api.ServiceAccount, out *ServiceAccount, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ServiceAccount))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Secrets != nil {
		out.Secrets = make([]ObjectReference, len(in.Secrets))
		for i := range in.Secrets {
			if err := convert_api_ObjectReference_To_v1_ObjectReference(&in.Secrets[i], &out.Secrets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Secrets = nil
	}
	if in.ImagePullSecrets != nil {
		out.ImagePullSecrets = make([]LocalObjectReference, len(in.ImagePullSecrets))
		for i := range in.ImagePullSecrets {
			if err := convert_api_LocalObjectReference_To_v1_LocalObjectReference(&in.ImagePullSecrets[i], &out.ImagePullSecrets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.ImagePullSecrets = nil
	}
	return nil
}

func convert_api_ServiceAccountList_To_v1_ServiceAccountList(in *api.ServiceAccountList, out *ServiceAccountList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ServiceAccountList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ServiceAccount, len(in.Items))
		for i := range in.Items {
			if err := convert_api_ServiceAccount_To_v1_ServiceAccount(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ServiceList_To_v1_ServiceList(in *api.ServiceList, out *ServiceList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ServiceList))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Service, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Service_To_v1_Service(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ServicePort_To_v1_ServicePort(in *api.ServicePort, out *ServicePort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ServicePort))(in)
	}
	out.Name = in.Name
	out.Protocol = Protocol(in.Protocol)
	out.Port = in.Port
	if err := s.Convert(&in.TargetPort, &out.TargetPort, 0); err != nil {
		return err
	}
	out.NodePort = in.NodePort
	return nil
}

func convert_api_ServiceSpec_To_v1_ServiceSpec(in *api.ServiceSpec, out *ServiceSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ServiceSpec))(in)
	}
	if in.Ports != nil {
		out.Ports = make([]ServicePort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_api_ServicePort_To_v1_ServicePort(&in.Ports[i], &out.Ports[i], s); err != nil {
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
	out.Type = ServiceType(in.Type)
	if in.DeprecatedPublicIPs != nil {
		out.DeprecatedPublicIPs = make([]string, len(in.DeprecatedPublicIPs))
		for i := range in.DeprecatedPublicIPs {
			out.DeprecatedPublicIPs[i] = in.DeprecatedPublicIPs[i]
		}
	} else {
		out.DeprecatedPublicIPs = nil
	}
	out.SessionAffinity = ServiceAffinity(in.SessionAffinity)
	return nil
}

func convert_api_ServiceStatus_To_v1_ServiceStatus(in *api.ServiceStatus, out *ServiceStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.ServiceStatus))(in)
	}
	if err := convert_api_LoadBalancerStatus_To_v1_LoadBalancerStatus(&in.LoadBalancer, &out.LoadBalancer, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Status_To_v1_Status(in *api.Status, out *Status, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Status))(in)
	}
	if err := convert_api_TypeMeta_To_v1_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	out.Status = in.Status
	out.Message = in.Message
	out.Reason = StatusReason(in.Reason)
	if in.Details != nil {
		out.Details = new(StatusDetails)
		if err := convert_api_StatusDetails_To_v1_StatusDetails(in.Details, out.Details, s); err != nil {
			return err
		}
	} else {
		out.Details = nil
	}
	out.Code = in.Code
	return nil
}

func convert_api_StatusCause_To_v1_StatusCause(in *api.StatusCause, out *StatusCause, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.StatusCause))(in)
	}
	out.Type = CauseType(in.Type)
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func convert_api_StatusDetails_To_v1_StatusDetails(in *api.StatusDetails, out *StatusDetails, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.StatusDetails))(in)
	}
	out.Name = in.Name
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := convert_api_StatusCause_To_v1_StatusCause(&in.Causes[i], &out.Causes[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Causes = nil
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func convert_api_TCPSocketAction_To_v1_TCPSocketAction(in *api.TCPSocketAction, out *TCPSocketAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.TCPSocketAction))(in)
	}
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_TypeMeta_To_v1_TypeMeta(in *api.TypeMeta, out *TypeMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.TypeMeta))(in)
	}
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	return nil
}

func convert_api_Volume_To_v1_Volume(in *api.Volume, out *Volume, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.Volume))(in)
	}
	out.Name = in.Name
	if err := convert_api_VolumeSource_To_v1_VolumeSource(&in.VolumeSource, &out.VolumeSource, s); err != nil {
		return err
	}
	return nil
}

func convert_api_VolumeMount_To_v1_VolumeMount(in *api.VolumeMount, out *VolumeMount, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.VolumeMount))(in)
	}
	out.Name = in.Name
	out.ReadOnly = in.ReadOnly
	out.MountPath = in.MountPath
	return nil
}

func convert_api_VolumeSource_To_v1_VolumeSource(in *api.VolumeSource, out *VolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*api.VolumeSource))(in)
	}
	if in.HostPath != nil {
		out.HostPath = new(HostPathVolumeSource)
		if err := convert_api_HostPathVolumeSource_To_v1_HostPathVolumeSource(in.HostPath, out.HostPath, s); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.EmptyDir != nil {
		out.EmptyDir = new(EmptyDirVolumeSource)
		if err := convert_api_EmptyDirVolumeSource_To_v1_EmptyDirVolumeSource(in.EmptyDir, out.EmptyDir, s); err != nil {
			return err
		}
	} else {
		out.EmptyDir = nil
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(GCEPersistentDiskVolumeSource)
		if err := convert_api_GCEPersistentDiskVolumeSource_To_v1_GCEPersistentDiskVolumeSource(in.GCEPersistentDisk, out.GCEPersistentDisk, s); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(AWSElasticBlockStoreVolumeSource)
		if err := convert_api_AWSElasticBlockStoreVolumeSource_To_v1_AWSElasticBlockStoreVolumeSource(in.AWSElasticBlockStore, out.AWSElasticBlockStore, s); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.GitRepo != nil {
		out.GitRepo = new(GitRepoVolumeSource)
		if err := convert_api_GitRepoVolumeSource_To_v1_GitRepoVolumeSource(in.GitRepo, out.GitRepo, s); err != nil {
			return err
		}
	} else {
		out.GitRepo = nil
	}
	if in.Secret != nil {
		out.Secret = new(SecretVolumeSource)
		if err := convert_api_SecretVolumeSource_To_v1_SecretVolumeSource(in.Secret, out.Secret, s); err != nil {
			return err
		}
	} else {
		out.Secret = nil
	}
	if in.NFS != nil {
		out.NFS = new(NFSVolumeSource)
		if err := convert_api_NFSVolumeSource_To_v1_NFSVolumeSource(in.NFS, out.NFS, s); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	if in.ISCSI != nil {
		out.ISCSI = new(ISCSIVolumeSource)
		if err := convert_api_ISCSIVolumeSource_To_v1_ISCSIVolumeSource(in.ISCSI, out.ISCSI, s); err != nil {
			return err
		}
	} else {
		out.ISCSI = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(GlusterfsVolumeSource)
		if err := convert_api_GlusterfsVolumeSource_To_v1_GlusterfsVolumeSource(in.Glusterfs, out.Glusterfs, s); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.PersistentVolumeClaim != nil {
		out.PersistentVolumeClaim = new(PersistentVolumeClaimVolumeSource)
		if err := convert_api_PersistentVolumeClaimVolumeSource_To_v1_PersistentVolumeClaimVolumeSource(in.PersistentVolumeClaim, out.PersistentVolumeClaim, s); err != nil {
			return err
		}
	} else {
		out.PersistentVolumeClaim = nil
	}
	if in.RBD != nil {
		out.RBD = new(RBDVolumeSource)
		if err := convert_api_RBDVolumeSource_To_v1_RBDVolumeSource(in.RBD, out.RBD, s); err != nil {
			return err
		}
	} else {
		out.RBD = nil
	}
	return nil
}

func convert_v1_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource(in *AWSElasticBlockStoreVolumeSource, out *api.AWSElasticBlockStoreVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*AWSElasticBlockStoreVolumeSource))(in)
	}
	out.VolumeID = in.VolumeID
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1_Binding_To_api_Binding(in *Binding, out *api.Binding, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Binding))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectReference_To_api_ObjectReference(&in.Target, &out.Target, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_Capabilities_To_api_Capabilities(in *Capabilities, out *api.Capabilities, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Capabilities))(in)
	}
	if in.Add != nil {
		out.Add = make([]api.Capability, len(in.Add))
		for i := range in.Add {
			out.Add[i] = api.Capability(in.Add[i])
		}
	} else {
		out.Add = nil
	}
	if in.Drop != nil {
		out.Drop = make([]api.Capability, len(in.Drop))
		for i := range in.Drop {
			out.Drop[i] = api.Capability(in.Drop[i])
		}
	} else {
		out.Drop = nil
	}
	return nil
}

func convert_v1_ComponentCondition_To_api_ComponentCondition(in *ComponentCondition, out *api.ComponentCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ComponentCondition))(in)
	}
	out.Type = api.ComponentConditionType(in.Type)
	out.Status = api.ConditionStatus(in.Status)
	out.Message = in.Message
	out.Error = in.Error
	return nil
}

func convert_v1_ComponentStatus_To_api_ComponentStatus(in *ComponentStatus, out *api.ComponentStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ComponentStatus))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Conditions != nil {
		out.Conditions = make([]api.ComponentCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_v1_ComponentCondition_To_api_ComponentCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	return nil
}

func convert_v1_ComponentStatusList_To_api_ComponentStatusList(in *ComponentStatusList, out *api.ComponentStatusList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ComponentStatusList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.ComponentStatus, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_ComponentStatus_To_api_ComponentStatus(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_Container_To_api_Container(in *Container, out *api.Container, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Container))(in)
	}
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
		out.Ports = make([]api.ContainerPort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_v1_ContainerPort_To_api_ContainerPort(&in.Ports[i], &out.Ports[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	if in.Env != nil {
		out.Env = make([]api.EnvVar, len(in.Env))
		for i := range in.Env {
			if err := convert_v1_EnvVar_To_api_EnvVar(&in.Env[i], &out.Env[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Env = nil
	}
	if err := convert_v1_ResourceRequirements_To_api_ResourceRequirements(&in.Resources, &out.Resources, s); err != nil {
		return err
	}
	if in.VolumeMounts != nil {
		out.VolumeMounts = make([]api.VolumeMount, len(in.VolumeMounts))
		for i := range in.VolumeMounts {
			if err := convert_v1_VolumeMount_To_api_VolumeMount(&in.VolumeMounts[i], &out.VolumeMounts[i], s); err != nil {
				return err
			}
		}
	} else {
		out.VolumeMounts = nil
	}
	if in.LivenessProbe != nil {
		out.LivenessProbe = new(api.Probe)
		if err := convert_v1_Probe_To_api_Probe(in.LivenessProbe, out.LivenessProbe, s); err != nil {
			return err
		}
	} else {
		out.LivenessProbe = nil
	}
	if in.ReadinessProbe != nil {
		out.ReadinessProbe = new(api.Probe)
		if err := convert_v1_Probe_To_api_Probe(in.ReadinessProbe, out.ReadinessProbe, s); err != nil {
			return err
		}
	} else {
		out.ReadinessProbe = nil
	}
	if in.Lifecycle != nil {
		out.Lifecycle = new(api.Lifecycle)
		if err := convert_v1_Lifecycle_To_api_Lifecycle(in.Lifecycle, out.Lifecycle, s); err != nil {
			return err
		}
	} else {
		out.Lifecycle = nil
	}
	out.TerminationMessagePath = in.TerminationMessagePath
	out.ImagePullPolicy = api.PullPolicy(in.ImagePullPolicy)
	if in.SecurityContext != nil {
		out.SecurityContext = new(api.SecurityContext)
		if err := convert_v1_SecurityContext_To_api_SecurityContext(in.SecurityContext, out.SecurityContext, s); err != nil {
			return err
		}
	} else {
		out.SecurityContext = nil
	}
	return nil
}

func convert_v1_ContainerPort_To_api_ContainerPort(in *ContainerPort, out *api.ContainerPort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerPort))(in)
	}
	out.Name = in.Name
	out.HostPort = in.HostPort
	out.ContainerPort = in.ContainerPort
	out.Protocol = api.Protocol(in.Protocol)
	out.HostIP = in.HostIP
	return nil
}

func convert_v1_ContainerState_To_api_ContainerState(in *ContainerState, out *api.ContainerState, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerState))(in)
	}
	if in.Waiting != nil {
		out.Waiting = new(api.ContainerStateWaiting)
		if err := convert_v1_ContainerStateWaiting_To_api_ContainerStateWaiting(in.Waiting, out.Waiting, s); err != nil {
			return err
		}
	} else {
		out.Waiting = nil
	}
	if in.Running != nil {
		out.Running = new(api.ContainerStateRunning)
		if err := convert_v1_ContainerStateRunning_To_api_ContainerStateRunning(in.Running, out.Running, s); err != nil {
			return err
		}
	} else {
		out.Running = nil
	}
	if in.Terminated != nil {
		out.Terminated = new(api.ContainerStateTerminated)
		if err := convert_v1_ContainerStateTerminated_To_api_ContainerStateTerminated(in.Terminated, out.Terminated, s); err != nil {
			return err
		}
	} else {
		out.Terminated = nil
	}
	return nil
}

func convert_v1_ContainerStateRunning_To_api_ContainerStateRunning(in *ContainerStateRunning, out *api.ContainerStateRunning, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerStateRunning))(in)
	}
	if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1_ContainerStateTerminated_To_api_ContainerStateTerminated(in *ContainerStateTerminated, out *api.ContainerStateTerminated, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerStateTerminated))(in)
	}
	out.ExitCode = in.ExitCode
	out.Signal = in.Signal
	out.Reason = in.Reason
	out.Message = in.Message
	if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.FinishedAt, &out.FinishedAt, 0); err != nil {
		return err
	}
	out.ContainerID = in.ContainerID
	return nil
}

func convert_v1_ContainerStateWaiting_To_api_ContainerStateWaiting(in *ContainerStateWaiting, out *api.ContainerStateWaiting, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerStateWaiting))(in)
	}
	out.Reason = in.Reason
	return nil
}

func convert_v1_ContainerStatus_To_api_ContainerStatus(in *ContainerStatus, out *api.ContainerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerStatus))(in)
	}
	out.Name = in.Name
	if err := convert_v1_ContainerState_To_api_ContainerState(&in.State, &out.State, s); err != nil {
		return err
	}
	if err := convert_v1_ContainerState_To_api_ContainerState(&in.LastTerminationState, &out.LastTerminationState, s); err != nil {
		return err
	}
	out.Ready = in.Ready
	out.RestartCount = in.RestartCount
	out.Image = in.Image
	out.ImageID = in.ImageID
	out.ContainerID = in.ContainerID
	return nil
}

func convert_v1_DeleteOptions_To_api_DeleteOptions(in *DeleteOptions, out *api.DeleteOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*DeleteOptions))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
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

func convert_v1_EmptyDirVolumeSource_To_api_EmptyDirVolumeSource(in *EmptyDirVolumeSource, out *api.EmptyDirVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EmptyDirVolumeSource))(in)
	}
	out.Medium = api.StorageMedium(in.Medium)
	return nil
}

func convert_v1_EndpointAddress_To_api_EndpointAddress(in *EndpointAddress, out *api.EndpointAddress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EndpointAddress))(in)
	}
	out.IP = in.IP
	if in.TargetRef != nil {
		out.TargetRef = new(api.ObjectReference)
		if err := convert_v1_ObjectReference_To_api_ObjectReference(in.TargetRef, out.TargetRef, s); err != nil {
			return err
		}
	} else {
		out.TargetRef = nil
	}
	return nil
}

func convert_v1_EndpointPort_To_api_EndpointPort(in *EndpointPort, out *api.EndpointPort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EndpointPort))(in)
	}
	out.Name = in.Name
	out.Port = in.Port
	out.Protocol = api.Protocol(in.Protocol)
	return nil
}

func convert_v1_EndpointSubset_To_api_EndpointSubset(in *EndpointSubset, out *api.EndpointSubset, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EndpointSubset))(in)
	}
	if in.Addresses != nil {
		out.Addresses = make([]api.EndpointAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := convert_v1_EndpointAddress_To_api_EndpointAddress(&in.Addresses[i], &out.Addresses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if in.Ports != nil {
		out.Ports = make([]api.EndpointPort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_v1_EndpointPort_To_api_EndpointPort(&in.Ports[i], &out.Ports[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	return nil
}

func convert_v1_Endpoints_To_api_Endpoints(in *Endpoints, out *api.Endpoints, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Endpoints))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Subsets != nil {
		out.Subsets = make([]api.EndpointSubset, len(in.Subsets))
		for i := range in.Subsets {
			if err := convert_v1_EndpointSubset_To_api_EndpointSubset(&in.Subsets[i], &out.Subsets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Subsets = nil
	}
	return nil
}

func convert_v1_EndpointsList_To_api_EndpointsList(in *EndpointsList, out *api.EndpointsList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EndpointsList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.Endpoints, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_Endpoints_To_api_Endpoints(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_EnvVar_To_api_EnvVar(in *EnvVar, out *api.EnvVar, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EnvVar))(in)
	}
	out.Name = in.Name
	out.Value = in.Value
	if in.ValueFrom != nil {
		out.ValueFrom = new(api.EnvVarSource)
		if err := convert_v1_EnvVarSource_To_api_EnvVarSource(in.ValueFrom, out.ValueFrom, s); err != nil {
			return err
		}
	} else {
		out.ValueFrom = nil
	}
	return nil
}

func convert_v1_EnvVarSource_To_api_EnvVarSource(in *EnvVarSource, out *api.EnvVarSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EnvVarSource))(in)
	}
	if in.FieldRef != nil {
		out.FieldRef = new(api.ObjectFieldSelector)
		if err := convert_v1_ObjectFieldSelector_To_api_ObjectFieldSelector(in.FieldRef, out.FieldRef, s); err != nil {
			return err
		}
	} else {
		out.FieldRef = nil
	}
	return nil
}

func convert_v1_Event_To_api_Event(in *Event, out *api.Event, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Event))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectReference_To_api_ObjectReference(&in.InvolvedObject, &out.InvolvedObject, s); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	if err := convert_v1_EventSource_To_api_EventSource(&in.Source, &out.Source, s); err != nil {
		return err
	}
	if err := s.Convert(&in.FirstTimestamp, &out.FirstTimestamp, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.LastTimestamp, &out.LastTimestamp, 0); err != nil {
		return err
	}
	out.Count = in.Count
	return nil
}

func convert_v1_EventList_To_api_EventList(in *EventList, out *api.EventList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EventList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.Event, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_Event_To_api_Event(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_EventSource_To_api_EventSource(in *EventSource, out *api.EventSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EventSource))(in)
	}
	out.Component = in.Component
	out.Host = in.Host
	return nil
}

func convert_v1_ExecAction_To_api_ExecAction(in *ExecAction, out *api.ExecAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ExecAction))(in)
	}
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

func convert_v1_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource(in *GCEPersistentDiskVolumeSource, out *api.GCEPersistentDiskVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*GCEPersistentDiskVolumeSource))(in)
	}
	out.PDName = in.PDName
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1_GitRepoVolumeSource_To_api_GitRepoVolumeSource(in *GitRepoVolumeSource, out *api.GitRepoVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*GitRepoVolumeSource))(in)
	}
	out.Repository = in.Repository
	out.Revision = in.Revision
	return nil
}

func convert_v1_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource(in *GlusterfsVolumeSource, out *api.GlusterfsVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*GlusterfsVolumeSource))(in)
	}
	out.EndpointsName = in.EndpointsName
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1_HTTPGetAction_To_api_HTTPGetAction(in *HTTPGetAction, out *api.HTTPGetAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*HTTPGetAction))(in)
	}
	out.Path = in.Path
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	out.Host = in.Host
	out.Scheme = api.URIScheme(in.Scheme)
	return nil
}

func convert_v1_Handler_To_api_Handler(in *Handler, out *api.Handler, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Handler))(in)
	}
	if in.Exec != nil {
		out.Exec = new(api.ExecAction)
		if err := convert_v1_ExecAction_To_api_ExecAction(in.Exec, out.Exec, s); err != nil {
			return err
		}
	} else {
		out.Exec = nil
	}
	if in.HTTPGet != nil {
		out.HTTPGet = new(api.HTTPGetAction)
		if err := convert_v1_HTTPGetAction_To_api_HTTPGetAction(in.HTTPGet, out.HTTPGet, s); err != nil {
			return err
		}
	} else {
		out.HTTPGet = nil
	}
	if in.TCPSocket != nil {
		out.TCPSocket = new(api.TCPSocketAction)
		if err := convert_v1_TCPSocketAction_To_api_TCPSocketAction(in.TCPSocket, out.TCPSocket, s); err != nil {
			return err
		}
	} else {
		out.TCPSocket = nil
	}
	return nil
}

func convert_v1_HostPathVolumeSource_To_api_HostPathVolumeSource(in *HostPathVolumeSource, out *api.HostPathVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*HostPathVolumeSource))(in)
	}
	out.Path = in.Path
	return nil
}

func convert_v1_ISCSIVolumeSource_To_api_ISCSIVolumeSource(in *ISCSIVolumeSource, out *api.ISCSIVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ISCSIVolumeSource))(in)
	}
	out.TargetPortal = in.TargetPortal
	out.IQN = in.IQN
	out.Lun = in.Lun
	out.FSType = in.FSType
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1_Lifecycle_To_api_Lifecycle(in *Lifecycle, out *api.Lifecycle, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Lifecycle))(in)
	}
	if in.PostStart != nil {
		out.PostStart = new(api.Handler)
		if err := convert_v1_Handler_To_api_Handler(in.PostStart, out.PostStart, s); err != nil {
			return err
		}
	} else {
		out.PostStart = nil
	}
	if in.PreStop != nil {
		out.PreStop = new(api.Handler)
		if err := convert_v1_Handler_To_api_Handler(in.PreStop, out.PreStop, s); err != nil {
			return err
		}
	} else {
		out.PreStop = nil
	}
	return nil
}

func convert_v1_LimitRange_To_api_LimitRange(in *LimitRange, out *api.LimitRange, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LimitRange))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_LimitRangeSpec_To_api_LimitRangeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_LimitRangeItem_To_api_LimitRangeItem(in *LimitRangeItem, out *api.LimitRangeItem, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LimitRangeItem))(in)
	}
	out.Type = api.LimitType(in.Type)
	if in.Max != nil {
		out.Max = make(api.ResourceList)
		for key, val := range in.Max {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Max[api.ResourceName(key)] = newVal
		}
	} else {
		out.Max = nil
	}
	if in.Min != nil {
		out.Min = make(api.ResourceList)
		for key, val := range in.Min {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Min[api.ResourceName(key)] = newVal
		}
	} else {
		out.Min = nil
	}
	if in.Default != nil {
		out.Default = make(api.ResourceList)
		for key, val := range in.Default {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Default[api.ResourceName(key)] = newVal
		}
	} else {
		out.Default = nil
	}
	return nil
}

func convert_v1_LimitRangeList_To_api_LimitRangeList(in *LimitRangeList, out *api.LimitRangeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LimitRangeList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.LimitRange, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_LimitRange_To_api_LimitRange(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_LimitRangeSpec_To_api_LimitRangeSpec(in *LimitRangeSpec, out *api.LimitRangeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LimitRangeSpec))(in)
	}
	if in.Limits != nil {
		out.Limits = make([]api.LimitRangeItem, len(in.Limits))
		for i := range in.Limits {
			if err := convert_v1_LimitRangeItem_To_api_LimitRangeItem(&in.Limits[i], &out.Limits[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Limits = nil
	}
	return nil
}

func convert_v1_List_To_api_List(in *List, out *api.List, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*List))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1_ListMeta_To_api_ListMeta(in *ListMeta, out *api.ListMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ListMeta))(in)
	}
	out.SelfLink = in.SelfLink
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func convert_v1_ListOptions_To_api_ListOptions(in *ListOptions, out *api.ListOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ListOptions))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := s.Convert(&in.LabelSelector, &out.LabelSelector, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.FieldSelector, &out.FieldSelector, 0); err != nil {
		return err
	}
	out.Watch = in.Watch
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func convert_v1_LoadBalancerIngress_To_api_LoadBalancerIngress(in *LoadBalancerIngress, out *api.LoadBalancerIngress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LoadBalancerIngress))(in)
	}
	out.IP = in.IP
	out.Hostname = in.Hostname
	return nil
}

func convert_v1_LoadBalancerStatus_To_api_LoadBalancerStatus(in *LoadBalancerStatus, out *api.LoadBalancerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LoadBalancerStatus))(in)
	}
	if in.Ingress != nil {
		out.Ingress = make([]api.LoadBalancerIngress, len(in.Ingress))
		for i := range in.Ingress {
			if err := convert_v1_LoadBalancerIngress_To_api_LoadBalancerIngress(&in.Ingress[i], &out.Ingress[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Ingress = nil
	}
	return nil
}

func convert_v1_LocalObjectReference_To_api_LocalObjectReference(in *LocalObjectReference, out *api.LocalObjectReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LocalObjectReference))(in)
	}
	out.Name = in.Name
	return nil
}

func convert_v1_NFSVolumeSource_To_api_NFSVolumeSource(in *NFSVolumeSource, out *api.NFSVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NFSVolumeSource))(in)
	}
	out.Server = in.Server
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1_Namespace_To_api_Namespace(in *Namespace, out *api.Namespace, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Namespace))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_NamespaceSpec_To_api_NamespaceSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1_NamespaceStatus_To_api_NamespaceStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_NamespaceList_To_api_NamespaceList(in *NamespaceList, out *api.NamespaceList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NamespaceList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.Namespace, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_Namespace_To_api_Namespace(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_NamespaceSpec_To_api_NamespaceSpec(in *NamespaceSpec, out *api.NamespaceSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NamespaceSpec))(in)
	}
	if in.Finalizers != nil {
		out.Finalizers = make([]api.FinalizerName, len(in.Finalizers))
		for i := range in.Finalizers {
			out.Finalizers[i] = api.FinalizerName(in.Finalizers[i])
		}
	} else {
		out.Finalizers = nil
	}
	return nil
}

func convert_v1_NamespaceStatus_To_api_NamespaceStatus(in *NamespaceStatus, out *api.NamespaceStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NamespaceStatus))(in)
	}
	out.Phase = api.NamespacePhase(in.Phase)
	return nil
}

func convert_v1_Node_To_api_Node(in *Node, out *api.Node, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Node))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_NodeSpec_To_api_NodeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1_NodeStatus_To_api_NodeStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_NodeAddress_To_api_NodeAddress(in *NodeAddress, out *api.NodeAddress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeAddress))(in)
	}
	out.Type = api.NodeAddressType(in.Type)
	out.Address = in.Address
	return nil
}

func convert_v1_NodeCondition_To_api_NodeCondition(in *NodeCondition, out *api.NodeCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeCondition))(in)
	}
	out.Type = api.NodeConditionType(in.Type)
	out.Status = api.ConditionStatus(in.Status)
	if err := s.Convert(&in.LastHeartbeatTime, &out.LastHeartbeatTime, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.LastTransitionTime, &out.LastTransitionTime, 0); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	return nil
}

func convert_v1_NodeList_To_api_NodeList(in *NodeList, out *api.NodeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.Node, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_Node_To_api_Node(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_NodeSpec_To_api_NodeSpec(in *NodeSpec, out *api.NodeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeSpec))(in)
	}
	out.PodCIDR = in.PodCIDR
	out.ExternalID = in.ExternalID
	out.ProviderID = in.ProviderID
	out.Unschedulable = in.Unschedulable
	return nil
}

func convert_v1_NodeStatus_To_api_NodeStatus(in *NodeStatus, out *api.NodeStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeStatus))(in)
	}
	if in.Capacity != nil {
		out.Capacity = make(api.ResourceList)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[api.ResourceName(key)] = newVal
		}
	} else {
		out.Capacity = nil
	}
	out.Phase = api.NodePhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]api.NodeCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_v1_NodeCondition_To_api_NodeCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	if in.Addresses != nil {
		out.Addresses = make([]api.NodeAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := convert_v1_NodeAddress_To_api_NodeAddress(&in.Addresses[i], &out.Addresses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if err := convert_v1_NodeSystemInfo_To_api_NodeSystemInfo(&in.NodeInfo, &out.NodeInfo, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_NodeSystemInfo_To_api_NodeSystemInfo(in *NodeSystemInfo, out *api.NodeSystemInfo, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeSystemInfo))(in)
	}
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

func convert_v1_ObjectFieldSelector_To_api_ObjectFieldSelector(in *ObjectFieldSelector, out *api.ObjectFieldSelector, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ObjectFieldSelector))(in)
	}
	out.APIVersion = in.APIVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_v1_ObjectMeta_To_api_ObjectMeta(in *ObjectMeta, out *api.ObjectMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ObjectMeta))(in)
	}
	out.Name = in.Name
	out.GenerateName = in.GenerateName
	out.Namespace = in.Namespace
	out.SelfLink = in.SelfLink
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
	out.Generation = in.Generation
	if err := s.Convert(&in.CreationTimestamp, &out.CreationTimestamp, 0); err != nil {
		return err
	}
	if in.DeletionTimestamp != nil {
		if err := s.Convert(&in.DeletionTimestamp, &out.DeletionTimestamp, 0); err != nil {
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

func convert_v1_ObjectReference_To_api_ObjectReference(in *ObjectReference, out *api.ObjectReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ObjectReference))(in)
	}
	out.Kind = in.Kind
	out.Namespace = in.Namespace
	out.Name = in.Name
	out.UID = in.UID
	out.APIVersion = in.APIVersion
	out.ResourceVersion = in.ResourceVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_v1_PersistentVolume_To_api_PersistentVolume(in *PersistentVolume, out *api.PersistentVolume, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolume))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_PersistentVolumeSpec_To_api_PersistentVolumeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1_PersistentVolumeStatus_To_api_PersistentVolumeStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_PersistentVolumeClaim_To_api_PersistentVolumeClaim(in *PersistentVolumeClaim, out *api.PersistentVolumeClaim, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaim))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_PersistentVolumeClaimSpec_To_api_PersistentVolumeClaimSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1_PersistentVolumeClaimStatus_To_api_PersistentVolumeClaimStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_PersistentVolumeClaimList_To_api_PersistentVolumeClaimList(in *PersistentVolumeClaimList, out *api.PersistentVolumeClaimList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaimList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.PersistentVolumeClaim, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_PersistentVolumeClaim_To_api_PersistentVolumeClaim(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_PersistentVolumeClaimSpec_To_api_PersistentVolumeClaimSpec(in *PersistentVolumeClaimSpec, out *api.PersistentVolumeClaimSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaimSpec))(in)
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]api.PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = api.PersistentVolumeAccessMode(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if err := convert_v1_ResourceRequirements_To_api_ResourceRequirements(&in.Resources, &out.Resources, s); err != nil {
		return err
	}
	out.VolumeName = in.VolumeName
	return nil
}

func convert_v1_PersistentVolumeClaimStatus_To_api_PersistentVolumeClaimStatus(in *PersistentVolumeClaimStatus, out *api.PersistentVolumeClaimStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaimStatus))(in)
	}
	out.Phase = api.PersistentVolumeClaimPhase(in.Phase)
	if in.AccessModes != nil {
		out.AccessModes = make([]api.PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = api.PersistentVolumeAccessMode(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if in.Capacity != nil {
		out.Capacity = make(api.ResourceList)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[api.ResourceName(key)] = newVal
		}
	} else {
		out.Capacity = nil
	}
	return nil
}

func convert_v1_PersistentVolumeClaimVolumeSource_To_api_PersistentVolumeClaimVolumeSource(in *PersistentVolumeClaimVolumeSource, out *api.PersistentVolumeClaimVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaimVolumeSource))(in)
	}
	out.ClaimName = in.ClaimName
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1_PersistentVolumeList_To_api_PersistentVolumeList(in *PersistentVolumeList, out *api.PersistentVolumeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.PersistentVolume, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_PersistentVolume_To_api_PersistentVolume(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_PersistentVolumeSource_To_api_PersistentVolumeSource(in *PersistentVolumeSource, out *api.PersistentVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeSource))(in)
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(api.GCEPersistentDiskVolumeSource)
		if err := convert_v1_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource(in.GCEPersistentDisk, out.GCEPersistentDisk, s); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(api.AWSElasticBlockStoreVolumeSource)
		if err := convert_v1_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource(in.AWSElasticBlockStore, out.AWSElasticBlockStore, s); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.HostPath != nil {
		out.HostPath = new(api.HostPathVolumeSource)
		if err := convert_v1_HostPathVolumeSource_To_api_HostPathVolumeSource(in.HostPath, out.HostPath, s); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(api.GlusterfsVolumeSource)
		if err := convert_v1_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource(in.Glusterfs, out.Glusterfs, s); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.NFS != nil {
		out.NFS = new(api.NFSVolumeSource)
		if err := convert_v1_NFSVolumeSource_To_api_NFSVolumeSource(in.NFS, out.NFS, s); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	if in.RBD != nil {
		out.RBD = new(api.RBDVolumeSource)
		if err := convert_v1_RBDVolumeSource_To_api_RBDVolumeSource(in.RBD, out.RBD, s); err != nil {
			return err
		}
	} else {
		out.RBD = nil
	}
	if in.ISCSI != nil {
		out.ISCSI = new(api.ISCSIVolumeSource)
		if err := convert_v1_ISCSIVolumeSource_To_api_ISCSIVolumeSource(in.ISCSI, out.ISCSI, s); err != nil {
			return err
		}
	} else {
		out.ISCSI = nil
	}
	return nil
}

func convert_v1_PersistentVolumeSpec_To_api_PersistentVolumeSpec(in *PersistentVolumeSpec, out *api.PersistentVolumeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeSpec))(in)
	}
	if in.Capacity != nil {
		out.Capacity = make(api.ResourceList)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[api.ResourceName(key)] = newVal
		}
	} else {
		out.Capacity = nil
	}
	if err := convert_v1_PersistentVolumeSource_To_api_PersistentVolumeSource(&in.PersistentVolumeSource, &out.PersistentVolumeSource, s); err != nil {
		return err
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]api.PersistentVolumeAccessMode, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = api.PersistentVolumeAccessMode(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if in.ClaimRef != nil {
		out.ClaimRef = new(api.ObjectReference)
		if err := convert_v1_ObjectReference_To_api_ObjectReference(in.ClaimRef, out.ClaimRef, s); err != nil {
			return err
		}
	} else {
		out.ClaimRef = nil
	}
	out.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimPolicy(in.PersistentVolumeReclaimPolicy)
	return nil
}

func convert_v1_PersistentVolumeStatus_To_api_PersistentVolumeStatus(in *PersistentVolumeStatus, out *api.PersistentVolumeStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeStatus))(in)
	}
	out.Phase = api.PersistentVolumePhase(in.Phase)
	out.Message = in.Message
	out.Reason = in.Reason
	return nil
}

func convert_v1_Pod_To_api_Pod(in *Pod, out *api.Pod, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Pod))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_PodSpec_To_api_PodSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1_PodStatus_To_api_PodStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_PodAttachOptions_To_api_PodAttachOptions(in *PodAttachOptions, out *api.PodAttachOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodAttachOptions))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Stdin = in.Stdin
	out.Stdout = in.Stdout
	out.Stderr = in.Stderr
	out.TTY = in.TTY
	out.Container = in.Container
	return nil
}

func convert_v1_PodCondition_To_api_PodCondition(in *PodCondition, out *api.PodCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodCondition))(in)
	}
	out.Type = api.PodConditionType(in.Type)
	out.Status = api.ConditionStatus(in.Status)
	return nil
}

func convert_v1_PodExecOptions_To_api_PodExecOptions(in *PodExecOptions, out *api.PodExecOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodExecOptions))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
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

func convert_v1_PodList_To_api_PodList(in *PodList, out *api.PodList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.Pod, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_Pod_To_api_Pod(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_PodLogOptions_To_api_PodLogOptions(in *PodLogOptions, out *api.PodLogOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodLogOptions))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Container = in.Container
	out.Follow = in.Follow
	out.Previous = in.Previous
	return nil
}

func convert_v1_PodProxyOptions_To_api_PodProxyOptions(in *PodProxyOptions, out *api.PodProxyOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodProxyOptions))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Path = in.Path
	return nil
}

func convert_v1_PodStatus_To_api_PodStatus(in *PodStatus, out *api.PodStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodStatus))(in)
	}
	out.Phase = api.PodPhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]api.PodCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_v1_PodCondition_To_api_PodCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
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
		if err := s.Convert(&in.StartTime, &out.StartTime, 0); err != nil {
			return err
		}
	} else {
		out.StartTime = nil
	}
	if in.ContainerStatuses != nil {
		out.ContainerStatuses = make([]api.ContainerStatus, len(in.ContainerStatuses))
		for i := range in.ContainerStatuses {
			if err := convert_v1_ContainerStatus_To_api_ContainerStatus(&in.ContainerStatuses[i], &out.ContainerStatuses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.ContainerStatuses = nil
	}
	return nil
}

func convert_v1_PodStatusResult_To_api_PodStatusResult(in *PodStatusResult, out *api.PodStatusResult, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodStatusResult))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_PodStatus_To_api_PodStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_PodTemplate_To_api_PodTemplate(in *PodTemplate, out *api.PodTemplate, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodTemplate))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_PodTemplateList_To_api_PodTemplateList(in *PodTemplateList, out *api.PodTemplateList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodTemplateList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.PodTemplate, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_PodTemplate_To_api_PodTemplate(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(in *PodTemplateSpec, out *api.PodTemplateSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodTemplateSpec))(in)
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_PodSpec_To_api_PodSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_Probe_To_api_Probe(in *Probe, out *api.Probe, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Probe))(in)
	}
	if err := convert_v1_Handler_To_api_Handler(&in.Handler, &out.Handler, s); err != nil {
		return err
	}
	out.InitialDelaySeconds = in.InitialDelaySeconds
	out.TimeoutSeconds = in.TimeoutSeconds
	return nil
}

func convert_v1_RBDVolumeSource_To_api_RBDVolumeSource(in *RBDVolumeSource, out *api.RBDVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*RBDVolumeSource))(in)
	}
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
		out.SecretRef = new(api.LocalObjectReference)
		if err := convert_v1_LocalObjectReference_To_api_LocalObjectReference(in.SecretRef, out.SecretRef, s); err != nil {
			return err
		}
	} else {
		out.SecretRef = nil
	}
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1_RangeAllocation_To_api_RangeAllocation(in *RangeAllocation, out *api.RangeAllocation, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*RangeAllocation))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	out.Range = in.Range
	if err := s.Convert(&in.Data, &out.Data, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1_ReplicationController_To_api_ReplicationController(in *ReplicationController, out *api.ReplicationController, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ReplicationController))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ReplicationControllerSpec_To_api_ReplicationControllerSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1_ReplicationControllerStatus_To_api_ReplicationControllerStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_ReplicationControllerList_To_api_ReplicationControllerList(in *ReplicationControllerList, out *api.ReplicationControllerList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ReplicationControllerList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.ReplicationController, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_ReplicationController_To_api_ReplicationController(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_ReplicationControllerStatus_To_api_ReplicationControllerStatus(in *ReplicationControllerStatus, out *api.ReplicationControllerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ReplicationControllerStatus))(in)
	}
	out.Replicas = in.Replicas
	out.ObservedGeneration = in.ObservedGeneration
	return nil
}

func convert_v1_ResourceQuota_To_api_ResourceQuota(in *ResourceQuota, out *api.ResourceQuota, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceQuota))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ResourceQuotaSpec_To_api_ResourceQuotaSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1_ResourceQuotaStatus_To_api_ResourceQuotaStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_ResourceQuotaList_To_api_ResourceQuotaList(in *ResourceQuotaList, out *api.ResourceQuotaList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceQuotaList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.ResourceQuota, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_ResourceQuota_To_api_ResourceQuota(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_ResourceQuotaSpec_To_api_ResourceQuotaSpec(in *ResourceQuotaSpec, out *api.ResourceQuotaSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceQuotaSpec))(in)
	}
	if in.Hard != nil {
		out.Hard = make(api.ResourceList)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[api.ResourceName(key)] = newVal
		}
	} else {
		out.Hard = nil
	}
	return nil
}

func convert_v1_ResourceQuotaStatus_To_api_ResourceQuotaStatus(in *ResourceQuotaStatus, out *api.ResourceQuotaStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceQuotaStatus))(in)
	}
	if in.Hard != nil {
		out.Hard = make(api.ResourceList)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[api.ResourceName(key)] = newVal
		}
	} else {
		out.Hard = nil
	}
	if in.Used != nil {
		out.Used = make(api.ResourceList)
		for key, val := range in.Used {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Used[api.ResourceName(key)] = newVal
		}
	} else {
		out.Used = nil
	}
	return nil
}

func convert_v1_ResourceRequirements_To_api_ResourceRequirements(in *ResourceRequirements, out *api.ResourceRequirements, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceRequirements))(in)
	}
	if in.Limits != nil {
		out.Limits = make(api.ResourceList)
		for key, val := range in.Limits {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Limits[api.ResourceName(key)] = newVal
		}
	} else {
		out.Limits = nil
	}
	if in.Requests != nil {
		out.Requests = make(api.ResourceList)
		for key, val := range in.Requests {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Requests[api.ResourceName(key)] = newVal
		}
	} else {
		out.Requests = nil
	}
	return nil
}

func convert_v1_SELinuxOptions_To_api_SELinuxOptions(in *SELinuxOptions, out *api.SELinuxOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SELinuxOptions))(in)
	}
	out.User = in.User
	out.Role = in.Role
	out.Type = in.Type
	out.Level = in.Level
	return nil
}

func convert_v1_Secret_To_api_Secret(in *Secret, out *api.Secret, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Secret))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Data != nil {
		out.Data = make(map[string][]uint8)
		for key, val := range in.Data {
			newVal := []uint8{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Data[key] = newVal
		}
	} else {
		out.Data = nil
	}
	out.Type = api.SecretType(in.Type)
	return nil
}

func convert_v1_SecretList_To_api_SecretList(in *SecretList, out *api.SecretList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SecretList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.Secret, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_Secret_To_api_Secret(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_SecretVolumeSource_To_api_SecretVolumeSource(in *SecretVolumeSource, out *api.SecretVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SecretVolumeSource))(in)
	}
	out.SecretName = in.SecretName
	return nil
}

func convert_v1_SecurityContext_To_api_SecurityContext(in *SecurityContext, out *api.SecurityContext, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SecurityContext))(in)
	}
	if in.Capabilities != nil {
		out.Capabilities = new(api.Capabilities)
		if err := convert_v1_Capabilities_To_api_Capabilities(in.Capabilities, out.Capabilities, s); err != nil {
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
		out.SELinuxOptions = new(api.SELinuxOptions)
		if err := convert_v1_SELinuxOptions_To_api_SELinuxOptions(in.SELinuxOptions, out.SELinuxOptions, s); err != nil {
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

func convert_v1_SerializedReference_To_api_SerializedReference(in *SerializedReference, out *api.SerializedReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SerializedReference))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectReference_To_api_ObjectReference(&in.Reference, &out.Reference, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_Service_To_api_Service(in *Service, out *api.Service, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Service))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ServiceSpec_To_api_ServiceSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1_ServiceStatus_To_api_ServiceStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_ServiceAccount_To_api_ServiceAccount(in *ServiceAccount, out *api.ServiceAccount, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceAccount))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Secrets != nil {
		out.Secrets = make([]api.ObjectReference, len(in.Secrets))
		for i := range in.Secrets {
			if err := convert_v1_ObjectReference_To_api_ObjectReference(&in.Secrets[i], &out.Secrets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Secrets = nil
	}
	if in.ImagePullSecrets != nil {
		out.ImagePullSecrets = make([]api.LocalObjectReference, len(in.ImagePullSecrets))
		for i := range in.ImagePullSecrets {
			if err := convert_v1_LocalObjectReference_To_api_LocalObjectReference(&in.ImagePullSecrets[i], &out.ImagePullSecrets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.ImagePullSecrets = nil
	}
	return nil
}

func convert_v1_ServiceAccountList_To_api_ServiceAccountList(in *ServiceAccountList, out *api.ServiceAccountList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceAccountList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.ServiceAccount, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_ServiceAccount_To_api_ServiceAccount(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_ServiceList_To_api_ServiceList(in *ServiceList, out *api.ServiceList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceList))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]api.Service, len(in.Items))
		for i := range in.Items {
			if err := convert_v1_Service_To_api_Service(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1_ServicePort_To_api_ServicePort(in *ServicePort, out *api.ServicePort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServicePort))(in)
	}
	out.Name = in.Name
	out.Protocol = api.Protocol(in.Protocol)
	out.Port = in.Port
	if err := s.Convert(&in.TargetPort, &out.TargetPort, 0); err != nil {
		return err
	}
	out.NodePort = in.NodePort
	return nil
}

func convert_v1_ServiceSpec_To_api_ServiceSpec(in *ServiceSpec, out *api.ServiceSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceSpec))(in)
	}
	if in.Ports != nil {
		out.Ports = make([]api.ServicePort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_v1_ServicePort_To_api_ServicePort(&in.Ports[i], &out.Ports[i], s); err != nil {
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
	out.Type = api.ServiceType(in.Type)
	if in.DeprecatedPublicIPs != nil {
		out.DeprecatedPublicIPs = make([]string, len(in.DeprecatedPublicIPs))
		for i := range in.DeprecatedPublicIPs {
			out.DeprecatedPublicIPs[i] = in.DeprecatedPublicIPs[i]
		}
	} else {
		out.DeprecatedPublicIPs = nil
	}
	out.SessionAffinity = api.ServiceAffinity(in.SessionAffinity)
	return nil
}

func convert_v1_ServiceStatus_To_api_ServiceStatus(in *ServiceStatus, out *api.ServiceStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceStatus))(in)
	}
	if err := convert_v1_LoadBalancerStatus_To_api_LoadBalancerStatus(&in.LoadBalancer, &out.LoadBalancer, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_Status_To_api_Status(in *Status, out *api.Status, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Status))(in)
	}
	if err := convert_v1_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	out.Status = in.Status
	out.Message = in.Message
	out.Reason = api.StatusReason(in.Reason)
	if in.Details != nil {
		out.Details = new(api.StatusDetails)
		if err := convert_v1_StatusDetails_To_api_StatusDetails(in.Details, out.Details, s); err != nil {
			return err
		}
	} else {
		out.Details = nil
	}
	out.Code = in.Code
	return nil
}

func convert_v1_StatusCause_To_api_StatusCause(in *StatusCause, out *api.StatusCause, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*StatusCause))(in)
	}
	out.Type = api.CauseType(in.Type)
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func convert_v1_StatusDetails_To_api_StatusDetails(in *StatusDetails, out *api.StatusDetails, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*StatusDetails))(in)
	}
	out.Name = in.Name
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]api.StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := convert_v1_StatusCause_To_api_StatusCause(&in.Causes[i], &out.Causes[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Causes = nil
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func convert_v1_TCPSocketAction_To_api_TCPSocketAction(in *TCPSocketAction, out *api.TCPSocketAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*TCPSocketAction))(in)
	}
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1_TypeMeta_To_api_TypeMeta(in *TypeMeta, out *api.TypeMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*TypeMeta))(in)
	}
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	return nil
}

func convert_v1_Volume_To_api_Volume(in *Volume, out *api.Volume, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Volume))(in)
	}
	out.Name = in.Name
	if err := convert_v1_VolumeSource_To_api_VolumeSource(&in.VolumeSource, &out.VolumeSource, s); err != nil {
		return err
	}
	return nil
}

func convert_v1_VolumeMount_To_api_VolumeMount(in *VolumeMount, out *api.VolumeMount, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*VolumeMount))(in)
	}
	out.Name = in.Name
	out.ReadOnly = in.ReadOnly
	out.MountPath = in.MountPath
	return nil
}

func convert_v1_VolumeSource_To_api_VolumeSource(in *VolumeSource, out *api.VolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*VolumeSource))(in)
	}
	if in.HostPath != nil {
		out.HostPath = new(api.HostPathVolumeSource)
		if err := convert_v1_HostPathVolumeSource_To_api_HostPathVolumeSource(in.HostPath, out.HostPath, s); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.EmptyDir != nil {
		out.EmptyDir = new(api.EmptyDirVolumeSource)
		if err := convert_v1_EmptyDirVolumeSource_To_api_EmptyDirVolumeSource(in.EmptyDir, out.EmptyDir, s); err != nil {
			return err
		}
	} else {
		out.EmptyDir = nil
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(api.GCEPersistentDiskVolumeSource)
		if err := convert_v1_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource(in.GCEPersistentDisk, out.GCEPersistentDisk, s); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(api.AWSElasticBlockStoreVolumeSource)
		if err := convert_v1_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource(in.AWSElasticBlockStore, out.AWSElasticBlockStore, s); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.GitRepo != nil {
		out.GitRepo = new(api.GitRepoVolumeSource)
		if err := convert_v1_GitRepoVolumeSource_To_api_GitRepoVolumeSource(in.GitRepo, out.GitRepo, s); err != nil {
			return err
		}
	} else {
		out.GitRepo = nil
	}
	if in.Secret != nil {
		out.Secret = new(api.SecretVolumeSource)
		if err := convert_v1_SecretVolumeSource_To_api_SecretVolumeSource(in.Secret, out.Secret, s); err != nil {
			return err
		}
	} else {
		out.Secret = nil
	}
	if in.NFS != nil {
		out.NFS = new(api.NFSVolumeSource)
		if err := convert_v1_NFSVolumeSource_To_api_NFSVolumeSource(in.NFS, out.NFS, s); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	if in.ISCSI != nil {
		out.ISCSI = new(api.ISCSIVolumeSource)
		if err := convert_v1_ISCSIVolumeSource_To_api_ISCSIVolumeSource(in.ISCSI, out.ISCSI, s); err != nil {
			return err
		}
	} else {
		out.ISCSI = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(api.GlusterfsVolumeSource)
		if err := convert_v1_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource(in.Glusterfs, out.Glusterfs, s); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.PersistentVolumeClaim != nil {
		out.PersistentVolumeClaim = new(api.PersistentVolumeClaimVolumeSource)
		if err := convert_v1_PersistentVolumeClaimVolumeSource_To_api_PersistentVolumeClaimVolumeSource(in.PersistentVolumeClaim, out.PersistentVolumeClaim, s); err != nil {
			return err
		}
	} else {
		out.PersistentVolumeClaim = nil
	}
	if in.RBD != nil {
		out.RBD = new(api.RBDVolumeSource)
		if err := convert_v1_RBDVolumeSource_To_api_RBDVolumeSource(in.RBD, out.RBD, s); err != nil {
			return err
		}
	} else {
		out.RBD = nil
	}
	return nil
}

func init() {
	err := api.Scheme.AddGeneratedConversionFuncs(
		convert_api_AWSElasticBlockStoreVolumeSource_To_v1_AWSElasticBlockStoreVolumeSource,
		convert_api_Binding_To_v1_Binding,
		convert_api_Capabilities_To_v1_Capabilities,
		convert_api_ComponentCondition_To_v1_ComponentCondition,
		convert_api_ComponentStatusList_To_v1_ComponentStatusList,
		convert_api_ComponentStatus_To_v1_ComponentStatus,
		convert_api_ContainerPort_To_v1_ContainerPort,
		convert_api_ContainerStateRunning_To_v1_ContainerStateRunning,
		convert_api_ContainerStateTerminated_To_v1_ContainerStateTerminated,
		convert_api_ContainerStateWaiting_To_v1_ContainerStateWaiting,
		convert_api_ContainerState_To_v1_ContainerState,
		convert_api_ContainerStatus_To_v1_ContainerStatus,
		convert_api_Container_To_v1_Container,
		convert_api_DeleteOptions_To_v1_DeleteOptions,
		convert_api_EmptyDirVolumeSource_To_v1_EmptyDirVolumeSource,
		convert_api_EndpointAddress_To_v1_EndpointAddress,
		convert_api_EndpointPort_To_v1_EndpointPort,
		convert_api_EndpointSubset_To_v1_EndpointSubset,
		convert_api_EndpointsList_To_v1_EndpointsList,
		convert_api_Endpoints_To_v1_Endpoints,
		convert_api_EnvVarSource_To_v1_EnvVarSource,
		convert_api_EnvVar_To_v1_EnvVar,
		convert_api_EventList_To_v1_EventList,
		convert_api_EventSource_To_v1_EventSource,
		convert_api_Event_To_v1_Event,
		convert_api_ExecAction_To_v1_ExecAction,
		convert_api_GCEPersistentDiskVolumeSource_To_v1_GCEPersistentDiskVolumeSource,
		convert_api_GitRepoVolumeSource_To_v1_GitRepoVolumeSource,
		convert_api_GlusterfsVolumeSource_To_v1_GlusterfsVolumeSource,
		convert_api_HTTPGetAction_To_v1_HTTPGetAction,
		convert_api_Handler_To_v1_Handler,
		convert_api_HostPathVolumeSource_To_v1_HostPathVolumeSource,
		convert_api_ISCSIVolumeSource_To_v1_ISCSIVolumeSource,
		convert_api_Lifecycle_To_v1_Lifecycle,
		convert_api_LimitRangeItem_To_v1_LimitRangeItem,
		convert_api_LimitRangeList_To_v1_LimitRangeList,
		convert_api_LimitRangeSpec_To_v1_LimitRangeSpec,
		convert_api_LimitRange_To_v1_LimitRange,
		convert_api_ListMeta_To_v1_ListMeta,
		convert_api_ListOptions_To_v1_ListOptions,
		convert_api_List_To_v1_List,
		convert_api_LoadBalancerIngress_To_v1_LoadBalancerIngress,
		convert_api_LoadBalancerStatus_To_v1_LoadBalancerStatus,
		convert_api_LocalObjectReference_To_v1_LocalObjectReference,
		convert_api_NFSVolumeSource_To_v1_NFSVolumeSource,
		convert_api_NamespaceList_To_v1_NamespaceList,
		convert_api_NamespaceSpec_To_v1_NamespaceSpec,
		convert_api_NamespaceStatus_To_v1_NamespaceStatus,
		convert_api_Namespace_To_v1_Namespace,
		convert_api_NodeAddress_To_v1_NodeAddress,
		convert_api_NodeCondition_To_v1_NodeCondition,
		convert_api_NodeList_To_v1_NodeList,
		convert_api_NodeSpec_To_v1_NodeSpec,
		convert_api_NodeStatus_To_v1_NodeStatus,
		convert_api_NodeSystemInfo_To_v1_NodeSystemInfo,
		convert_api_Node_To_v1_Node,
		convert_api_ObjectFieldSelector_To_v1_ObjectFieldSelector,
		convert_api_ObjectMeta_To_v1_ObjectMeta,
		convert_api_ObjectReference_To_v1_ObjectReference,
		convert_api_PersistentVolumeClaimList_To_v1_PersistentVolumeClaimList,
		convert_api_PersistentVolumeClaimSpec_To_v1_PersistentVolumeClaimSpec,
		convert_api_PersistentVolumeClaimStatus_To_v1_PersistentVolumeClaimStatus,
		convert_api_PersistentVolumeClaimVolumeSource_To_v1_PersistentVolumeClaimVolumeSource,
		convert_api_PersistentVolumeClaim_To_v1_PersistentVolumeClaim,
		convert_api_PersistentVolumeList_To_v1_PersistentVolumeList,
		convert_api_PersistentVolumeSource_To_v1_PersistentVolumeSource,
		convert_api_PersistentVolumeSpec_To_v1_PersistentVolumeSpec,
		convert_api_PersistentVolumeStatus_To_v1_PersistentVolumeStatus,
		convert_api_PersistentVolume_To_v1_PersistentVolume,
		convert_api_PodAttachOptions_To_v1_PodAttachOptions,
		convert_api_PodCondition_To_v1_PodCondition,
		convert_api_PodExecOptions_To_v1_PodExecOptions,
		convert_api_PodList_To_v1_PodList,
		convert_api_PodLogOptions_To_v1_PodLogOptions,
		convert_api_PodProxyOptions_To_v1_PodProxyOptions,
		convert_api_PodStatusResult_To_v1_PodStatusResult,
		convert_api_PodStatus_To_v1_PodStatus,
		convert_api_PodTemplateList_To_v1_PodTemplateList,
		convert_api_PodTemplateSpec_To_v1_PodTemplateSpec,
		convert_api_PodTemplate_To_v1_PodTemplate,
		convert_api_Pod_To_v1_Pod,
		convert_api_Probe_To_v1_Probe,
		convert_api_RBDVolumeSource_To_v1_RBDVolumeSource,
		convert_api_RangeAllocation_To_v1_RangeAllocation,
		convert_api_ReplicationControllerList_To_v1_ReplicationControllerList,
		convert_api_ReplicationControllerStatus_To_v1_ReplicationControllerStatus,
		convert_api_ReplicationController_To_v1_ReplicationController,
		convert_api_ResourceQuotaList_To_v1_ResourceQuotaList,
		convert_api_ResourceQuotaSpec_To_v1_ResourceQuotaSpec,
		convert_api_ResourceQuotaStatus_To_v1_ResourceQuotaStatus,
		convert_api_ResourceQuota_To_v1_ResourceQuota,
		convert_api_ResourceRequirements_To_v1_ResourceRequirements,
		convert_api_SELinuxOptions_To_v1_SELinuxOptions,
		convert_api_SecretList_To_v1_SecretList,
		convert_api_SecretVolumeSource_To_v1_SecretVolumeSource,
		convert_api_Secret_To_v1_Secret,
		convert_api_SecurityContext_To_v1_SecurityContext,
		convert_api_SerializedReference_To_v1_SerializedReference,
		convert_api_ServiceAccountList_To_v1_ServiceAccountList,
		convert_api_ServiceAccount_To_v1_ServiceAccount,
		convert_api_ServiceList_To_v1_ServiceList,
		convert_api_ServicePort_To_v1_ServicePort,
		convert_api_ServiceSpec_To_v1_ServiceSpec,
		convert_api_ServiceStatus_To_v1_ServiceStatus,
		convert_api_Service_To_v1_Service,
		convert_api_StatusCause_To_v1_StatusCause,
		convert_api_StatusDetails_To_v1_StatusDetails,
		convert_api_Status_To_v1_Status,
		convert_api_TCPSocketAction_To_v1_TCPSocketAction,
		convert_api_TypeMeta_To_v1_TypeMeta,
		convert_api_VolumeMount_To_v1_VolumeMount,
		convert_api_VolumeSource_To_v1_VolumeSource,
		convert_api_Volume_To_v1_Volume,
		convert_v1_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource,
		convert_v1_Binding_To_api_Binding,
		convert_v1_Capabilities_To_api_Capabilities,
		convert_v1_ComponentCondition_To_api_ComponentCondition,
		convert_v1_ComponentStatusList_To_api_ComponentStatusList,
		convert_v1_ComponentStatus_To_api_ComponentStatus,
		convert_v1_ContainerPort_To_api_ContainerPort,
		convert_v1_ContainerStateRunning_To_api_ContainerStateRunning,
		convert_v1_ContainerStateTerminated_To_api_ContainerStateTerminated,
		convert_v1_ContainerStateWaiting_To_api_ContainerStateWaiting,
		convert_v1_ContainerState_To_api_ContainerState,
		convert_v1_ContainerStatus_To_api_ContainerStatus,
		convert_v1_Container_To_api_Container,
		convert_v1_DeleteOptions_To_api_DeleteOptions,
		convert_v1_EmptyDirVolumeSource_To_api_EmptyDirVolumeSource,
		convert_v1_EndpointAddress_To_api_EndpointAddress,
		convert_v1_EndpointPort_To_api_EndpointPort,
		convert_v1_EndpointSubset_To_api_EndpointSubset,
		convert_v1_EndpointsList_To_api_EndpointsList,
		convert_v1_Endpoints_To_api_Endpoints,
		convert_v1_EnvVarSource_To_api_EnvVarSource,
		convert_v1_EnvVar_To_api_EnvVar,
		convert_v1_EventList_To_api_EventList,
		convert_v1_EventSource_To_api_EventSource,
		convert_v1_Event_To_api_Event,
		convert_v1_ExecAction_To_api_ExecAction,
		convert_v1_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource,
		convert_v1_GitRepoVolumeSource_To_api_GitRepoVolumeSource,
		convert_v1_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource,
		convert_v1_HTTPGetAction_To_api_HTTPGetAction,
		convert_v1_Handler_To_api_Handler,
		convert_v1_HostPathVolumeSource_To_api_HostPathVolumeSource,
		convert_v1_ISCSIVolumeSource_To_api_ISCSIVolumeSource,
		convert_v1_Lifecycle_To_api_Lifecycle,
		convert_v1_LimitRangeItem_To_api_LimitRangeItem,
		convert_v1_LimitRangeList_To_api_LimitRangeList,
		convert_v1_LimitRangeSpec_To_api_LimitRangeSpec,
		convert_v1_LimitRange_To_api_LimitRange,
		convert_v1_ListMeta_To_api_ListMeta,
		convert_v1_ListOptions_To_api_ListOptions,
		convert_v1_List_To_api_List,
		convert_v1_LoadBalancerIngress_To_api_LoadBalancerIngress,
		convert_v1_LoadBalancerStatus_To_api_LoadBalancerStatus,
		convert_v1_LocalObjectReference_To_api_LocalObjectReference,
		convert_v1_NFSVolumeSource_To_api_NFSVolumeSource,
		convert_v1_NamespaceList_To_api_NamespaceList,
		convert_v1_NamespaceSpec_To_api_NamespaceSpec,
		convert_v1_NamespaceStatus_To_api_NamespaceStatus,
		convert_v1_Namespace_To_api_Namespace,
		convert_v1_NodeAddress_To_api_NodeAddress,
		convert_v1_NodeCondition_To_api_NodeCondition,
		convert_v1_NodeList_To_api_NodeList,
		convert_v1_NodeSpec_To_api_NodeSpec,
		convert_v1_NodeStatus_To_api_NodeStatus,
		convert_v1_NodeSystemInfo_To_api_NodeSystemInfo,
		convert_v1_Node_To_api_Node,
		convert_v1_ObjectFieldSelector_To_api_ObjectFieldSelector,
		convert_v1_ObjectMeta_To_api_ObjectMeta,
		convert_v1_ObjectReference_To_api_ObjectReference,
		convert_v1_PersistentVolumeClaimList_To_api_PersistentVolumeClaimList,
		convert_v1_PersistentVolumeClaimSpec_To_api_PersistentVolumeClaimSpec,
		convert_v1_PersistentVolumeClaimStatus_To_api_PersistentVolumeClaimStatus,
		convert_v1_PersistentVolumeClaimVolumeSource_To_api_PersistentVolumeClaimVolumeSource,
		convert_v1_PersistentVolumeClaim_To_api_PersistentVolumeClaim,
		convert_v1_PersistentVolumeList_To_api_PersistentVolumeList,
		convert_v1_PersistentVolumeSource_To_api_PersistentVolumeSource,
		convert_v1_PersistentVolumeSpec_To_api_PersistentVolumeSpec,
		convert_v1_PersistentVolumeStatus_To_api_PersistentVolumeStatus,
		convert_v1_PersistentVolume_To_api_PersistentVolume,
		convert_v1_PodAttachOptions_To_api_PodAttachOptions,
		convert_v1_PodCondition_To_api_PodCondition,
		convert_v1_PodExecOptions_To_api_PodExecOptions,
		convert_v1_PodList_To_api_PodList,
		convert_v1_PodLogOptions_To_api_PodLogOptions,
		convert_v1_PodProxyOptions_To_api_PodProxyOptions,
		convert_v1_PodStatusResult_To_api_PodStatusResult,
		convert_v1_PodStatus_To_api_PodStatus,
		convert_v1_PodTemplateList_To_api_PodTemplateList,
		convert_v1_PodTemplateSpec_To_api_PodTemplateSpec,
		convert_v1_PodTemplate_To_api_PodTemplate,
		convert_v1_Pod_To_api_Pod,
		convert_v1_Probe_To_api_Probe,
		convert_v1_RBDVolumeSource_To_api_RBDVolumeSource,
		convert_v1_RangeAllocation_To_api_RangeAllocation,
		convert_v1_ReplicationControllerList_To_api_ReplicationControllerList,
		convert_v1_ReplicationControllerStatus_To_api_ReplicationControllerStatus,
		convert_v1_ReplicationController_To_api_ReplicationController,
		convert_v1_ResourceQuotaList_To_api_ResourceQuotaList,
		convert_v1_ResourceQuotaSpec_To_api_ResourceQuotaSpec,
		convert_v1_ResourceQuotaStatus_To_api_ResourceQuotaStatus,
		convert_v1_ResourceQuota_To_api_ResourceQuota,
		convert_v1_ResourceRequirements_To_api_ResourceRequirements,
		convert_v1_SELinuxOptions_To_api_SELinuxOptions,
		convert_v1_SecretList_To_api_SecretList,
		convert_v1_SecretVolumeSource_To_api_SecretVolumeSource,
		convert_v1_Secret_To_api_Secret,
		convert_v1_SecurityContext_To_api_SecurityContext,
		convert_v1_SerializedReference_To_api_SerializedReference,
		convert_v1_ServiceAccountList_To_api_ServiceAccountList,
		convert_v1_ServiceAccount_To_api_ServiceAccount,
		convert_v1_ServiceList_To_api_ServiceList,
		convert_v1_ServicePort_To_api_ServicePort,
		convert_v1_ServiceSpec_To_api_ServiceSpec,
		convert_v1_ServiceStatus_To_api_ServiceStatus,
		convert_v1_Service_To_api_Service,
		convert_v1_StatusCause_To_api_StatusCause,
		convert_v1_StatusDetails_To_api_StatusDetails,
		convert_v1_Status_To_api_Status,
		convert_v1_TCPSocketAction_To_api_TCPSocketAction,
		convert_v1_TypeMeta_To_api_TypeMeta,
		convert_v1_VolumeMount_To_api_VolumeMount,
		convert_v1_VolumeSource_To_api_VolumeSource,
		convert_v1_Volume_To_api_Volume,
	)
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}

// AUTO-GENERATED FUNCTIONS END HERE
