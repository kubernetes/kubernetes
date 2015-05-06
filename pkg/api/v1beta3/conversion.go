/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package v1beta3

import (
	"fmt"
	"reflect"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

// AUTO-GENERATED FUNCTIONS START HERE
func convert_v1beta3_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource(in *AWSElasticBlockStoreVolumeSource, out *newer.AWSElasticBlockStoreVolumeSource, s conversion.Scope) error {
	out.VolumeID = in.VolumeID
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_AWSElasticBlockStoreVolumeSource_To_v1beta3_AWSElasticBlockStoreVolumeSource(in *newer.AWSElasticBlockStoreVolumeSource, out *AWSElasticBlockStoreVolumeSource, s conversion.Scope) error {
	out.VolumeID = in.VolumeID
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_Binding_To_api_Binding(in *Binding, out *newer.Binding, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Target, &out.Target, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Binding_To_v1beta3_Binding(in *newer.Binding, out *Binding, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Target, &out.Target, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Capabilities_To_v1beta3_Capabilities(in *newer.Capabilities, out *Capabilities, s conversion.Scope) error {
	if in.Add != nil {
		out.Add = make([]CapabilityType, len(in.Add))
		for i := range in.Add {
			out.Add[i] = CapabilityType(in.Add[i])
		}
	}
	if in.Drop != nil {
		out.Drop = make([]CapabilityType, len(in.Drop))
		for i := range in.Drop {
			out.Drop[i] = CapabilityType(in.Drop[i])
		}
	}
	return nil
}

func convert_v1beta3_Capabilities_To_api_Capabilities(in *Capabilities, out *newer.Capabilities, s conversion.Scope) error {
	if in.Add != nil {
		out.Add = make([]newer.CapabilityType, len(in.Add))
		for i := range in.Add {
			out.Add[i] = newer.CapabilityType(in.Add[i])
		}
	}
	if in.Drop != nil {
		out.Drop = make([]newer.CapabilityType, len(in.Drop))
		for i := range in.Drop {
			out.Drop[i] = newer.CapabilityType(in.Drop[i])
		}
	}
	return nil
}

func convert_v1beta3_ComponentCondition_To_api_ComponentCondition(in *ComponentCondition, out *newer.ComponentCondition, s conversion.Scope) error {
	out.Type = newer.ComponentConditionType(in.Type)
	out.Status = newer.ConditionStatus(in.Status)
	out.Message = in.Message
	out.Error = in.Error
	return nil
}

func convert_api_ComponentCondition_To_v1beta3_ComponentCondition(in *newer.ComponentCondition, out *ComponentCondition, s conversion.Scope) error {
	out.Type = ComponentConditionType(in.Type)
	out.Status = ConditionStatus(in.Status)
	out.Message = in.Message
	out.Error = in.Error
	return nil
}

func convert_v1beta3_ComponentStatus_To_api_ComponentStatus(in *ComponentStatus, out *newer.ComponentStatus, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if in.Conditions != nil {
		out.Conditions = make([]newer.ComponentCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := s.Convert(&in.Conditions[i], &out.Conditions[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_ComponentStatus_To_v1beta3_ComponentStatus(in *newer.ComponentStatus, out *ComponentStatus, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if in.Conditions != nil {
		out.Conditions = make([]ComponentCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := s.Convert(&in.Conditions[i], &out.Conditions[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_ComponentStatusList_To_api_ComponentStatusList(in *ComponentStatusList, out *newer.ComponentStatusList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.ComponentStatus, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_ComponentStatusList_To_v1beta3_ComponentStatusList(in *newer.ComponentStatusList, out *ComponentStatusList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ComponentStatus, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_ContainerPort_To_api_ContainerPort(in *ContainerPort, out *newer.ContainerPort, s conversion.Scope) error {
	out.Name = in.Name
	out.HostPort = in.HostPort
	out.ContainerPort = in.ContainerPort
	out.Protocol = newer.Protocol(in.Protocol)
	out.HostIP = in.HostIP
	return nil
}

func convert_api_ContainerPort_To_v1beta3_ContainerPort(in *newer.ContainerPort, out *ContainerPort, s conversion.Scope) error {
	out.Name = in.Name
	out.HostPort = in.HostPort
	out.ContainerPort = in.ContainerPort
	out.Protocol = Protocol(in.Protocol)
	out.HostIP = in.HostIP
	return nil
}

func convert_v1beta3_ContainerState_To_api_ContainerState(in *ContainerState, out *newer.ContainerState, s conversion.Scope) error {
	if err := s.Convert(&in.Waiting, &out.Waiting, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Running, &out.Running, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Termination, &out.Termination, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ContainerState_To_v1beta3_ContainerState(in *newer.ContainerState, out *ContainerState, s conversion.Scope) error {
	if err := s.Convert(&in.Waiting, &out.Waiting, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Running, &out.Running, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Termination, &out.Termination, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ContainerStateRunning_To_api_ContainerStateRunning(in *ContainerStateRunning, out *newer.ContainerStateRunning, s conversion.Scope) error {
	if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ContainerStateRunning_To_v1beta3_ContainerStateRunning(in *newer.ContainerStateRunning, out *ContainerStateRunning, s conversion.Scope) error {
	if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ContainerStateTerminated_To_api_ContainerStateTerminated(in *ContainerStateTerminated, out *newer.ContainerStateTerminated, s conversion.Scope) error {
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

func convert_api_ContainerStateTerminated_To_v1beta3_ContainerStateTerminated(in *newer.ContainerStateTerminated, out *ContainerStateTerminated, s conversion.Scope) error {
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

func convert_v1beta3_ContainerStateWaiting_To_api_ContainerStateWaiting(in *ContainerStateWaiting, out *newer.ContainerStateWaiting, s conversion.Scope) error {
	out.Reason = in.Reason
	return nil
}

func convert_api_ContainerStateWaiting_To_v1beta3_ContainerStateWaiting(in *newer.ContainerStateWaiting, out *ContainerStateWaiting, s conversion.Scope) error {
	out.Reason = in.Reason
	return nil
}

func convert_v1beta3_ContainerStatus_To_api_ContainerStatus(in *ContainerStatus, out *newer.ContainerStatus, s conversion.Scope) error {
	out.Name = in.Name
	if err := s.Convert(&in.State, &out.State, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.LastTerminationState, &out.LastTerminationState, 0); err != nil {
		return err
	}
	out.Ready = in.Ready
	out.RestartCount = in.RestartCount
	out.Image = in.Image
	out.ImageID = in.ImageID
	out.ContainerID = in.ContainerID
	return nil
}

func convert_api_ContainerStatus_To_v1beta3_ContainerStatus(in *newer.ContainerStatus, out *ContainerStatus, s conversion.Scope) error {
	out.Name = in.Name
	if err := s.Convert(&in.State, &out.State, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.LastTerminationState, &out.LastTerminationState, 0); err != nil {
		return err
	}
	out.Ready = in.Ready
	out.RestartCount = in.RestartCount
	out.Image = in.Image
	out.ImageID = in.ImageID
	out.ContainerID = in.ContainerID
	return nil
}

func convert_v1beta3_DeleteOptions_To_api_DeleteOptions(in *DeleteOptions, out *newer.DeleteOptions, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if in.GracePeriodSeconds != nil {
		out.GracePeriodSeconds = new(int64)
		*out.GracePeriodSeconds = *in.GracePeriodSeconds
	}
	return nil
}

func convert_api_DeleteOptions_To_v1beta3_DeleteOptions(in *newer.DeleteOptions, out *DeleteOptions, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if in.GracePeriodSeconds != nil {
		out.GracePeriodSeconds = new(int64)
		*out.GracePeriodSeconds = *in.GracePeriodSeconds
	}
	return nil
}

func convert_v1beta3_EmptyDirVolumeSource_To_api_EmptyDirVolumeSource(in *EmptyDirVolumeSource, out *newer.EmptyDirVolumeSource, s conversion.Scope) error {
	out.Medium = newer.StorageType(in.Medium)
	return nil
}

func convert_api_EmptyDirVolumeSource_To_v1beta3_EmptyDirVolumeSource(in *newer.EmptyDirVolumeSource, out *EmptyDirVolumeSource, s conversion.Scope) error {
	out.Medium = StorageType(in.Medium)
	return nil
}

func convert_v1beta3_EndpointAddress_To_api_EndpointAddress(in *EndpointAddress, out *newer.EndpointAddress, s conversion.Scope) error {
	out.IP = in.IP
	if err := s.Convert(&in.TargetRef, &out.TargetRef, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_EndpointAddress_To_v1beta3_EndpointAddress(in *newer.EndpointAddress, out *EndpointAddress, s conversion.Scope) error {
	out.IP = in.IP
	if err := s.Convert(&in.TargetRef, &out.TargetRef, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_EndpointPort_To_api_EndpointPort(in *EndpointPort, out *newer.EndpointPort, s conversion.Scope) error {
	out.Name = in.Name
	out.Port = in.Port
	out.Protocol = newer.Protocol(in.Protocol)
	return nil
}

func convert_api_EndpointPort_To_v1beta3_EndpointPort(in *newer.EndpointPort, out *EndpointPort, s conversion.Scope) error {
	out.Name = in.Name
	out.Port = in.Port
	out.Protocol = Protocol(in.Protocol)
	return nil
}

func convert_v1beta3_EndpointSubset_To_api_EndpointSubset(in *EndpointSubset, out *newer.EndpointSubset, s conversion.Scope) error {
	if in.Addresses != nil {
		out.Addresses = make([]newer.EndpointAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := s.Convert(&in.Addresses[i], &out.Addresses[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Ports != nil {
		out.Ports = make([]newer.EndpointPort, len(in.Ports))
		for i := range in.Ports {
			if err := s.Convert(&in.Ports[i], &out.Ports[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_EndpointSubset_To_v1beta3_EndpointSubset(in *newer.EndpointSubset, out *EndpointSubset, s conversion.Scope) error {
	if in.Addresses != nil {
		out.Addresses = make([]EndpointAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := s.Convert(&in.Addresses[i], &out.Addresses[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Ports != nil {
		out.Ports = make([]EndpointPort, len(in.Ports))
		for i := range in.Ports {
			if err := s.Convert(&in.Ports[i], &out.Ports[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_Endpoints_To_api_Endpoints(in *Endpoints, out *newer.Endpoints, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if in.Subsets != nil {
		out.Subsets = make([]newer.EndpointSubset, len(in.Subsets))
		for i := range in.Subsets {
			if err := s.Convert(&in.Subsets[i], &out.Subsets[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_Endpoints_To_v1beta3_Endpoints(in *newer.Endpoints, out *Endpoints, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if in.Subsets != nil {
		out.Subsets = make([]EndpointSubset, len(in.Subsets))
		for i := range in.Subsets {
			if err := s.Convert(&in.Subsets[i], &out.Subsets[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_EndpointsList_To_api_EndpointsList(in *EndpointsList, out *newer.EndpointsList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Endpoints, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_EndpointsList_To_v1beta3_EndpointsList(in *newer.EndpointsList, out *EndpointsList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Endpoints, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_EnvVar_To_api_EnvVar(in *EnvVar, out *newer.EnvVar, s conversion.Scope) error {
	out.Name = in.Name
	out.Value = in.Value
	if err := s.Convert(&in.ValueFrom, &out.ValueFrom, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_EnvVar_To_v1beta3_EnvVar(in *newer.EnvVar, out *EnvVar, s conversion.Scope) error {
	out.Name = in.Name
	out.Value = in.Value
	if err := s.Convert(&in.ValueFrom, &out.ValueFrom, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_EnvVarSource_To_api_EnvVarSource(in *EnvVarSource, out *newer.EnvVarSource, s conversion.Scope) error {
	if err := s.Convert(&in.FieldRef, &out.FieldRef, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_EnvVarSource_To_v1beta3_EnvVarSource(in *newer.EnvVarSource, out *EnvVarSource, s conversion.Scope) error {
	if err := s.Convert(&in.FieldRef, &out.FieldRef, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_Event_To_api_Event(in *Event, out *newer.Event, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.InvolvedObject, &out.InvolvedObject, 0); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	if err := s.Convert(&in.Source, &out.Source, 0); err != nil {
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

func convert_api_Event_To_v1beta3_Event(in *newer.Event, out *Event, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.InvolvedObject, &out.InvolvedObject, 0); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	if err := s.Convert(&in.Source, &out.Source, 0); err != nil {
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

func convert_v1beta3_EventList_To_api_EventList(in *EventList, out *newer.EventList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Event, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_EventList_To_v1beta3_EventList(in *newer.EventList, out *EventList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Event, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_EventSource_To_api_EventSource(in *EventSource, out *newer.EventSource, s conversion.Scope) error {
	out.Component = in.Component
	out.Host = in.Host
	return nil
}

func convert_api_EventSource_To_v1beta3_EventSource(in *newer.EventSource, out *EventSource, s conversion.Scope) error {
	out.Component = in.Component
	out.Host = in.Host
	return nil
}

func convert_v1beta3_ExecAction_To_api_ExecAction(in *ExecAction, out *newer.ExecAction, s conversion.Scope) error {
	if in.Command != nil {
		out.Command = make([]string, len(in.Command))
		for i := range in.Command {
			out.Command[i] = in.Command[i]
		}
	}
	return nil
}

func convert_api_ExecAction_To_v1beta3_ExecAction(in *newer.ExecAction, out *ExecAction, s conversion.Scope) error {
	if in.Command != nil {
		out.Command = make([]string, len(in.Command))
		for i := range in.Command {
			out.Command[i] = in.Command[i]
		}
	}
	return nil
}

func convert_v1beta3_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource(in *GCEPersistentDiskVolumeSource, out *newer.GCEPersistentDiskVolumeSource, s conversion.Scope) error {
	out.PDName = in.PDName
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_GCEPersistentDiskVolumeSource_To_v1beta3_GCEPersistentDiskVolumeSource(in *newer.GCEPersistentDiskVolumeSource, out *GCEPersistentDiskVolumeSource, s conversion.Scope) error {
	out.PDName = in.PDName
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_GitRepoVolumeSource_To_api_GitRepoVolumeSource(in *GitRepoVolumeSource, out *newer.GitRepoVolumeSource, s conversion.Scope) error {
	out.Repository = in.Repository
	out.Revision = in.Revision
	return nil
}

func convert_api_GitRepoVolumeSource_To_v1beta3_GitRepoVolumeSource(in *newer.GitRepoVolumeSource, out *GitRepoVolumeSource, s conversion.Scope) error {
	out.Repository = in.Repository
	out.Revision = in.Revision
	return nil
}

func convert_v1beta3_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource(in *GlusterfsVolumeSource, out *newer.GlusterfsVolumeSource, s conversion.Scope) error {
	out.EndpointsName = in.EndpointsName
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_GlusterfsVolumeSource_To_v1beta3_GlusterfsVolumeSource(in *newer.GlusterfsVolumeSource, out *GlusterfsVolumeSource, s conversion.Scope) error {
	out.EndpointsName = in.EndpointsName
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_HTTPGetAction_To_api_HTTPGetAction(in *HTTPGetAction, out *newer.HTTPGetAction, s conversion.Scope) error {
	out.Path = in.Path
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	out.Host = in.Host
	return nil
}

func convert_api_HTTPGetAction_To_v1beta3_HTTPGetAction(in *newer.HTTPGetAction, out *HTTPGetAction, s conversion.Scope) error {
	out.Path = in.Path
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	out.Host = in.Host
	return nil
}

func convert_v1beta3_Handler_To_api_Handler(in *Handler, out *newer.Handler, s conversion.Scope) error {
	if err := s.Convert(&in.Exec, &out.Exec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.HTTPGet, &out.HTTPGet, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.TCPSocket, &out.TCPSocket, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Handler_To_v1beta3_Handler(in *newer.Handler, out *Handler, s conversion.Scope) error {
	if err := s.Convert(&in.Exec, &out.Exec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.HTTPGet, &out.HTTPGet, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.TCPSocket, &out.TCPSocket, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_HostPathVolumeSource_To_api_HostPathVolumeSource(in *HostPathVolumeSource, out *newer.HostPathVolumeSource, s conversion.Scope) error {
	out.Path = in.Path
	return nil
}

func convert_api_HostPathVolumeSource_To_v1beta3_HostPathVolumeSource(in *newer.HostPathVolumeSource, out *HostPathVolumeSource, s conversion.Scope) error {
	out.Path = in.Path
	return nil
}

func convert_v1beta3_ISCSIVolumeSource_To_api_ISCSIVolumeSource(in *ISCSIVolumeSource, out *newer.ISCSIVolumeSource, s conversion.Scope) error {
	out.TargetPortal = in.TargetPortal
	out.IQN = in.IQN
	out.Lun = in.Lun
	out.FSType = in.FSType
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_ISCSIVolumeSource_To_v1beta3_ISCSIVolumeSource(in *newer.ISCSIVolumeSource, out *ISCSIVolumeSource, s conversion.Scope) error {
	out.TargetPortal = in.TargetPortal
	out.IQN = in.IQN
	out.Lun = in.Lun
	out.FSType = in.FSType
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_Lifecycle_To_api_Lifecycle(in *Lifecycle, out *newer.Lifecycle, s conversion.Scope) error {
	if err := s.Convert(&in.PostStart, &out.PostStart, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.PreStop, &out.PreStop, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Lifecycle_To_v1beta3_Lifecycle(in *newer.Lifecycle, out *Lifecycle, s conversion.Scope) error {
	if err := s.Convert(&in.PostStart, &out.PostStart, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.PreStop, &out.PreStop, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_LimitRange_To_api_LimitRange(in *LimitRange, out *newer.LimitRange, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_LimitRange_To_v1beta3_LimitRange(in *newer.LimitRange, out *LimitRange, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_LimitRangeItem_To_api_LimitRangeItem(in *LimitRangeItem, out *newer.LimitRangeItem, s conversion.Scope) error {
	out.Type = newer.LimitType(in.Type)
	if in.Max != nil {
		out.Max = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Max {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Max[newer.ResourceName(key)] = newVal
		}
	}
	if in.Min != nil {
		out.Min = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Min {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Min[newer.ResourceName(key)] = newVal
		}
	}
	if in.Default != nil {
		out.Default = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Default {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Default[newer.ResourceName(key)] = newVal
		}
	}
	return nil
}

func convert_api_LimitRangeItem_To_v1beta3_LimitRangeItem(in *newer.LimitRangeItem, out *LimitRangeItem, s conversion.Scope) error {
	out.Type = LimitType(in.Type)
	if in.Max != nil {
		out.Max = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Max {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Max[ResourceName(key)] = newVal
		}
	}
	if in.Min != nil {
		out.Min = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Min {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Min[ResourceName(key)] = newVal
		}
	}
	if in.Default != nil {
		out.Default = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Default {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Default[ResourceName(key)] = newVal
		}
	}
	return nil
}

func convert_v1beta3_LimitRangeList_To_api_LimitRangeList(in *LimitRangeList, out *newer.LimitRangeList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.LimitRange, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_LimitRangeList_To_v1beta3_LimitRangeList(in *newer.LimitRangeList, out *LimitRangeList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]LimitRange, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_LimitRangeSpec_To_api_LimitRangeSpec(in *LimitRangeSpec, out *newer.LimitRangeSpec, s conversion.Scope) error {
	if in.Limits != nil {
		out.Limits = make([]newer.LimitRangeItem, len(in.Limits))
		for i := range in.Limits {
			if err := s.Convert(&in.Limits[i], &out.Limits[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_LimitRangeSpec_To_v1beta3_LimitRangeSpec(in *newer.LimitRangeSpec, out *LimitRangeSpec, s conversion.Scope) error {
	if in.Limits != nil {
		out.Limits = make([]LimitRangeItem, len(in.Limits))
		for i := range in.Limits {
			if err := s.Convert(&in.Limits[i], &out.Limits[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_List_To_api_List(in *List, out *newer.List, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_List_To_v1beta3_List(in *newer.List, out *List, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ListMeta_To_api_ListMeta(in *ListMeta, out *newer.ListMeta, s conversion.Scope) error {
	out.SelfLink = in.SelfLink
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func convert_api_ListMeta_To_v1beta3_ListMeta(in *newer.ListMeta, out *ListMeta, s conversion.Scope) error {
	out.SelfLink = in.SelfLink
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func convert_v1beta3_NFSVolumeSource_To_api_NFSVolumeSource(in *NFSVolumeSource, out *newer.NFSVolumeSource, s conversion.Scope) error {
	out.Server = in.Server
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_NFSVolumeSource_To_v1beta3_NFSVolumeSource(in *newer.NFSVolumeSource, out *NFSVolumeSource, s conversion.Scope) error {
	out.Server = in.Server
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_Namespace_To_api_Namespace(in *Namespace, out *newer.Namespace, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Namespace_To_v1beta3_Namespace(in *newer.Namespace, out *Namespace, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_NamespaceList_To_api_NamespaceList(in *NamespaceList, out *newer.NamespaceList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Namespace, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_NamespaceList_To_v1beta3_NamespaceList(in *newer.NamespaceList, out *NamespaceList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Namespace, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_NamespaceSpec_To_api_NamespaceSpec(in *NamespaceSpec, out *newer.NamespaceSpec, s conversion.Scope) error {
	if in.Finalizers != nil {
		out.Finalizers = make([]newer.FinalizerName, len(in.Finalizers))
		for i := range in.Finalizers {
			out.Finalizers[i] = newer.FinalizerName(in.Finalizers[i])
		}
	}
	return nil
}

func convert_api_NamespaceSpec_To_v1beta3_NamespaceSpec(in *newer.NamespaceSpec, out *NamespaceSpec, s conversion.Scope) error {
	if in.Finalizers != nil {
		out.Finalizers = make([]FinalizerName, len(in.Finalizers))
		for i := range in.Finalizers {
			out.Finalizers[i] = FinalizerName(in.Finalizers[i])
		}
	}
	return nil
}

func convert_v1beta3_NamespaceStatus_To_api_NamespaceStatus(in *NamespaceStatus, out *newer.NamespaceStatus, s conversion.Scope) error {
	out.Phase = newer.NamespacePhase(in.Phase)
	return nil
}

func convert_api_NamespaceStatus_To_v1beta3_NamespaceStatus(in *newer.NamespaceStatus, out *NamespaceStatus, s conversion.Scope) error {
	out.Phase = NamespacePhase(in.Phase)
	return nil
}

func convert_v1beta3_Node_To_api_Node(in *Node, out *newer.Node, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Node_To_v1beta3_Node(in *newer.Node, out *Node, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_NodeAddress_To_api_NodeAddress(in *NodeAddress, out *newer.NodeAddress, s conversion.Scope) error {
	out.Type = newer.NodeAddressType(in.Type)
	out.Address = in.Address
	return nil
}

func convert_api_NodeAddress_To_v1beta3_NodeAddress(in *newer.NodeAddress, out *NodeAddress, s conversion.Scope) error {
	out.Type = NodeAddressType(in.Type)
	out.Address = in.Address
	return nil
}

func convert_v1beta3_NodeCondition_To_api_NodeCondition(in *NodeCondition, out *newer.NodeCondition, s conversion.Scope) error {
	out.Type = newer.NodeConditionType(in.Type)
	out.Status = newer.ConditionStatus(in.Status)
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

func convert_api_NodeCondition_To_v1beta3_NodeCondition(in *newer.NodeCondition, out *NodeCondition, s conversion.Scope) error {
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

func convert_v1beta3_NodeList_To_api_NodeList(in *NodeList, out *newer.NodeList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Node, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_NodeList_To_v1beta3_NodeList(in *newer.NodeList, out *NodeList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Node, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_NodeSpec_To_api_NodeSpec(in *NodeSpec, out *newer.NodeSpec, s conversion.Scope) error {
	out.PodCIDR = in.PodCIDR
	out.ExternalID = in.ExternalID
	out.Unschedulable = in.Unschedulable
	return nil
}

func convert_api_NodeSpec_To_v1beta3_NodeSpec(in *newer.NodeSpec, out *NodeSpec, s conversion.Scope) error {
	out.PodCIDR = in.PodCIDR
	out.ExternalID = in.ExternalID
	out.Unschedulable = in.Unschedulable
	return nil
}

func convert_v1beta3_NodeStatus_To_api_NodeStatus(in *NodeStatus, out *newer.NodeStatus, s conversion.Scope) error {
	if in.Capacity != nil {
		out.Capacity = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[newer.ResourceName(key)] = newVal
		}
	}
	out.Phase = newer.NodePhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]newer.NodeCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := s.Convert(&in.Conditions[i], &out.Conditions[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Addresses != nil {
		out.Addresses = make([]newer.NodeAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := s.Convert(&in.Addresses[i], &out.Addresses[i], 0); err != nil {
				return err
			}
		}
	}
	if err := s.Convert(&in.NodeInfo, &out.NodeInfo, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_NodeStatus_To_v1beta3_NodeStatus(in *newer.NodeStatus, out *NodeStatus, s conversion.Scope) error {
	if in.Capacity != nil {
		out.Capacity = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[ResourceName(key)] = newVal
		}
	}
	out.Phase = NodePhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]NodeCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := s.Convert(&in.Conditions[i], &out.Conditions[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Addresses != nil {
		out.Addresses = make([]NodeAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := s.Convert(&in.Addresses[i], &out.Addresses[i], 0); err != nil {
				return err
			}
		}
	}
	if err := s.Convert(&in.NodeInfo, &out.NodeInfo, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_NodeSystemInfo_To_api_NodeSystemInfo(in *NodeSystemInfo, out *newer.NodeSystemInfo, s conversion.Scope) error {
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

func convert_api_NodeSystemInfo_To_v1beta3_NodeSystemInfo(in *newer.NodeSystemInfo, out *NodeSystemInfo, s conversion.Scope) error {
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

func convert_v1beta3_ObjectFieldSelector_To_api_ObjectFieldSelector(in *ObjectFieldSelector, out *newer.ObjectFieldSelector, s conversion.Scope) error {
	out.APIVersion = in.APIVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_api_ObjectFieldSelector_To_v1beta3_ObjectFieldSelector(in *newer.ObjectFieldSelector, out *ObjectFieldSelector, s conversion.Scope) error {
	out.APIVersion = in.APIVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_v1beta3_ObjectMeta_To_api_ObjectMeta(in *ObjectMeta, out *newer.ObjectMeta, s conversion.Scope) error {
	out.Name = in.Name
	out.GenerateName = in.GenerateName
	out.Namespace = in.Namespace
	out.SelfLink = in.SelfLink
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
	if err := s.Convert(&in.CreationTimestamp, &out.CreationTimestamp, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.DeletionTimestamp, &out.DeletionTimestamp, 0); err != nil {
		return err
	}
	if in.Labels != nil {
		out.Labels = make(map[string]string)
		for key, val := range in.Labels {
			out.Labels[key] = val
		}
	}
	if in.Annotations != nil {
		out.Annotations = make(map[string]string)
		for key, val := range in.Annotations {
			out.Annotations[key] = val
		}
	}
	return nil
}

func convert_api_ObjectMeta_To_v1beta3_ObjectMeta(in *newer.ObjectMeta, out *ObjectMeta, s conversion.Scope) error {
	out.Name = in.Name
	out.GenerateName = in.GenerateName
	out.Namespace = in.Namespace
	out.SelfLink = in.SelfLink
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
	if err := s.Convert(&in.CreationTimestamp, &out.CreationTimestamp, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.DeletionTimestamp, &out.DeletionTimestamp, 0); err != nil {
		return err
	}
	if in.Labels != nil {
		out.Labels = make(map[string]string)
		for key, val := range in.Labels {
			out.Labels[key] = val
		}
	}
	if in.Annotations != nil {
		out.Annotations = make(map[string]string)
		for key, val := range in.Annotations {
			out.Annotations[key] = val
		}
	}
	return nil
}

func convert_v1beta3_ObjectReference_To_api_ObjectReference(in *ObjectReference, out *newer.ObjectReference, s conversion.Scope) error {
	out.Kind = in.Kind
	out.Namespace = in.Namespace
	out.Name = in.Name
	out.UID = in.UID
	out.APIVersion = in.APIVersion
	out.ResourceVersion = in.ResourceVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_api_ObjectReference_To_v1beta3_ObjectReference(in *newer.ObjectReference, out *ObjectReference, s conversion.Scope) error {
	out.Kind = in.Kind
	out.Namespace = in.Namespace
	out.Name = in.Name
	out.UID = in.UID
	out.APIVersion = in.APIVersion
	out.ResourceVersion = in.ResourceVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_v1beta3_PersistentVolume_To_api_PersistentVolume(in *PersistentVolume, out *newer.PersistentVolume, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolume_To_v1beta3_PersistentVolume(in *newer.PersistentVolume, out *PersistentVolume, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaim_To_api_PersistentVolumeClaim(in *PersistentVolumeClaim, out *newer.PersistentVolumeClaim, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeClaim_To_v1beta3_PersistentVolumeClaim(in *newer.PersistentVolumeClaim, out *PersistentVolumeClaim, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaimList_To_api_PersistentVolumeClaimList(in *PersistentVolumeClaimList, out *newer.PersistentVolumeClaimList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.PersistentVolumeClaim, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_PersistentVolumeClaimList_To_v1beta3_PersistentVolumeClaimList(in *newer.PersistentVolumeClaimList, out *PersistentVolumeClaimList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PersistentVolumeClaim, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaimSpec_To_api_PersistentVolumeClaimSpec(in *PersistentVolumeClaimSpec, out *newer.PersistentVolumeClaimSpec, s conversion.Scope) error {
	if in.AccessModes != nil {
		out.AccessModes = make([]newer.AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = newer.AccessModeType(in.AccessModes[i])
		}
	}
	if err := s.Convert(&in.Resources, &out.Resources, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeClaimSpec_To_v1beta3_PersistentVolumeClaimSpec(in *newer.PersistentVolumeClaimSpec, out *PersistentVolumeClaimSpec, s conversion.Scope) error {
	if in.AccessModes != nil {
		out.AccessModes = make([]AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = AccessModeType(in.AccessModes[i])
		}
	}
	if err := s.Convert(&in.Resources, &out.Resources, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaimStatus_To_api_PersistentVolumeClaimStatus(in *PersistentVolumeClaimStatus, out *newer.PersistentVolumeClaimStatus, s conversion.Scope) error {
	out.Phase = newer.PersistentVolumeClaimPhase(in.Phase)
	if in.AccessModes != nil {
		out.AccessModes = make([]newer.AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = newer.AccessModeType(in.AccessModes[i])
		}
	}
	if in.Capacity != nil {
		out.Capacity = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[newer.ResourceName(key)] = newVal
		}
	}
	if err := s.Convert(&in.VolumeRef, &out.VolumeRef, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeClaimStatus_To_v1beta3_PersistentVolumeClaimStatus(in *newer.PersistentVolumeClaimStatus, out *PersistentVolumeClaimStatus, s conversion.Scope) error {
	out.Phase = PersistentVolumeClaimPhase(in.Phase)
	if in.AccessModes != nil {
		out.AccessModes = make([]AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = AccessModeType(in.AccessModes[i])
		}
	}
	if in.Capacity != nil {
		out.Capacity = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[ResourceName(key)] = newVal
		}
	}
	if err := s.Convert(&in.VolumeRef, &out.VolumeRef, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaimVolumeSource_To_api_PersistentVolumeClaimVolumeSource(in *PersistentVolumeClaimVolumeSource, out *newer.PersistentVolumeClaimVolumeSource, s conversion.Scope) error {
	out.ClaimName = in.ClaimName
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_PersistentVolumeClaimVolumeSource_To_v1beta3_PersistentVolumeClaimVolumeSource(in *newer.PersistentVolumeClaimVolumeSource, out *PersistentVolumeClaimVolumeSource, s conversion.Scope) error {
	out.ClaimName = in.ClaimName
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_PersistentVolumeList_To_api_PersistentVolumeList(in *PersistentVolumeList, out *newer.PersistentVolumeList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.PersistentVolume, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_PersistentVolumeList_To_v1beta3_PersistentVolumeList(in *newer.PersistentVolumeList, out *PersistentVolumeList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PersistentVolume, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_PersistentVolumeSource_To_api_PersistentVolumeSource(in *PersistentVolumeSource, out *newer.PersistentVolumeSource, s conversion.Scope) error {
	if err := s.Convert(&in.GCEPersistentDisk, &out.GCEPersistentDisk, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.AWSElasticBlockStore, &out.AWSElasticBlockStore, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.HostPath, &out.HostPath, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Glusterfs, &out.Glusterfs, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeSource_To_v1beta3_PersistentVolumeSource(in *newer.PersistentVolumeSource, out *PersistentVolumeSource, s conversion.Scope) error {
	if err := s.Convert(&in.GCEPersistentDisk, &out.GCEPersistentDisk, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.AWSElasticBlockStore, &out.AWSElasticBlockStore, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.HostPath, &out.HostPath, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Glusterfs, &out.Glusterfs, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeSpec_To_api_PersistentVolumeSpec(in *PersistentVolumeSpec, out *newer.PersistentVolumeSpec, s conversion.Scope) error {
	if in.Capacity != nil {
		out.Capacity = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[newer.ResourceName(key)] = newVal
		}
	}
	if err := s.Convert(&in.PersistentVolumeSource, &out.PersistentVolumeSource, 0); err != nil {
		return err
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]newer.AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = newer.AccessModeType(in.AccessModes[i])
		}
	}
	if err := s.Convert(&in.ClaimRef, &out.ClaimRef, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeSpec_To_v1beta3_PersistentVolumeSpec(in *newer.PersistentVolumeSpec, out *PersistentVolumeSpec, s conversion.Scope) error {
	if in.Capacity != nil {
		out.Capacity = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Capacity {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Capacity[ResourceName(key)] = newVal
		}
	}
	if err := s.Convert(&in.PersistentVolumeSource, &out.PersistentVolumeSource, 0); err != nil {
		return err
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = AccessModeType(in.AccessModes[i])
		}
	}
	if err := s.Convert(&in.ClaimRef, &out.ClaimRef, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeStatus_To_api_PersistentVolumeStatus(in *PersistentVolumeStatus, out *newer.PersistentVolumeStatus, s conversion.Scope) error {
	out.Phase = newer.PersistentVolumePhase(in.Phase)
	return nil
}

func convert_api_PersistentVolumeStatus_To_v1beta3_PersistentVolumeStatus(in *newer.PersistentVolumeStatus, out *PersistentVolumeStatus, s conversion.Scope) error {
	out.Phase = PersistentVolumePhase(in.Phase)
	return nil
}

func convert_v1beta3_Pod_To_api_Pod(in *Pod, out *newer.Pod, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Pod_To_v1beta3_Pod(in *newer.Pod, out *Pod, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PodCondition_To_api_PodCondition(in *PodCondition, out *newer.PodCondition, s conversion.Scope) error {
	out.Type = newer.PodConditionType(in.Type)
	out.Status = newer.ConditionStatus(in.Status)
	return nil
}

func convert_api_PodCondition_To_v1beta3_PodCondition(in *newer.PodCondition, out *PodCondition, s conversion.Scope) error {
	out.Type = PodConditionType(in.Type)
	out.Status = ConditionStatus(in.Status)
	return nil
}

func convert_v1beta3_PodExecOptions_To_api_PodExecOptions(in *PodExecOptions, out *newer.PodExecOptions, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
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
	}
	return nil
}

func convert_api_PodExecOptions_To_v1beta3_PodExecOptions(in *newer.PodExecOptions, out *PodExecOptions, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
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
	}
	return nil
}

func convert_v1beta3_PodList_To_api_PodList(in *PodList, out *newer.PodList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Pod, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_PodList_To_v1beta3_PodList(in *newer.PodList, out *PodList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Pod, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_PodLogOptions_To_api_PodLogOptions(in *PodLogOptions, out *newer.PodLogOptions, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	out.Container = in.Container
	out.Follow = in.Follow
	return nil
}

func convert_api_PodLogOptions_To_v1beta3_PodLogOptions(in *newer.PodLogOptions, out *PodLogOptions, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	out.Container = in.Container
	out.Follow = in.Follow
	return nil
}

func convert_v1beta3_PodProxyOptions_To_api_PodProxyOptions(in *PodProxyOptions, out *newer.PodProxyOptions, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	out.Path = in.Path
	return nil
}

func convert_api_PodProxyOptions_To_v1beta3_PodProxyOptions(in *newer.PodProxyOptions, out *PodProxyOptions, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	out.Path = in.Path
	return nil
}

func convert_v1beta3_PodSpec_To_api_PodSpec(in *PodSpec, out *newer.PodSpec, s conversion.Scope) error {
	if in.Volumes != nil {
		out.Volumes = make([]newer.Volume, len(in.Volumes))
		for i := range in.Volumes {
			if err := s.Convert(&in.Volumes[i], &out.Volumes[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Containers != nil {
		out.Containers = make([]newer.Container, len(in.Containers))
		for i := range in.Containers {
			if err := s.Convert(&in.Containers[i], &out.Containers[i], 0); err != nil {
				return err
			}
		}
	}
	out.RestartPolicy = newer.RestartPolicy(in.RestartPolicy)
	if in.TerminationGracePeriodSeconds != nil {
		out.TerminationGracePeriodSeconds = new(int64)
		*out.TerminationGracePeriodSeconds = *in.TerminationGracePeriodSeconds
	}
	out.DNSPolicy = newer.DNSPolicy(in.DNSPolicy)
	if in.NodeSelector != nil {
		out.NodeSelector = make(map[string]string)
		for key, val := range in.NodeSelector {
			out.NodeSelector[key] = val
		}
	}
	out.Host = in.Host
	out.HostNetwork = in.HostNetwork
	return nil
}

func convert_api_PodSpec_To_v1beta3_PodSpec(in *newer.PodSpec, out *PodSpec, s conversion.Scope) error {
	if in.Volumes != nil {
		out.Volumes = make([]Volume, len(in.Volumes))
		for i := range in.Volumes {
			if err := s.Convert(&in.Volumes[i], &out.Volumes[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Containers != nil {
		out.Containers = make([]Container, len(in.Containers))
		for i := range in.Containers {
			if err := s.Convert(&in.Containers[i], &out.Containers[i], 0); err != nil {
				return err
			}
		}
	}
	out.RestartPolicy = RestartPolicy(in.RestartPolicy)
	if in.TerminationGracePeriodSeconds != nil {
		out.TerminationGracePeriodSeconds = new(int64)
		*out.TerminationGracePeriodSeconds = *in.TerminationGracePeriodSeconds
	}
	out.DNSPolicy = DNSPolicy(in.DNSPolicy)
	if in.NodeSelector != nil {
		out.NodeSelector = make(map[string]string)
		for key, val := range in.NodeSelector {
			out.NodeSelector[key] = val
		}
	}
	out.Host = in.Host
	out.HostNetwork = in.HostNetwork
	return nil
}

func convert_v1beta3_PodStatus_To_api_PodStatus(in *PodStatus, out *newer.PodStatus, s conversion.Scope) error {
	out.Phase = newer.PodPhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]newer.PodCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := s.Convert(&in.Conditions[i], &out.Conditions[i], 0); err != nil {
				return err
			}
		}
	}
	out.Message = in.Message
	out.HostIP = in.HostIP
	out.PodIP = in.PodIP
	if in.ContainerStatuses != nil {
		out.ContainerStatuses = make([]newer.ContainerStatus, len(in.ContainerStatuses))
		for i := range in.ContainerStatuses {
			if err := s.Convert(&in.ContainerStatuses[i], &out.ContainerStatuses[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_PodStatus_To_v1beta3_PodStatus(in *newer.PodStatus, out *PodStatus, s conversion.Scope) error {
	out.Phase = PodPhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]PodCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := s.Convert(&in.Conditions[i], &out.Conditions[i], 0); err != nil {
				return err
			}
		}
	}
	out.Message = in.Message
	out.HostIP = in.HostIP
	out.PodIP = in.PodIP
	if in.ContainerStatuses != nil {
		out.ContainerStatuses = make([]ContainerStatus, len(in.ContainerStatuses))
		for i := range in.ContainerStatuses {
			if err := s.Convert(&in.ContainerStatuses[i], &out.ContainerStatuses[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_PodStatusResult_To_api_PodStatusResult(in *PodStatusResult, out *newer.PodStatusResult, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PodStatusResult_To_v1beta3_PodStatusResult(in *newer.PodStatusResult, out *PodStatusResult, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PodTemplate_To_api_PodTemplate(in *PodTemplate, out *newer.PodTemplate, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Template, &out.Template, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PodTemplate_To_v1beta3_PodTemplate(in *newer.PodTemplate, out *PodTemplate, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Template, &out.Template, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PodTemplateList_To_api_PodTemplateList(in *PodTemplateList, out *newer.PodTemplateList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.PodTemplate, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_PodTemplateList_To_v1beta3_PodTemplateList(in *newer.PodTemplateList, out *PodTemplateList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PodTemplate, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_PodTemplateSpec_To_api_PodTemplateSpec(in *PodTemplateSpec, out *newer.PodTemplateSpec, s conversion.Scope) error {
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_PodTemplateSpec_To_v1beta3_PodTemplateSpec(in *newer.PodTemplateSpec, out *PodTemplateSpec, s conversion.Scope) error {
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_Probe_To_api_Probe(in *Probe, out *newer.Probe, s conversion.Scope) error {
	if err := s.Convert(&in.Handler, &out.Handler, 0); err != nil {
		return err
	}
	out.InitialDelaySeconds = in.InitialDelaySeconds
	out.TimeoutSeconds = in.TimeoutSeconds
	return nil
}

func convert_api_Probe_To_v1beta3_Probe(in *newer.Probe, out *Probe, s conversion.Scope) error {
	if err := s.Convert(&in.Handler, &out.Handler, 0); err != nil {
		return err
	}
	out.InitialDelaySeconds = in.InitialDelaySeconds
	out.TimeoutSeconds = in.TimeoutSeconds
	return nil
}

func convert_v1beta3_ReplicationController_To_api_ReplicationController(in *ReplicationController, out *newer.ReplicationController, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ReplicationController_To_v1beta3_ReplicationController(in *newer.ReplicationController, out *ReplicationController, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ReplicationControllerList_To_api_ReplicationControllerList(in *ReplicationControllerList, out *newer.ReplicationControllerList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.ReplicationController, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_ReplicationControllerList_To_v1beta3_ReplicationControllerList(in *newer.ReplicationControllerList, out *ReplicationControllerList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ReplicationController, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_ReplicationControllerSpec_To_api_ReplicationControllerSpec(in *ReplicationControllerSpec, out *newer.ReplicationControllerSpec, s conversion.Scope) error {
	out.Replicas = in.Replicas
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	}
	if err := s.Convert(&in.TemplateRef, &out.TemplateRef, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Template, &out.Template, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ReplicationControllerSpec_To_v1beta3_ReplicationControllerSpec(in *newer.ReplicationControllerSpec, out *ReplicationControllerSpec, s conversion.Scope) error {
	out.Replicas = in.Replicas
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	}
	if err := s.Convert(&in.TemplateRef, &out.TemplateRef, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Template, &out.Template, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ReplicationControllerStatus_To_api_ReplicationControllerStatus(in *ReplicationControllerStatus, out *newer.ReplicationControllerStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	return nil
}

func convert_api_ReplicationControllerStatus_To_v1beta3_ReplicationControllerStatus(in *newer.ReplicationControllerStatus, out *ReplicationControllerStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	return nil
}

func convert_v1beta3_ResourceQuota_To_api_ResourceQuota(in *ResourceQuota, out *newer.ResourceQuota, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ResourceQuota_To_v1beta3_ResourceQuota(in *newer.ResourceQuota, out *ResourceQuota, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ResourceQuotaList_To_api_ResourceQuotaList(in *ResourceQuotaList, out *newer.ResourceQuotaList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.ResourceQuota, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_ResourceQuotaList_To_v1beta3_ResourceQuotaList(in *newer.ResourceQuotaList, out *ResourceQuotaList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ResourceQuota, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_ResourceQuotaSpec_To_api_ResourceQuotaSpec(in *ResourceQuotaSpec, out *newer.ResourceQuotaSpec, s conversion.Scope) error {
	if in.Hard != nil {
		out.Hard = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[newer.ResourceName(key)] = newVal
		}
	}
	return nil
}

func convert_api_ResourceQuotaSpec_To_v1beta3_ResourceQuotaSpec(in *newer.ResourceQuotaSpec, out *ResourceQuotaSpec, s conversion.Scope) error {
	if in.Hard != nil {
		out.Hard = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[ResourceName(key)] = newVal
		}
	}
	return nil
}

func convert_v1beta3_ResourceQuotaStatus_To_api_ResourceQuotaStatus(in *ResourceQuotaStatus, out *newer.ResourceQuotaStatus, s conversion.Scope) error {
	if in.Hard != nil {
		out.Hard = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[newer.ResourceName(key)] = newVal
		}
	}
	if in.Used != nil {
		out.Used = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Used {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Used[newer.ResourceName(key)] = newVal
		}
	}
	return nil
}

func convert_api_ResourceQuotaStatus_To_v1beta3_ResourceQuotaStatus(in *newer.ResourceQuotaStatus, out *ResourceQuotaStatus, s conversion.Scope) error {
	if in.Hard != nil {
		out.Hard = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[ResourceName(key)] = newVal
		}
	}
	if in.Used != nil {
		out.Used = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Used {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Used[ResourceName(key)] = newVal
		}
	}
	return nil
}

func convert_v1beta3_ResourceRequirements_To_api_ResourceRequirements(in *ResourceRequirements, out *newer.ResourceRequirements, s conversion.Scope) error {
	if in.Limits != nil {
		out.Limits = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Limits {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Limits[newer.ResourceName(key)] = newVal
		}
	}
	if in.Requests != nil {
		out.Requests = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Requests {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Requests[newer.ResourceName(key)] = newVal
		}
	}
	return nil
}

func convert_api_ResourceRequirements_To_v1beta3_ResourceRequirements(in *newer.ResourceRequirements, out *ResourceRequirements, s conversion.Scope) error {
	if in.Limits != nil {
		out.Limits = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Limits {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Limits[ResourceName(key)] = newVal
		}
	}
	if in.Requests != nil {
		out.Requests = make(map[ResourceName]resource.Quantity)
		for key, val := range in.Requests {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Requests[ResourceName(key)] = newVal
		}
	}
	return nil
}

func convert_api_SELinuxOptions_To_v1beta3_SELinuxOptions(in *newer.SELinuxOptions, out *SELinuxOptions, s conversion.Scope) error {
	out.User = in.User
	out.Role = in.Role
	out.Type = in.Type
	out.Level = in.Level
	return nil
}

func convert_v1beta3_SELinuxOptions_To_api_SELinuxOptions(in *SELinuxOptions, out *newer.SELinuxOptions, s conversion.Scope) error {
	out.User = in.User
	out.Role = in.Role
	out.Type = in.Type
	out.Level = in.Level
	return nil
}

func convert_v1beta3_Secret_To_api_Secret(in *Secret, out *newer.Secret, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
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
	}
	out.Type = newer.SecretType(in.Type)
	return nil
}

func convert_api_Secret_To_v1beta3_Secret(in *newer.Secret, out *Secret, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
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
	}
	out.Type = SecretType(in.Type)
	return nil
}

func convert_v1beta3_SecretList_To_api_SecretList(in *SecretList, out *newer.SecretList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Secret, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_SecretList_To_v1beta3_SecretList(in *newer.SecretList, out *SecretList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Secret, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_SecretVolumeSource_To_api_SecretVolumeSource(in *SecretVolumeSource, out *newer.SecretVolumeSource, s conversion.Scope) error {
	out.SecretName = in.SecretName
	return nil
}

func convert_api_SecretVolumeSource_To_v1beta3_SecretVolumeSource(in *newer.SecretVolumeSource, out *SecretVolumeSource, s conversion.Scope) error {
	out.SecretName = in.SecretName
	return nil
}

func convert_api_SecurityContext_To_v1beta3_SecurityContext(in *newer.SecurityContext, out *SecurityContext, s conversion.Scope) error {
	if err := s.Convert(&in.Capabilities, &out.Capabilities, 0); err != nil {
		return err
	}
	if in.Privileged != nil {
		out.Privileged = new(bool)
		*out.Privileged = *in.Privileged
	}
	if err := s.Convert(&in.SELinuxOptions, &out.SELinuxOptions, 0); err != nil {
		return err
	}
	if in.RunAsUser != nil {
		out.RunAsUser = new(int64)
		*out.RunAsUser = *in.RunAsUser
	}
	return nil
}

func convert_v1beta3_SecurityContext_To_api_SecurityContext(in *SecurityContext, out *newer.SecurityContext, s conversion.Scope) error {
	if err := s.Convert(&in.Capabilities, &out.Capabilities, 0); err != nil {
		return err
	}
	if in.Privileged != nil {
		out.Privileged = new(bool)
		*out.Privileged = *in.Privileged
	}
	if err := s.Convert(&in.SELinuxOptions, &out.SELinuxOptions, 0); err != nil {
		return err
	}
	if in.RunAsUser != nil {
		out.RunAsUser = new(int64)
		*out.RunAsUser = *in.RunAsUser
	}
	return nil
}

func convert_v1beta3_SerializedReference_To_api_SerializedReference(in *SerializedReference, out *newer.SerializedReference, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Reference, &out.Reference, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_SerializedReference_To_v1beta3_SerializedReference(in *newer.SerializedReference, out *SerializedReference, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Reference, &out.Reference, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_Service_To_api_Service(in *Service, out *newer.Service, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Service_To_v1beta3_Service(in *newer.Service, out *Service, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ServiceList_To_api_ServiceList(in *ServiceList, out *newer.ServiceList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Service, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_api_ServiceList_To_v1beta3_ServiceList(in *newer.ServiceList, out *ServiceList, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Service, len(in.Items))
		for i := range in.Items {
			if err := s.Convert(&in.Items[i], &out.Items[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func convert_v1beta3_ServicePort_To_api_ServicePort(in *ServicePort, out *newer.ServicePort, s conversion.Scope) error {
	out.Name = in.Name
	out.Protocol = newer.Protocol(in.Protocol)
	out.Port = in.Port
	if err := s.Convert(&in.TargetPort, &out.TargetPort, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ServicePort_To_v1beta3_ServicePort(in *newer.ServicePort, out *ServicePort, s conversion.Scope) error {
	out.Name = in.Name
	out.Protocol = Protocol(in.Protocol)
	out.Port = in.Port
	if err := s.Convert(&in.TargetPort, &out.TargetPort, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ServiceSpec_To_api_ServiceSpec(in *ServiceSpec, out *newer.ServiceSpec, s conversion.Scope) error {
	if in.Ports != nil {
		out.Ports = make([]newer.ServicePort, len(in.Ports))
		for i := range in.Ports {
			if err := s.Convert(&in.Ports[i], &out.Ports[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	}
	out.PortalIP = in.PortalIP
	out.CreateExternalLoadBalancer = in.CreateExternalLoadBalancer
	if in.PublicIPs != nil {
		out.PublicIPs = make([]string, len(in.PublicIPs))
		for i := range in.PublicIPs {
			out.PublicIPs[i] = in.PublicIPs[i]
		}
	}
	out.SessionAffinity = newer.AffinityType(in.SessionAffinity)
	return nil
}

func convert_api_ServiceSpec_To_v1beta3_ServiceSpec(in *newer.ServiceSpec, out *ServiceSpec, s conversion.Scope) error {
	if in.Ports != nil {
		out.Ports = make([]ServicePort, len(in.Ports))
		for i := range in.Ports {
			if err := s.Convert(&in.Ports[i], &out.Ports[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	}
	out.PortalIP = in.PortalIP
	out.CreateExternalLoadBalancer = in.CreateExternalLoadBalancer
	if in.PublicIPs != nil {
		out.PublicIPs = make([]string, len(in.PublicIPs))
		for i := range in.PublicIPs {
			out.PublicIPs[i] = in.PublicIPs[i]
		}
	}
	out.SessionAffinity = AffinityType(in.SessionAffinity)
	return nil
}

func convert_v1beta3_ServiceStatus_To_api_ServiceStatus(in *ServiceStatus, out *newer.ServiceStatus, s conversion.Scope) error {
	return nil
}

func convert_api_ServiceStatus_To_v1beta3_ServiceStatus(in *newer.ServiceStatus, out *ServiceStatus, s conversion.Scope) error {
	return nil
}

func convert_v1beta3_Status_To_api_Status(in *Status, out *newer.Status, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	out.Status = in.Status
	out.Message = in.Message
	out.Reason = newer.StatusReason(in.Reason)
	if err := s.Convert(&in.Details, &out.Details, 0); err != nil {
		return err
	}
	out.Code = in.Code
	return nil
}

func convert_api_Status_To_v1beta3_Status(in *newer.Status, out *Status, s conversion.Scope) error {
	if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ListMeta, &out.ListMeta, 0); err != nil {
		return err
	}
	out.Status = in.Status
	out.Message = in.Message
	out.Reason = StatusReason(in.Reason)
	if err := s.Convert(&in.Details, &out.Details, 0); err != nil {
		return err
	}
	out.Code = in.Code
	return nil
}

func convert_v1beta3_StatusCause_To_api_StatusCause(in *StatusCause, out *newer.StatusCause, s conversion.Scope) error {
	out.Type = newer.CauseType(in.Type)
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func convert_api_StatusCause_To_v1beta3_StatusCause(in *newer.StatusCause, out *StatusCause, s conversion.Scope) error {
	out.Type = CauseType(in.Type)
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func convert_v1beta3_StatusDetails_To_api_StatusDetails(in *StatusDetails, out *newer.StatusDetails, s conversion.Scope) error {
	out.ID = in.ID
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]newer.StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := s.Convert(&in.Causes[i], &out.Causes[i], 0); err != nil {
				return err
			}
		}
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func convert_api_StatusDetails_To_v1beta3_StatusDetails(in *newer.StatusDetails, out *StatusDetails, s conversion.Scope) error {
	out.ID = in.ID
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := s.Convert(&in.Causes[i], &out.Causes[i], 0); err != nil {
				return err
			}
		}
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func convert_v1beta3_TCPSocketAction_To_api_TCPSocketAction(in *TCPSocketAction, out *newer.TCPSocketAction, s conversion.Scope) error {
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_TCPSocketAction_To_v1beta3_TCPSocketAction(in *newer.TCPSocketAction, out *TCPSocketAction, s conversion.Scope) error {
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_TypeMeta_To_api_TypeMeta(in *TypeMeta, out *newer.TypeMeta, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	return nil
}

func convert_api_TypeMeta_To_v1beta3_TypeMeta(in *newer.TypeMeta, out *TypeMeta, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	return nil
}

func convert_v1beta3_Volume_To_api_Volume(in *Volume, out *newer.Volume, s conversion.Scope) error {
	out.Name = in.Name
	if err := s.Convert(&in.VolumeSource, &out.VolumeSource, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Volume_To_v1beta3_Volume(in *newer.Volume, out *Volume, s conversion.Scope) error {
	out.Name = in.Name
	if err := s.Convert(&in.VolumeSource, &out.VolumeSource, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_VolumeMount_To_api_VolumeMount(in *VolumeMount, out *newer.VolumeMount, s conversion.Scope) error {
	out.Name = in.Name
	out.ReadOnly = in.ReadOnly
	out.MountPath = in.MountPath
	return nil
}

func convert_api_VolumeMount_To_v1beta3_VolumeMount(in *newer.VolumeMount, out *VolumeMount, s conversion.Scope) error {
	out.Name = in.Name
	out.ReadOnly = in.ReadOnly
	out.MountPath = in.MountPath
	return nil
}

func convert_v1beta3_VolumeSource_To_api_VolumeSource(in *VolumeSource, out *newer.VolumeSource, s conversion.Scope) error {
	if err := s.Convert(&in.HostPath, &out.HostPath, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.EmptyDir, &out.EmptyDir, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.GCEPersistentDisk, &out.GCEPersistentDisk, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.AWSElasticBlockStore, &out.AWSElasticBlockStore, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.GitRepo, &out.GitRepo, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Secret, &out.Secret, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.NFS, &out.NFS, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ISCSI, &out.ISCSI, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Glusterfs, &out.Glusterfs, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.PersistentVolumeClaimVolumeSource, &out.PersistentVolumeClaimVolumeSource, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_VolumeSource_To_v1beta3_VolumeSource(in *newer.VolumeSource, out *VolumeSource, s conversion.Scope) error {
	if err := s.Convert(&in.HostPath, &out.HostPath, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.EmptyDir, &out.EmptyDir, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.GCEPersistentDisk, &out.GCEPersistentDisk, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.AWSElasticBlockStore, &out.AWSElasticBlockStore, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.GitRepo, &out.GitRepo, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Secret, &out.Secret, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.NFS, &out.NFS, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ISCSI, &out.ISCSI, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Glusterfs, &out.Glusterfs, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.PersistentVolumeClaimVolumeSource, &out.PersistentVolumeClaimVolumeSource, 0); err != nil {
		return err
	}
	return nil
}

// AUTO-GENERATED FUNCTIONS END HERE

func init() {
	err := newer.Scheme.AddGeneratedConversionFuncs(
		convert_api_AWSElasticBlockStoreVolumeSource_To_v1beta3_AWSElasticBlockStoreVolumeSource,
		convert_api_Binding_To_v1beta3_Binding,
		convert_api_Capabilities_To_v1beta3_Capabilities,
		convert_api_ComponentCondition_To_v1beta3_ComponentCondition,
		convert_api_ComponentStatusList_To_v1beta3_ComponentStatusList,
		convert_api_ComponentStatus_To_v1beta3_ComponentStatus,
		convert_api_ContainerPort_To_v1beta3_ContainerPort,
		convert_api_ContainerStateRunning_To_v1beta3_ContainerStateRunning,
		convert_api_ContainerStateTerminated_To_v1beta3_ContainerStateTerminated,
		convert_api_ContainerStateWaiting_To_v1beta3_ContainerStateWaiting,
		convert_api_ContainerState_To_v1beta3_ContainerState,
		convert_api_ContainerStatus_To_v1beta3_ContainerStatus,
		convert_api_DeleteOptions_To_v1beta3_DeleteOptions,
		convert_api_EmptyDirVolumeSource_To_v1beta3_EmptyDirVolumeSource,
		convert_api_EndpointAddress_To_v1beta3_EndpointAddress,
		convert_api_EndpointPort_To_v1beta3_EndpointPort,
		convert_api_EndpointSubset_To_v1beta3_EndpointSubset,
		convert_api_EndpointsList_To_v1beta3_EndpointsList,
		convert_api_Endpoints_To_v1beta3_Endpoints,
		convert_api_EnvVarSource_To_v1beta3_EnvVarSource,
		convert_api_EnvVar_To_v1beta3_EnvVar,
		convert_api_EventList_To_v1beta3_EventList,
		convert_api_EventSource_To_v1beta3_EventSource,
		convert_api_Event_To_v1beta3_Event,
		convert_api_ExecAction_To_v1beta3_ExecAction,
		convert_api_GCEPersistentDiskVolumeSource_To_v1beta3_GCEPersistentDiskVolumeSource,
		convert_api_GitRepoVolumeSource_To_v1beta3_GitRepoVolumeSource,
		convert_api_GlusterfsVolumeSource_To_v1beta3_GlusterfsVolumeSource,
		convert_api_HTTPGetAction_To_v1beta3_HTTPGetAction,
		convert_api_Handler_To_v1beta3_Handler,
		convert_api_HostPathVolumeSource_To_v1beta3_HostPathVolumeSource,
		convert_api_ISCSIVolumeSource_To_v1beta3_ISCSIVolumeSource,
		convert_api_Lifecycle_To_v1beta3_Lifecycle,
		convert_api_LimitRangeItem_To_v1beta3_LimitRangeItem,
		convert_api_LimitRangeList_To_v1beta3_LimitRangeList,
		convert_api_LimitRangeSpec_To_v1beta3_LimitRangeSpec,
		convert_api_LimitRange_To_v1beta3_LimitRange,
		convert_api_ListMeta_To_v1beta3_ListMeta,
		convert_api_List_To_v1beta3_List,
		convert_api_NFSVolumeSource_To_v1beta3_NFSVolumeSource,
		convert_api_NamespaceList_To_v1beta3_NamespaceList,
		convert_api_NamespaceSpec_To_v1beta3_NamespaceSpec,
		convert_api_NamespaceStatus_To_v1beta3_NamespaceStatus,
		convert_api_Namespace_To_v1beta3_Namespace,
		convert_api_NodeAddress_To_v1beta3_NodeAddress,
		convert_api_NodeCondition_To_v1beta3_NodeCondition,
		convert_api_NodeList_To_v1beta3_NodeList,
		convert_api_NodeSpec_To_v1beta3_NodeSpec,
		convert_api_NodeStatus_To_v1beta3_NodeStatus,
		convert_api_NodeSystemInfo_To_v1beta3_NodeSystemInfo,
		convert_api_Node_To_v1beta3_Node,
		convert_api_ObjectFieldSelector_To_v1beta3_ObjectFieldSelector,
		convert_api_ObjectMeta_To_v1beta3_ObjectMeta,
		convert_api_ObjectReference_To_v1beta3_ObjectReference,
		convert_api_PersistentVolumeClaimList_To_v1beta3_PersistentVolumeClaimList,
		convert_api_PersistentVolumeClaimSpec_To_v1beta3_PersistentVolumeClaimSpec,
		convert_api_PersistentVolumeClaimStatus_To_v1beta3_PersistentVolumeClaimStatus,
		convert_api_PersistentVolumeClaimVolumeSource_To_v1beta3_PersistentVolumeClaimVolumeSource,
		convert_api_PersistentVolumeClaim_To_v1beta3_PersistentVolumeClaim,
		convert_api_PersistentVolumeList_To_v1beta3_PersistentVolumeList,
		convert_api_PersistentVolumeSource_To_v1beta3_PersistentVolumeSource,
		convert_api_PersistentVolumeSpec_To_v1beta3_PersistentVolumeSpec,
		convert_api_PersistentVolumeStatus_To_v1beta3_PersistentVolumeStatus,
		convert_api_PersistentVolume_To_v1beta3_PersistentVolume,
		convert_api_PodCondition_To_v1beta3_PodCondition,
		convert_api_PodExecOptions_To_v1beta3_PodExecOptions,
		convert_api_PodList_To_v1beta3_PodList,
		convert_api_PodLogOptions_To_v1beta3_PodLogOptions,
		convert_api_PodProxyOptions_To_v1beta3_PodProxyOptions,
		convert_api_PodSpec_To_v1beta3_PodSpec,
		convert_api_PodStatusResult_To_v1beta3_PodStatusResult,
		convert_api_PodStatus_To_v1beta3_PodStatus,
		convert_api_PodTemplateList_To_v1beta3_PodTemplateList,
		convert_api_PodTemplateSpec_To_v1beta3_PodTemplateSpec,
		convert_api_PodTemplate_To_v1beta3_PodTemplate,
		convert_api_Pod_To_v1beta3_Pod,
		convert_api_Probe_To_v1beta3_Probe,
		convert_api_ReplicationControllerList_To_v1beta3_ReplicationControllerList,
		convert_api_ReplicationControllerSpec_To_v1beta3_ReplicationControllerSpec,
		convert_api_ReplicationControllerStatus_To_v1beta3_ReplicationControllerStatus,
		convert_api_ReplicationController_To_v1beta3_ReplicationController,
		convert_api_ResourceQuotaList_To_v1beta3_ResourceQuotaList,
		convert_api_ResourceQuotaSpec_To_v1beta3_ResourceQuotaSpec,
		convert_api_ResourceQuotaStatus_To_v1beta3_ResourceQuotaStatus,
		convert_api_ResourceQuota_To_v1beta3_ResourceQuota,
		convert_api_ResourceRequirements_To_v1beta3_ResourceRequirements,
		convert_api_SELinuxOptions_To_v1beta3_SELinuxOptions,
		convert_api_SecretList_To_v1beta3_SecretList,
		convert_api_SecretVolumeSource_To_v1beta3_SecretVolumeSource,
		convert_api_Secret_To_v1beta3_Secret,
		convert_api_SecurityContext_To_v1beta3_SecurityContext,
		convert_api_SerializedReference_To_v1beta3_SerializedReference,
		convert_api_ServiceList_To_v1beta3_ServiceList,
		convert_api_ServicePort_To_v1beta3_ServicePort,
		convert_api_ServiceSpec_To_v1beta3_ServiceSpec,
		convert_api_ServiceStatus_To_v1beta3_ServiceStatus,
		convert_api_Service_To_v1beta3_Service,
		convert_api_StatusCause_To_v1beta3_StatusCause,
		convert_api_StatusDetails_To_v1beta3_StatusDetails,
		convert_api_Status_To_v1beta3_Status,
		convert_api_TCPSocketAction_To_v1beta3_TCPSocketAction,
		convert_api_TypeMeta_To_v1beta3_TypeMeta,
		convert_api_VolumeMount_To_v1beta3_VolumeMount,
		convert_api_VolumeSource_To_v1beta3_VolumeSource,
		convert_api_Volume_To_v1beta3_Volume,
		convert_v1beta3_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource,
		convert_v1beta3_Binding_To_api_Binding,
		convert_v1beta3_Capabilities_To_api_Capabilities,
		convert_v1beta3_ComponentCondition_To_api_ComponentCondition,
		convert_v1beta3_ComponentStatusList_To_api_ComponentStatusList,
		convert_v1beta3_ComponentStatus_To_api_ComponentStatus,
		convert_v1beta3_ContainerPort_To_api_ContainerPort,
		convert_v1beta3_ContainerStateRunning_To_api_ContainerStateRunning,
		convert_v1beta3_ContainerStateTerminated_To_api_ContainerStateTerminated,
		convert_v1beta3_ContainerStateWaiting_To_api_ContainerStateWaiting,
		convert_v1beta3_ContainerState_To_api_ContainerState,
		convert_v1beta3_ContainerStatus_To_api_ContainerStatus,
		convert_v1beta3_DeleteOptions_To_api_DeleteOptions,
		convert_v1beta3_EmptyDirVolumeSource_To_api_EmptyDirVolumeSource,
		convert_v1beta3_EndpointAddress_To_api_EndpointAddress,
		convert_v1beta3_EndpointPort_To_api_EndpointPort,
		convert_v1beta3_EndpointSubset_To_api_EndpointSubset,
		convert_v1beta3_EndpointsList_To_api_EndpointsList,
		convert_v1beta3_Endpoints_To_api_Endpoints,
		convert_v1beta3_EnvVarSource_To_api_EnvVarSource,
		convert_v1beta3_EnvVar_To_api_EnvVar,
		convert_v1beta3_EventList_To_api_EventList,
		convert_v1beta3_EventSource_To_api_EventSource,
		convert_v1beta3_Event_To_api_Event,
		convert_v1beta3_ExecAction_To_api_ExecAction,
		convert_v1beta3_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource,
		convert_v1beta3_GitRepoVolumeSource_To_api_GitRepoVolumeSource,
		convert_v1beta3_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource,
		convert_v1beta3_HTTPGetAction_To_api_HTTPGetAction,
		convert_v1beta3_Handler_To_api_Handler,
		convert_v1beta3_HostPathVolumeSource_To_api_HostPathVolumeSource,
		convert_v1beta3_ISCSIVolumeSource_To_api_ISCSIVolumeSource,
		convert_v1beta3_Lifecycle_To_api_Lifecycle,
		convert_v1beta3_LimitRangeItem_To_api_LimitRangeItem,
		convert_v1beta3_LimitRangeList_To_api_LimitRangeList,
		convert_v1beta3_LimitRangeSpec_To_api_LimitRangeSpec,
		convert_v1beta3_LimitRange_To_api_LimitRange,
		convert_v1beta3_ListMeta_To_api_ListMeta,
		convert_v1beta3_List_To_api_List,
		convert_v1beta3_NFSVolumeSource_To_api_NFSVolumeSource,
		convert_v1beta3_NamespaceList_To_api_NamespaceList,
		convert_v1beta3_NamespaceSpec_To_api_NamespaceSpec,
		convert_v1beta3_NamespaceStatus_To_api_NamespaceStatus,
		convert_v1beta3_Namespace_To_api_Namespace,
		convert_v1beta3_NodeAddress_To_api_NodeAddress,
		convert_v1beta3_NodeCondition_To_api_NodeCondition,
		convert_v1beta3_NodeList_To_api_NodeList,
		convert_v1beta3_NodeSpec_To_api_NodeSpec,
		convert_v1beta3_NodeStatus_To_api_NodeStatus,
		convert_v1beta3_NodeSystemInfo_To_api_NodeSystemInfo,
		convert_v1beta3_Node_To_api_Node,
		convert_v1beta3_ObjectFieldSelector_To_api_ObjectFieldSelector,
		convert_v1beta3_ObjectMeta_To_api_ObjectMeta,
		convert_v1beta3_ObjectReference_To_api_ObjectReference,
		convert_v1beta3_PersistentVolumeClaimList_To_api_PersistentVolumeClaimList,
		convert_v1beta3_PersistentVolumeClaimSpec_To_api_PersistentVolumeClaimSpec,
		convert_v1beta3_PersistentVolumeClaimStatus_To_api_PersistentVolumeClaimStatus,
		convert_v1beta3_PersistentVolumeClaimVolumeSource_To_api_PersistentVolumeClaimVolumeSource,
		convert_v1beta3_PersistentVolumeClaim_To_api_PersistentVolumeClaim,
		convert_v1beta3_PersistentVolumeList_To_api_PersistentVolumeList,
		convert_v1beta3_PersistentVolumeSource_To_api_PersistentVolumeSource,
		convert_v1beta3_PersistentVolumeSpec_To_api_PersistentVolumeSpec,
		convert_v1beta3_PersistentVolumeStatus_To_api_PersistentVolumeStatus,
		convert_v1beta3_PersistentVolume_To_api_PersistentVolume,
		convert_v1beta3_PodCondition_To_api_PodCondition,
		convert_v1beta3_PodExecOptions_To_api_PodExecOptions,
		convert_v1beta3_PodList_To_api_PodList,
		convert_v1beta3_PodLogOptions_To_api_PodLogOptions,
		convert_v1beta3_PodProxyOptions_To_api_PodProxyOptions,
		convert_v1beta3_PodSpec_To_api_PodSpec,
		convert_v1beta3_PodStatusResult_To_api_PodStatusResult,
		convert_v1beta3_PodStatus_To_api_PodStatus,
		convert_v1beta3_PodTemplateList_To_api_PodTemplateList,
		convert_v1beta3_PodTemplateSpec_To_api_PodTemplateSpec,
		convert_v1beta3_PodTemplate_To_api_PodTemplate,
		convert_v1beta3_Pod_To_api_Pod,
		convert_v1beta3_Probe_To_api_Probe,
		convert_v1beta3_ReplicationControllerList_To_api_ReplicationControllerList,
		convert_v1beta3_ReplicationControllerSpec_To_api_ReplicationControllerSpec,
		convert_v1beta3_ReplicationControllerStatus_To_api_ReplicationControllerStatus,
		convert_v1beta3_ReplicationController_To_api_ReplicationController,
		convert_v1beta3_ResourceQuotaList_To_api_ResourceQuotaList,
		convert_v1beta3_ResourceQuotaSpec_To_api_ResourceQuotaSpec,
		convert_v1beta3_ResourceQuotaStatus_To_api_ResourceQuotaStatus,
		convert_v1beta3_ResourceQuota_To_api_ResourceQuota,
		convert_v1beta3_ResourceRequirements_To_api_ResourceRequirements,
		convert_v1beta3_SELinuxOptions_To_api_SELinuxOptions,
		convert_v1beta3_SecretList_To_api_SecretList,
		convert_v1beta3_SecretVolumeSource_To_api_SecretVolumeSource,
		convert_v1beta3_Secret_To_api_Secret,
		convert_v1beta3_SecurityContext_To_api_SecurityContext,
		convert_v1beta3_SerializedReference_To_api_SerializedReference,
		convert_v1beta3_ServiceList_To_api_ServiceList,
		convert_v1beta3_ServicePort_To_api_ServicePort,
		convert_v1beta3_ServiceSpec_To_api_ServiceSpec,
		convert_v1beta3_ServiceStatus_To_api_ServiceStatus,
		convert_v1beta3_Service_To_api_Service,
		convert_v1beta3_StatusCause_To_api_StatusCause,
		convert_v1beta3_StatusDetails_To_api_StatusDetails,
		convert_v1beta3_Status_To_api_Status,
		convert_v1beta3_TCPSocketAction_To_api_TCPSocketAction,
		convert_v1beta3_TypeMeta_To_api_TypeMeta,
		convert_v1beta3_VolumeMount_To_api_VolumeMount,
		convert_v1beta3_VolumeSource_To_api_VolumeSource,
		convert_v1beta3_Volume_To_api_Volume,
	)

	// Add non-generated conversion functions
	newer.Scheme.AddConversionFuncs(
		convert_v1beta3_Container_To_api_Container,
		convert_api_Container_To_v1beta3_Container,
	)

	// Add field conversion funcs.
	err = newer.Scheme.AddFieldLabelConversionFunc("v1beta3", "Pod",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name",
				"metadata.namespace",
				"status.phase",
				"spec.host":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = newer.Scheme.AddFieldLabelConversionFunc("v1beta3", "Node",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name":
				return label, value, nil
			case "spec.unschedulable":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = newer.Scheme.AddFieldLabelConversionFunc("v1beta3", "ReplicationController",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name",
				"status.replicas":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = newer.Scheme.AddFieldLabelConversionFunc("v1beta3", "Event",
		func(label, value string) (string, string, error) {
			switch label {
			case "involvedObject.kind",
				"involvedObject.namespace",
				"involvedObject.name",
				"involvedObject.uid",
				"involvedObject.apiVersion",
				"involvedObject.resourceVersion",
				"involvedObject.fieldPath",
				"reason",
				"source":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = newer.Scheme.AddFieldLabelConversionFunc("v1beta3", "Namespace",
		func(label, value string) (string, string, error) {
			switch label {
			case "status.phase":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = newer.Scheme.AddFieldLabelConversionFunc("v1beta3", "Secret",
		func(label, value string) (string, string, error) {
			switch label {
			case "type":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}

func convert_v1beta3_Container_To_api_Container(in *Container, out *newer.Container, s conversion.Scope) error {
	out.Name = in.Name
	out.Image = in.Image
	if in.Command != nil {
		out.Command = make([]string, len(in.Command))
		for i := range in.Command {
			out.Command[i] = in.Command[i]
		}
	}
	if in.Args != nil {
		out.Args = make([]string, len(in.Args))
		for i := range in.Args {
			out.Args[i] = in.Args[i]
		}
	}
	out.WorkingDir = in.WorkingDir
	if in.Ports != nil {
		out.Ports = make([]newer.ContainerPort, len(in.Ports))
		for i := range in.Ports {
			if err := s.Convert(&in.Ports[i], &out.Ports[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Env != nil {
		out.Env = make([]newer.EnvVar, len(in.Env))
		for i := range in.Env {
			if err := s.Convert(&in.Env[i], &out.Env[i], 0); err != nil {
				return err
			}
		}
	}
	if err := s.Convert(&in.Resources, &out.Resources, 0); err != nil {
		return err
	}
	if in.VolumeMounts != nil {
		out.VolumeMounts = make([]newer.VolumeMount, len(in.VolumeMounts))
		for i := range in.VolumeMounts {
			if err := s.Convert(&in.VolumeMounts[i], &out.VolumeMounts[i], 0); err != nil {
				return err
			}
		}
	}
	if err := s.Convert(&in.LivenessProbe, &out.LivenessProbe, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ReadinessProbe, &out.ReadinessProbe, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Lifecycle, &out.Lifecycle, 0); err != nil {
		return err
	}
	out.TerminationMessagePath = in.TerminationMessagePath
	out.ImagePullPolicy = newer.PullPolicy(in.ImagePullPolicy)
	if in.SecurityContext != nil {
		if in.SecurityContext.Capabilities != nil {
			if !reflect.DeepEqual(in.SecurityContext.Capabilities.Add, in.Capabilities.Add) ||
				!reflect.DeepEqual(in.SecurityContext.Capabilities.Drop, in.Capabilities.Drop) {
				return fmt.Errorf("container capability settings do not match security context settings, cannot convert")
			}
		}
		if in.SecurityContext.Privileged != nil {
			if in.Privileged != *in.SecurityContext.Privileged {
				return fmt.Errorf("container privileged settings do not match security context settings, cannot convert")
			}
		}
	}
	if err := s.Convert(&in.SecurityContext, &out.SecurityContext, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_Container_To_v1beta3_Container(in *newer.Container, out *Container, s conversion.Scope) error {
	out.Name = in.Name
	out.Image = in.Image
	if in.Command != nil {
		out.Command = make([]string, len(in.Command))
		for i := range in.Command {
			out.Command[i] = in.Command[i]
		}
	}
	if in.Args != nil {
		out.Args = make([]string, len(in.Args))
		for i := range in.Args {
			out.Args[i] = in.Args[i]
		}
	}
	out.WorkingDir = in.WorkingDir
	if in.Ports != nil {
		out.Ports = make([]ContainerPort, len(in.Ports))
		for i := range in.Ports {
			if err := s.Convert(&in.Ports[i], &out.Ports[i], 0); err != nil {
				return err
			}
		}
	}
	if in.Env != nil {
		out.Env = make([]EnvVar, len(in.Env))
		for i := range in.Env {
			if err := s.Convert(&in.Env[i], &out.Env[i], 0); err != nil {
				return err
			}
		}
	}
	if err := s.Convert(&in.Resources, &out.Resources, 0); err != nil {
		return err
	}
	if in.VolumeMounts != nil {
		out.VolumeMounts = make([]VolumeMount, len(in.VolumeMounts))
		for i := range in.VolumeMounts {
			if err := s.Convert(&in.VolumeMounts[i], &out.VolumeMounts[i], 0); err != nil {
				return err
			}
		}
	}
	if err := s.Convert(&in.LivenessProbe, &out.LivenessProbe, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.ReadinessProbe, &out.ReadinessProbe, 0); err != nil {
		return err
	}
	if err := s.Convert(&in.Lifecycle, &out.Lifecycle, 0); err != nil {
		return err
	}
	out.TerminationMessagePath = in.TerminationMessagePath
	out.ImagePullPolicy = PullPolicy(in.ImagePullPolicy)
	if err := s.Convert(&in.SecurityContext, &out.SecurityContext, 0); err != nil {
		return err
	}
	// now that we've converted set the container field from security context
	if out.SecurityContext != nil && out.SecurityContext.Privileged != nil {
		out.Privileged = *out.SecurityContext.Privileged
	}
	// now that we've converted set the container field from security context
	if out.SecurityContext != nil && out.SecurityContext.Capabilities != nil {
		out.Capabilities = *out.SecurityContext.Capabilities
	}
	return nil
}
