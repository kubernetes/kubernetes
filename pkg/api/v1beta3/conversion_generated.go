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

package v1beta3

import (
	"reflect"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

// AUTO-GENERATED FUNCTIONS START HERE
func convert_v1beta3_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource(in *AWSElasticBlockStoreVolumeSource, out *newer.AWSElasticBlockStoreVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*AWSElasticBlockStoreVolumeSource))(in)
	}
	out.VolumeID = in.VolumeID
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_AWSElasticBlockStoreVolumeSource_To_v1beta3_AWSElasticBlockStoreVolumeSource(in *newer.AWSElasticBlockStoreVolumeSource, out *AWSElasticBlockStoreVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.AWSElasticBlockStoreVolumeSource))(in)
	}
	out.VolumeID = in.VolumeID
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_Binding_To_api_Binding(in *Binding, out *newer.Binding, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Binding))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectReference_To_api_ObjectReference(&in.Target, &out.Target, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Binding_To_v1beta3_Binding(in *newer.Binding, out *Binding, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Binding))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectReference_To_v1beta3_ObjectReference(&in.Target, &out.Target, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Capabilities_To_v1beta3_Capabilities(in *newer.Capabilities, out *Capabilities, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Capabilities))(in)
	}
	if in.Add != nil {
		out.Add = make([]CapabilityType, len(in.Add))
		for i := range in.Add {
			out.Add[i] = CapabilityType(in.Add[i])
		}
	} else {
		out.Add = nil
	}
	if in.Drop != nil {
		out.Drop = make([]CapabilityType, len(in.Drop))
		for i := range in.Drop {
			out.Drop[i] = CapabilityType(in.Drop[i])
		}
	} else {
		out.Drop = nil
	}
	return nil
}

func convert_v1beta3_Capabilities_To_api_Capabilities(in *Capabilities, out *newer.Capabilities, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Capabilities))(in)
	}
	if in.Add != nil {
		out.Add = make([]newer.CapabilityType, len(in.Add))
		for i := range in.Add {
			out.Add[i] = newer.CapabilityType(in.Add[i])
		}
	} else {
		out.Add = nil
	}
	if in.Drop != nil {
		out.Drop = make([]newer.CapabilityType, len(in.Drop))
		for i := range in.Drop {
			out.Drop[i] = newer.CapabilityType(in.Drop[i])
		}
	} else {
		out.Drop = nil
	}
	return nil
}

func convert_v1beta3_ComponentCondition_To_api_ComponentCondition(in *ComponentCondition, out *newer.ComponentCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ComponentCondition))(in)
	}
	out.Type = newer.ComponentConditionType(in.Type)
	out.Status = newer.ConditionStatus(in.Status)
	out.Message = in.Message
	out.Error = in.Error
	return nil
}

func convert_api_ComponentCondition_To_v1beta3_ComponentCondition(in *newer.ComponentCondition, out *ComponentCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ComponentCondition))(in)
	}
	out.Type = ComponentConditionType(in.Type)
	out.Status = ConditionStatus(in.Status)
	out.Message = in.Message
	out.Error = in.Error
	return nil
}

func convert_v1beta3_ComponentStatus_To_api_ComponentStatus(in *ComponentStatus, out *newer.ComponentStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ComponentStatus))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Conditions != nil {
		out.Conditions = make([]newer.ComponentCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_v1beta3_ComponentCondition_To_api_ComponentCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	return nil
}

func convert_api_ComponentStatus_To_v1beta3_ComponentStatus(in *newer.ComponentStatus, out *ComponentStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ComponentStatus))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Conditions != nil {
		out.Conditions = make([]ComponentCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_api_ComponentCondition_To_v1beta3_ComponentCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	return nil
}

func convert_v1beta3_ComponentStatusList_To_api_ComponentStatusList(in *ComponentStatusList, out *newer.ComponentStatusList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ComponentStatusList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.ComponentStatus, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_ComponentStatus_To_api_ComponentStatus(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ComponentStatusList_To_v1beta3_ComponentStatusList(in *newer.ComponentStatusList, out *ComponentStatusList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ComponentStatusList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ComponentStatus, len(in.Items))
		for i := range in.Items {
			if err := convert_api_ComponentStatus_To_v1beta3_ComponentStatus(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_ContainerPort_To_api_ContainerPort(in *ContainerPort, out *newer.ContainerPort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerPort))(in)
	}
	out.Name = in.Name
	out.HostPort = in.HostPort
	out.ContainerPort = in.ContainerPort
	out.Protocol = newer.Protocol(in.Protocol)
	out.HostIP = in.HostIP
	return nil
}

func convert_api_ContainerPort_To_v1beta3_ContainerPort(in *newer.ContainerPort, out *ContainerPort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ContainerPort))(in)
	}
	out.Name = in.Name
	out.HostPort = in.HostPort
	out.ContainerPort = in.ContainerPort
	out.Protocol = Protocol(in.Protocol)
	out.HostIP = in.HostIP
	return nil
}

func convert_v1beta3_ContainerState_To_api_ContainerState(in *ContainerState, out *newer.ContainerState, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerState))(in)
	}
	if in.Waiting != nil {
		out.Waiting = new(newer.ContainerStateWaiting)
		if err := convert_v1beta3_ContainerStateWaiting_To_api_ContainerStateWaiting(in.Waiting, out.Waiting, s); err != nil {
			return err
		}
	} else {
		out.Waiting = nil
	}
	if in.Running != nil {
		out.Running = new(newer.ContainerStateRunning)
		if err := convert_v1beta3_ContainerStateRunning_To_api_ContainerStateRunning(in.Running, out.Running, s); err != nil {
			return err
		}
	} else {
		out.Running = nil
	}
	if in.Termination != nil {
		out.Termination = new(newer.ContainerStateTerminated)
		if err := convert_v1beta3_ContainerStateTerminated_To_api_ContainerStateTerminated(in.Termination, out.Termination, s); err != nil {
			return err
		}
	} else {
		out.Termination = nil
	}
	return nil
}

func convert_api_ContainerState_To_v1beta3_ContainerState(in *newer.ContainerState, out *ContainerState, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ContainerState))(in)
	}
	if in.Waiting != nil {
		out.Waiting = new(ContainerStateWaiting)
		if err := convert_api_ContainerStateWaiting_To_v1beta3_ContainerStateWaiting(in.Waiting, out.Waiting, s); err != nil {
			return err
		}
	} else {
		out.Waiting = nil
	}
	if in.Running != nil {
		out.Running = new(ContainerStateRunning)
		if err := convert_api_ContainerStateRunning_To_v1beta3_ContainerStateRunning(in.Running, out.Running, s); err != nil {
			return err
		}
	} else {
		out.Running = nil
	}
	if in.Termination != nil {
		out.Termination = new(ContainerStateTerminated)
		if err := convert_api_ContainerStateTerminated_To_v1beta3_ContainerStateTerminated(in.Termination, out.Termination, s); err != nil {
			return err
		}
	} else {
		out.Termination = nil
	}
	return nil
}

func convert_v1beta3_ContainerStateRunning_To_api_ContainerStateRunning(in *ContainerStateRunning, out *newer.ContainerStateRunning, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerStateRunning))(in)
	}
	if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ContainerStateRunning_To_v1beta3_ContainerStateRunning(in *newer.ContainerStateRunning, out *ContainerStateRunning, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ContainerStateRunning))(in)
	}
	if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ContainerStateTerminated_To_api_ContainerStateTerminated(in *ContainerStateTerminated, out *newer.ContainerStateTerminated, s conversion.Scope) error {
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

func convert_api_ContainerStateTerminated_To_v1beta3_ContainerStateTerminated(in *newer.ContainerStateTerminated, out *ContainerStateTerminated, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ContainerStateTerminated))(in)
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

func convert_v1beta3_ContainerStateWaiting_To_api_ContainerStateWaiting(in *ContainerStateWaiting, out *newer.ContainerStateWaiting, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerStateWaiting))(in)
	}
	out.Reason = in.Reason
	return nil
}

func convert_api_ContainerStateWaiting_To_v1beta3_ContainerStateWaiting(in *newer.ContainerStateWaiting, out *ContainerStateWaiting, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ContainerStateWaiting))(in)
	}
	out.Reason = in.Reason
	return nil
}

func convert_v1beta3_ContainerStatus_To_api_ContainerStatus(in *ContainerStatus, out *newer.ContainerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ContainerStatus))(in)
	}
	out.Name = in.Name
	if err := convert_v1beta3_ContainerState_To_api_ContainerState(&in.State, &out.State, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ContainerState_To_api_ContainerState(&in.LastTerminationState, &out.LastTerminationState, s); err != nil {
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
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ContainerStatus))(in)
	}
	out.Name = in.Name
	if err := convert_api_ContainerState_To_v1beta3_ContainerState(&in.State, &out.State, s); err != nil {
		return err
	}
	if err := convert_api_ContainerState_To_v1beta3_ContainerState(&in.LastTerminationState, &out.LastTerminationState, s); err != nil {
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
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*DeleteOptions))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
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

func convert_api_DeleteOptions_To_v1beta3_DeleteOptions(in *newer.DeleteOptions, out *DeleteOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.DeleteOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
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

func convert_v1beta3_EmptyDirVolumeSource_To_api_EmptyDirVolumeSource(in *EmptyDirVolumeSource, out *newer.EmptyDirVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EmptyDirVolumeSource))(in)
	}
	out.Medium = newer.StorageType(in.Medium)
	return nil
}

func convert_api_EmptyDirVolumeSource_To_v1beta3_EmptyDirVolumeSource(in *newer.EmptyDirVolumeSource, out *EmptyDirVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EmptyDirVolumeSource))(in)
	}
	out.Medium = StorageType(in.Medium)
	return nil
}

func convert_v1beta3_EndpointAddress_To_api_EndpointAddress(in *EndpointAddress, out *newer.EndpointAddress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EndpointAddress))(in)
	}
	out.IP = in.IP
	if in.TargetRef != nil {
		out.TargetRef = new(newer.ObjectReference)
		if err := convert_v1beta3_ObjectReference_To_api_ObjectReference(in.TargetRef, out.TargetRef, s); err != nil {
			return err
		}
	} else {
		out.TargetRef = nil
	}
	return nil
}

func convert_api_EndpointAddress_To_v1beta3_EndpointAddress(in *newer.EndpointAddress, out *EndpointAddress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EndpointAddress))(in)
	}
	out.IP = in.IP
	if in.TargetRef != nil {
		out.TargetRef = new(ObjectReference)
		if err := convert_api_ObjectReference_To_v1beta3_ObjectReference(in.TargetRef, out.TargetRef, s); err != nil {
			return err
		}
	} else {
		out.TargetRef = nil
	}
	return nil
}

func convert_v1beta3_EndpointPort_To_api_EndpointPort(in *EndpointPort, out *newer.EndpointPort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EndpointPort))(in)
	}
	out.Name = in.Name
	out.Port = in.Port
	out.Protocol = newer.Protocol(in.Protocol)
	return nil
}

func convert_api_EndpointPort_To_v1beta3_EndpointPort(in *newer.EndpointPort, out *EndpointPort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EndpointPort))(in)
	}
	out.Name = in.Name
	out.Port = in.Port
	out.Protocol = Protocol(in.Protocol)
	return nil
}

func convert_v1beta3_EndpointSubset_To_api_EndpointSubset(in *EndpointSubset, out *newer.EndpointSubset, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EndpointSubset))(in)
	}
	if in.Addresses != nil {
		out.Addresses = make([]newer.EndpointAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := convert_v1beta3_EndpointAddress_To_api_EndpointAddress(&in.Addresses[i], &out.Addresses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if in.Ports != nil {
		out.Ports = make([]newer.EndpointPort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_v1beta3_EndpointPort_To_api_EndpointPort(&in.Ports[i], &out.Ports[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	return nil
}

func convert_api_EndpointSubset_To_v1beta3_EndpointSubset(in *newer.EndpointSubset, out *EndpointSubset, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EndpointSubset))(in)
	}
	if in.Addresses != nil {
		out.Addresses = make([]EndpointAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := convert_api_EndpointAddress_To_v1beta3_EndpointAddress(&in.Addresses[i], &out.Addresses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if in.Ports != nil {
		out.Ports = make([]EndpointPort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_api_EndpointPort_To_v1beta3_EndpointPort(&in.Ports[i], &out.Ports[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Ports = nil
	}
	return nil
}

func convert_v1beta3_Endpoints_To_api_Endpoints(in *Endpoints, out *newer.Endpoints, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Endpoints))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Subsets != nil {
		out.Subsets = make([]newer.EndpointSubset, len(in.Subsets))
		for i := range in.Subsets {
			if err := convert_v1beta3_EndpointSubset_To_api_EndpointSubset(&in.Subsets[i], &out.Subsets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Subsets = nil
	}
	return nil
}

func convert_api_Endpoints_To_v1beta3_Endpoints(in *newer.Endpoints, out *Endpoints, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Endpoints))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Subsets != nil {
		out.Subsets = make([]EndpointSubset, len(in.Subsets))
		for i := range in.Subsets {
			if err := convert_api_EndpointSubset_To_v1beta3_EndpointSubset(&in.Subsets[i], &out.Subsets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Subsets = nil
	}
	return nil
}

func convert_v1beta3_EndpointsList_To_api_EndpointsList(in *EndpointsList, out *newer.EndpointsList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EndpointsList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Endpoints, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_Endpoints_To_api_Endpoints(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_EndpointsList_To_v1beta3_EndpointsList(in *newer.EndpointsList, out *EndpointsList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EndpointsList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Endpoints, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Endpoints_To_v1beta3_Endpoints(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_EnvVar_To_api_EnvVar(in *EnvVar, out *newer.EnvVar, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EnvVar))(in)
	}
	out.Name = in.Name
	out.Value = in.Value
	if in.ValueFrom != nil {
		out.ValueFrom = new(newer.EnvVarSource)
		if err := convert_v1beta3_EnvVarSource_To_api_EnvVarSource(in.ValueFrom, out.ValueFrom, s); err != nil {
			return err
		}
	} else {
		out.ValueFrom = nil
	}
	return nil
}

func convert_api_EnvVar_To_v1beta3_EnvVar(in *newer.EnvVar, out *EnvVar, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EnvVar))(in)
	}
	out.Name = in.Name
	out.Value = in.Value
	if in.ValueFrom != nil {
		out.ValueFrom = new(EnvVarSource)
		if err := convert_api_EnvVarSource_To_v1beta3_EnvVarSource(in.ValueFrom, out.ValueFrom, s); err != nil {
			return err
		}
	} else {
		out.ValueFrom = nil
	}
	return nil
}

func convert_v1beta3_EnvVarSource_To_api_EnvVarSource(in *EnvVarSource, out *newer.EnvVarSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EnvVarSource))(in)
	}
	if in.FieldRef != nil {
		out.FieldRef = new(newer.ObjectFieldSelector)
		if err := convert_v1beta3_ObjectFieldSelector_To_api_ObjectFieldSelector(in.FieldRef, out.FieldRef, s); err != nil {
			return err
		}
	} else {
		out.FieldRef = nil
	}
	return nil
}

func convert_api_EnvVarSource_To_v1beta3_EnvVarSource(in *newer.EnvVarSource, out *EnvVarSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EnvVarSource))(in)
	}
	if in.FieldRef != nil {
		out.FieldRef = new(ObjectFieldSelector)
		if err := convert_api_ObjectFieldSelector_To_v1beta3_ObjectFieldSelector(in.FieldRef, out.FieldRef, s); err != nil {
			return err
		}
	} else {
		out.FieldRef = nil
	}
	return nil
}

func convert_v1beta3_Event_To_api_Event(in *Event, out *newer.Event, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Event))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectReference_To_api_ObjectReference(&in.InvolvedObject, &out.InvolvedObject, s); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	if err := convert_v1beta3_EventSource_To_api_EventSource(&in.Source, &out.Source, s); err != nil {
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
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Event))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectReference_To_v1beta3_ObjectReference(&in.InvolvedObject, &out.InvolvedObject, s); err != nil {
		return err
	}
	out.Reason = in.Reason
	out.Message = in.Message
	if err := convert_api_EventSource_To_v1beta3_EventSource(&in.Source, &out.Source, s); err != nil {
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
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EventList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Event, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_Event_To_api_Event(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_EventList_To_v1beta3_EventList(in *newer.EventList, out *EventList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EventList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Event, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Event_To_v1beta3_Event(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_EventSource_To_api_EventSource(in *EventSource, out *newer.EventSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*EventSource))(in)
	}
	out.Component = in.Component
	out.Host = in.Host
	return nil
}

func convert_api_EventSource_To_v1beta3_EventSource(in *newer.EventSource, out *EventSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.EventSource))(in)
	}
	out.Component = in.Component
	out.Host = in.Host
	return nil
}

func convert_v1beta3_ExecAction_To_api_ExecAction(in *ExecAction, out *newer.ExecAction, s conversion.Scope) error {
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

func convert_api_ExecAction_To_v1beta3_ExecAction(in *newer.ExecAction, out *ExecAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ExecAction))(in)
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

func convert_v1beta3_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource(in *GCEPersistentDiskVolumeSource, out *newer.GCEPersistentDiskVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*GCEPersistentDiskVolumeSource))(in)
	}
	out.PDName = in.PDName
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_GCEPersistentDiskVolumeSource_To_v1beta3_GCEPersistentDiskVolumeSource(in *newer.GCEPersistentDiskVolumeSource, out *GCEPersistentDiskVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.GCEPersistentDiskVolumeSource))(in)
	}
	out.PDName = in.PDName
	out.FSType = in.FSType
	out.Partition = in.Partition
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_GitRepoVolumeSource_To_api_GitRepoVolumeSource(in *GitRepoVolumeSource, out *newer.GitRepoVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*GitRepoVolumeSource))(in)
	}
	out.Repository = in.Repository
	out.Revision = in.Revision
	return nil
}

func convert_api_GitRepoVolumeSource_To_v1beta3_GitRepoVolumeSource(in *newer.GitRepoVolumeSource, out *GitRepoVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.GitRepoVolumeSource))(in)
	}
	out.Repository = in.Repository
	out.Revision = in.Revision
	return nil
}

func convert_v1beta3_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource(in *GlusterfsVolumeSource, out *newer.GlusterfsVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*GlusterfsVolumeSource))(in)
	}
	out.EndpointsName = in.EndpointsName
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_GlusterfsVolumeSource_To_v1beta3_GlusterfsVolumeSource(in *newer.GlusterfsVolumeSource, out *GlusterfsVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.GlusterfsVolumeSource))(in)
	}
	out.EndpointsName = in.EndpointsName
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_HTTPGetAction_To_api_HTTPGetAction(in *HTTPGetAction, out *newer.HTTPGetAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*HTTPGetAction))(in)
	}
	out.Path = in.Path
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	out.Host = in.Host
	return nil
}

func convert_api_HTTPGetAction_To_v1beta3_HTTPGetAction(in *newer.HTTPGetAction, out *HTTPGetAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.HTTPGetAction))(in)
	}
	out.Path = in.Path
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	out.Host = in.Host
	return nil
}

func convert_v1beta3_Handler_To_api_Handler(in *Handler, out *newer.Handler, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Handler))(in)
	}
	if in.Exec != nil {
		out.Exec = new(newer.ExecAction)
		if err := convert_v1beta3_ExecAction_To_api_ExecAction(in.Exec, out.Exec, s); err != nil {
			return err
		}
	} else {
		out.Exec = nil
	}
	if in.HTTPGet != nil {
		out.HTTPGet = new(newer.HTTPGetAction)
		if err := convert_v1beta3_HTTPGetAction_To_api_HTTPGetAction(in.HTTPGet, out.HTTPGet, s); err != nil {
			return err
		}
	} else {
		out.HTTPGet = nil
	}
	if in.TCPSocket != nil {
		out.TCPSocket = new(newer.TCPSocketAction)
		if err := convert_v1beta3_TCPSocketAction_To_api_TCPSocketAction(in.TCPSocket, out.TCPSocket, s); err != nil {
			return err
		}
	} else {
		out.TCPSocket = nil
	}
	return nil
}

func convert_api_Handler_To_v1beta3_Handler(in *newer.Handler, out *Handler, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Handler))(in)
	}
	if in.Exec != nil {
		out.Exec = new(ExecAction)
		if err := convert_api_ExecAction_To_v1beta3_ExecAction(in.Exec, out.Exec, s); err != nil {
			return err
		}
	} else {
		out.Exec = nil
	}
	if in.HTTPGet != nil {
		out.HTTPGet = new(HTTPGetAction)
		if err := convert_api_HTTPGetAction_To_v1beta3_HTTPGetAction(in.HTTPGet, out.HTTPGet, s); err != nil {
			return err
		}
	} else {
		out.HTTPGet = nil
	}
	if in.TCPSocket != nil {
		out.TCPSocket = new(TCPSocketAction)
		if err := convert_api_TCPSocketAction_To_v1beta3_TCPSocketAction(in.TCPSocket, out.TCPSocket, s); err != nil {
			return err
		}
	} else {
		out.TCPSocket = nil
	}
	return nil
}

func convert_v1beta3_HostPathVolumeSource_To_api_HostPathVolumeSource(in *HostPathVolumeSource, out *newer.HostPathVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*HostPathVolumeSource))(in)
	}
	out.Path = in.Path
	return nil
}

func convert_api_HostPathVolumeSource_To_v1beta3_HostPathVolumeSource(in *newer.HostPathVolumeSource, out *HostPathVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.HostPathVolumeSource))(in)
	}
	out.Path = in.Path
	return nil
}

func convert_v1beta3_ISCSIVolumeSource_To_api_ISCSIVolumeSource(in *ISCSIVolumeSource, out *newer.ISCSIVolumeSource, s conversion.Scope) error {
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

func convert_api_ISCSIVolumeSource_To_v1beta3_ISCSIVolumeSource(in *newer.ISCSIVolumeSource, out *ISCSIVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ISCSIVolumeSource))(in)
	}
	out.TargetPortal = in.TargetPortal
	out.IQN = in.IQN
	out.Lun = in.Lun
	out.FSType = in.FSType
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_Lifecycle_To_api_Lifecycle(in *Lifecycle, out *newer.Lifecycle, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Lifecycle))(in)
	}
	if in.PostStart != nil {
		out.PostStart = new(newer.Handler)
		if err := convert_v1beta3_Handler_To_api_Handler(in.PostStart, out.PostStart, s); err != nil {
			return err
		}
	} else {
		out.PostStart = nil
	}
	if in.PreStop != nil {
		out.PreStop = new(newer.Handler)
		if err := convert_v1beta3_Handler_To_api_Handler(in.PreStop, out.PreStop, s); err != nil {
			return err
		}
	} else {
		out.PreStop = nil
	}
	return nil
}

func convert_api_Lifecycle_To_v1beta3_Lifecycle(in *newer.Lifecycle, out *Lifecycle, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Lifecycle))(in)
	}
	if in.PostStart != nil {
		out.PostStart = new(Handler)
		if err := convert_api_Handler_To_v1beta3_Handler(in.PostStart, out.PostStart, s); err != nil {
			return err
		}
	} else {
		out.PostStart = nil
	}
	if in.PreStop != nil {
		out.PreStop = new(Handler)
		if err := convert_api_Handler_To_v1beta3_Handler(in.PreStop, out.PreStop, s); err != nil {
			return err
		}
	} else {
		out.PreStop = nil
	}
	return nil
}

func convert_v1beta3_LimitRange_To_api_LimitRange(in *LimitRange, out *newer.LimitRange, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LimitRange))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_LimitRangeSpec_To_api_LimitRangeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	return nil
}

func convert_api_LimitRange_To_v1beta3_LimitRange(in *newer.LimitRange, out *LimitRange, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.LimitRange))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_LimitRangeSpec_To_v1beta3_LimitRangeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_LimitRangeItem_To_api_LimitRangeItem(in *LimitRangeItem, out *newer.LimitRangeItem, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LimitRangeItem))(in)
	}
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
	} else {
		out.Max = nil
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
	} else {
		out.Min = nil
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
	} else {
		out.Default = nil
	}
	return nil
}

func convert_api_LimitRangeItem_To_v1beta3_LimitRangeItem(in *newer.LimitRangeItem, out *LimitRangeItem, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.LimitRangeItem))(in)
	}
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
	} else {
		out.Max = nil
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
	} else {
		out.Min = nil
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
	} else {
		out.Default = nil
	}
	return nil
}

func convert_v1beta3_LimitRangeList_To_api_LimitRangeList(in *LimitRangeList, out *newer.LimitRangeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LimitRangeList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.LimitRange, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_LimitRange_To_api_LimitRange(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_LimitRangeList_To_v1beta3_LimitRangeList(in *newer.LimitRangeList, out *LimitRangeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.LimitRangeList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]LimitRange, len(in.Items))
		for i := range in.Items {
			if err := convert_api_LimitRange_To_v1beta3_LimitRange(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_LimitRangeSpec_To_api_LimitRangeSpec(in *LimitRangeSpec, out *newer.LimitRangeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*LimitRangeSpec))(in)
	}
	if in.Limits != nil {
		out.Limits = make([]newer.LimitRangeItem, len(in.Limits))
		for i := range in.Limits {
			if err := convert_v1beta3_LimitRangeItem_To_api_LimitRangeItem(&in.Limits[i], &out.Limits[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Limits = nil
	}
	return nil
}

func convert_api_LimitRangeSpec_To_v1beta3_LimitRangeSpec(in *newer.LimitRangeSpec, out *LimitRangeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.LimitRangeSpec))(in)
	}
	if in.Limits != nil {
		out.Limits = make([]LimitRangeItem, len(in.Limits))
		for i := range in.Limits {
			if err := convert_api_LimitRangeItem_To_v1beta3_LimitRangeItem(&in.Limits[i], &out.Limits[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Limits = nil
	}
	return nil
}

func convert_v1beta3_List_To_api_List(in *List, out *newer.List, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*List))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_List_To_v1beta3_List(in *newer.List, out *List, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.List))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ListMeta_To_api_ListMeta(in *ListMeta, out *newer.ListMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ListMeta))(in)
	}
	out.SelfLink = in.SelfLink
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func convert_api_ListMeta_To_v1beta3_ListMeta(in *newer.ListMeta, out *ListMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ListMeta))(in)
	}
	out.SelfLink = in.SelfLink
	out.ResourceVersion = in.ResourceVersion
	return nil
}

func convert_v1beta3_NFSVolumeSource_To_api_NFSVolumeSource(in *NFSVolumeSource, out *newer.NFSVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NFSVolumeSource))(in)
	}
	out.Server = in.Server
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_NFSVolumeSource_To_v1beta3_NFSVolumeSource(in *newer.NFSVolumeSource, out *NFSVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NFSVolumeSource))(in)
	}
	out.Server = in.Server
	out.Path = in.Path
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_Namespace_To_api_Namespace(in *Namespace, out *newer.Namespace, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Namespace))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_NamespaceSpec_To_api_NamespaceSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1beta3_NamespaceStatus_To_api_NamespaceStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Namespace_To_v1beta3_Namespace(in *newer.Namespace, out *Namespace, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Namespace))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_NamespaceSpec_To_v1beta3_NamespaceSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_NamespaceStatus_To_v1beta3_NamespaceStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_NamespaceList_To_api_NamespaceList(in *NamespaceList, out *newer.NamespaceList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NamespaceList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Namespace, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_Namespace_To_api_Namespace(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_NamespaceList_To_v1beta3_NamespaceList(in *newer.NamespaceList, out *NamespaceList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NamespaceList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Namespace, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Namespace_To_v1beta3_Namespace(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_NamespaceSpec_To_api_NamespaceSpec(in *NamespaceSpec, out *newer.NamespaceSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NamespaceSpec))(in)
	}
	if in.Finalizers != nil {
		out.Finalizers = make([]newer.FinalizerName, len(in.Finalizers))
		for i := range in.Finalizers {
			out.Finalizers[i] = newer.FinalizerName(in.Finalizers[i])
		}
	} else {
		out.Finalizers = nil
	}
	return nil
}

func convert_api_NamespaceSpec_To_v1beta3_NamespaceSpec(in *newer.NamespaceSpec, out *NamespaceSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NamespaceSpec))(in)
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

func convert_v1beta3_NamespaceStatus_To_api_NamespaceStatus(in *NamespaceStatus, out *newer.NamespaceStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NamespaceStatus))(in)
	}
	out.Phase = newer.NamespacePhase(in.Phase)
	return nil
}

func convert_api_NamespaceStatus_To_v1beta3_NamespaceStatus(in *newer.NamespaceStatus, out *NamespaceStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NamespaceStatus))(in)
	}
	out.Phase = NamespacePhase(in.Phase)
	return nil
}

func convert_v1beta3_Node_To_api_Node(in *Node, out *newer.Node, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Node))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_NodeSpec_To_api_NodeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1beta3_NodeStatus_To_api_NodeStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Node_To_v1beta3_Node(in *newer.Node, out *Node, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Node))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_NodeSpec_To_v1beta3_NodeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_NodeStatus_To_v1beta3_NodeStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_NodeAddress_To_api_NodeAddress(in *NodeAddress, out *newer.NodeAddress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeAddress))(in)
	}
	out.Type = newer.NodeAddressType(in.Type)
	out.Address = in.Address
	return nil
}

func convert_api_NodeAddress_To_v1beta3_NodeAddress(in *newer.NodeAddress, out *NodeAddress, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NodeAddress))(in)
	}
	out.Type = NodeAddressType(in.Type)
	out.Address = in.Address
	return nil
}

func convert_v1beta3_NodeCondition_To_api_NodeCondition(in *NodeCondition, out *newer.NodeCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeCondition))(in)
	}
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
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NodeCondition))(in)
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

func convert_v1beta3_NodeList_To_api_NodeList(in *NodeList, out *newer.NodeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Node, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_Node_To_api_Node(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_NodeList_To_v1beta3_NodeList(in *newer.NodeList, out *NodeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NodeList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Node, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Node_To_v1beta3_Node(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_NodeSpec_To_api_NodeSpec(in *NodeSpec, out *newer.NodeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeSpec))(in)
	}
	out.PodCIDR = in.PodCIDR
	out.ExternalID = in.ExternalID
	out.Unschedulable = in.Unschedulable
	return nil
}

func convert_api_NodeSpec_To_v1beta3_NodeSpec(in *newer.NodeSpec, out *NodeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NodeSpec))(in)
	}
	out.PodCIDR = in.PodCIDR
	out.ExternalID = in.ExternalID
	out.Unschedulable = in.Unschedulable
	return nil
}

func convert_v1beta3_NodeStatus_To_api_NodeStatus(in *NodeStatus, out *newer.NodeStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*NodeStatus))(in)
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
	} else {
		out.Capacity = nil
	}
	out.Phase = newer.NodePhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]newer.NodeCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_v1beta3_NodeCondition_To_api_NodeCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	if in.Addresses != nil {
		out.Addresses = make([]newer.NodeAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := convert_v1beta3_NodeAddress_To_api_NodeAddress(&in.Addresses[i], &out.Addresses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if err := convert_v1beta3_NodeSystemInfo_To_api_NodeSystemInfo(&in.NodeInfo, &out.NodeInfo, s); err != nil {
		return err
	}
	return nil
}

func convert_api_NodeStatus_To_v1beta3_NodeStatus(in *newer.NodeStatus, out *NodeStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NodeStatus))(in)
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
	} else {
		out.Capacity = nil
	}
	out.Phase = NodePhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]NodeCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_api_NodeCondition_To_v1beta3_NodeCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	if in.Addresses != nil {
		out.Addresses = make([]NodeAddress, len(in.Addresses))
		for i := range in.Addresses {
			if err := convert_api_NodeAddress_To_v1beta3_NodeAddress(&in.Addresses[i], &out.Addresses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Addresses = nil
	}
	if err := convert_api_NodeSystemInfo_To_v1beta3_NodeSystemInfo(&in.NodeInfo, &out.NodeInfo, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_NodeSystemInfo_To_api_NodeSystemInfo(in *NodeSystemInfo, out *newer.NodeSystemInfo, s conversion.Scope) error {
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

func convert_api_NodeSystemInfo_To_v1beta3_NodeSystemInfo(in *newer.NodeSystemInfo, out *NodeSystemInfo, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.NodeSystemInfo))(in)
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

func convert_v1beta3_ObjectFieldSelector_To_api_ObjectFieldSelector(in *ObjectFieldSelector, out *newer.ObjectFieldSelector, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ObjectFieldSelector))(in)
	}
	out.APIVersion = in.APIVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_api_ObjectFieldSelector_To_v1beta3_ObjectFieldSelector(in *newer.ObjectFieldSelector, out *ObjectFieldSelector, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ObjectFieldSelector))(in)
	}
	out.APIVersion = in.APIVersion
	out.FieldPath = in.FieldPath
	return nil
}

func convert_v1beta3_ObjectMeta_To_api_ObjectMeta(in *ObjectMeta, out *newer.ObjectMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ObjectMeta))(in)
	}
	out.Name = in.Name
	out.GenerateName = in.GenerateName
	out.Namespace = in.Namespace
	out.SelfLink = in.SelfLink
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
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

func convert_api_ObjectMeta_To_v1beta3_ObjectMeta(in *newer.ObjectMeta, out *ObjectMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ObjectMeta))(in)
	}
	out.Name = in.Name
	out.GenerateName = in.GenerateName
	out.Namespace = in.Namespace
	out.SelfLink = in.SelfLink
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
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

func convert_v1beta3_ObjectReference_To_api_ObjectReference(in *ObjectReference, out *newer.ObjectReference, s conversion.Scope) error {
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

func convert_api_ObjectReference_To_v1beta3_ObjectReference(in *newer.ObjectReference, out *ObjectReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ObjectReference))(in)
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

func convert_v1beta3_PersistentVolume_To_api_PersistentVolume(in *PersistentVolume, out *newer.PersistentVolume, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolume))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PersistentVolumeSpec_To_api_PersistentVolumeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PersistentVolumeStatus_To_api_PersistentVolumeStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolume_To_v1beta3_PersistentVolume(in *newer.PersistentVolume, out *PersistentVolume, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolume))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PersistentVolumeSpec_To_v1beta3_PersistentVolumeSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_PersistentVolumeStatus_To_v1beta3_PersistentVolumeStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaim_To_api_PersistentVolumeClaim(in *PersistentVolumeClaim, out *newer.PersistentVolumeClaim, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaim))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PersistentVolumeClaimSpec_To_api_PersistentVolumeClaimSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PersistentVolumeClaimStatus_To_api_PersistentVolumeClaimStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeClaim_To_v1beta3_PersistentVolumeClaim(in *newer.PersistentVolumeClaim, out *PersistentVolumeClaim, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeClaim))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PersistentVolumeClaimSpec_To_v1beta3_PersistentVolumeClaimSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_PersistentVolumeClaimStatus_To_v1beta3_PersistentVolumeClaimStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaimList_To_api_PersistentVolumeClaimList(in *PersistentVolumeClaimList, out *newer.PersistentVolumeClaimList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaimList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.PersistentVolumeClaim, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_PersistentVolumeClaim_To_api_PersistentVolumeClaim(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_PersistentVolumeClaimList_To_v1beta3_PersistentVolumeClaimList(in *newer.PersistentVolumeClaimList, out *PersistentVolumeClaimList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeClaimList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PersistentVolumeClaim, len(in.Items))
		for i := range in.Items {
			if err := convert_api_PersistentVolumeClaim_To_v1beta3_PersistentVolumeClaim(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaimSpec_To_api_PersistentVolumeClaimSpec(in *PersistentVolumeClaimSpec, out *newer.PersistentVolumeClaimSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaimSpec))(in)
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]newer.AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = newer.AccessModeType(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if err := convert_v1beta3_ResourceRequirements_To_api_ResourceRequirements(&in.Resources, &out.Resources, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PersistentVolumeClaimSpec_To_v1beta3_PersistentVolumeClaimSpec(in *newer.PersistentVolumeClaimSpec, out *PersistentVolumeClaimSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeClaimSpec))(in)
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = AccessModeType(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if err := convert_api_ResourceRequirements_To_v1beta3_ResourceRequirements(&in.Resources, &out.Resources, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaimStatus_To_api_PersistentVolumeClaimStatus(in *PersistentVolumeClaimStatus, out *newer.PersistentVolumeClaimStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaimStatus))(in)
	}
	out.Phase = newer.PersistentVolumeClaimPhase(in.Phase)
	if in.AccessModes != nil {
		out.AccessModes = make([]newer.AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = newer.AccessModeType(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
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
	} else {
		out.Capacity = nil
	}
	if in.VolumeRef != nil {
		out.VolumeRef = new(newer.ObjectReference)
		if err := convert_v1beta3_ObjectReference_To_api_ObjectReference(in.VolumeRef, out.VolumeRef, s); err != nil {
			return err
		}
	} else {
		out.VolumeRef = nil
	}
	return nil
}

func convert_api_PersistentVolumeClaimStatus_To_v1beta3_PersistentVolumeClaimStatus(in *newer.PersistentVolumeClaimStatus, out *PersistentVolumeClaimStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeClaimStatus))(in)
	}
	out.Phase = PersistentVolumeClaimPhase(in.Phase)
	if in.AccessModes != nil {
		out.AccessModes = make([]AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = AccessModeType(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
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
	} else {
		out.Capacity = nil
	}
	if in.VolumeRef != nil {
		out.VolumeRef = new(ObjectReference)
		if err := convert_api_ObjectReference_To_v1beta3_ObjectReference(in.VolumeRef, out.VolumeRef, s); err != nil {
			return err
		}
	} else {
		out.VolumeRef = nil
	}
	return nil
}

func convert_v1beta3_PersistentVolumeClaimVolumeSource_To_api_PersistentVolumeClaimVolumeSource(in *PersistentVolumeClaimVolumeSource, out *newer.PersistentVolumeClaimVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeClaimVolumeSource))(in)
	}
	out.ClaimName = in.ClaimName
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_api_PersistentVolumeClaimVolumeSource_To_v1beta3_PersistentVolumeClaimVolumeSource(in *newer.PersistentVolumeClaimVolumeSource, out *PersistentVolumeClaimVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeClaimVolumeSource))(in)
	}
	out.ClaimName = in.ClaimName
	out.ReadOnly = in.ReadOnly
	return nil
}

func convert_v1beta3_PersistentVolumeList_To_api_PersistentVolumeList(in *PersistentVolumeList, out *newer.PersistentVolumeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.PersistentVolume, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_PersistentVolume_To_api_PersistentVolume(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_PersistentVolumeList_To_v1beta3_PersistentVolumeList(in *newer.PersistentVolumeList, out *PersistentVolumeList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PersistentVolume, len(in.Items))
		for i := range in.Items {
			if err := convert_api_PersistentVolume_To_v1beta3_PersistentVolume(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_PersistentVolumeSource_To_api_PersistentVolumeSource(in *PersistentVolumeSource, out *newer.PersistentVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeSource))(in)
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(newer.GCEPersistentDiskVolumeSource)
		if err := convert_v1beta3_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource(in.GCEPersistentDisk, out.GCEPersistentDisk, s); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(newer.AWSElasticBlockStoreVolumeSource)
		if err := convert_v1beta3_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource(in.AWSElasticBlockStore, out.AWSElasticBlockStore, s); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.HostPath != nil {
		out.HostPath = new(newer.HostPathVolumeSource)
		if err := convert_v1beta3_HostPathVolumeSource_To_api_HostPathVolumeSource(in.HostPath, out.HostPath, s); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(newer.GlusterfsVolumeSource)
		if err := convert_v1beta3_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource(in.Glusterfs, out.Glusterfs, s); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.NFS != nil {
		out.NFS = new(newer.NFSVolumeSource)
		if err := convert_v1beta3_NFSVolumeSource_To_api_NFSVolumeSource(in.NFS, out.NFS, s); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	return nil
}

func convert_api_PersistentVolumeSource_To_v1beta3_PersistentVolumeSource(in *newer.PersistentVolumeSource, out *PersistentVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeSource))(in)
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(GCEPersistentDiskVolumeSource)
		if err := convert_api_GCEPersistentDiskVolumeSource_To_v1beta3_GCEPersistentDiskVolumeSource(in.GCEPersistentDisk, out.GCEPersistentDisk, s); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(AWSElasticBlockStoreVolumeSource)
		if err := convert_api_AWSElasticBlockStoreVolumeSource_To_v1beta3_AWSElasticBlockStoreVolumeSource(in.AWSElasticBlockStore, out.AWSElasticBlockStore, s); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.HostPath != nil {
		out.HostPath = new(HostPathVolumeSource)
		if err := convert_api_HostPathVolumeSource_To_v1beta3_HostPathVolumeSource(in.HostPath, out.HostPath, s); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(GlusterfsVolumeSource)
		if err := convert_api_GlusterfsVolumeSource_To_v1beta3_GlusterfsVolumeSource(in.Glusterfs, out.Glusterfs, s); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.NFS != nil {
		out.NFS = new(NFSVolumeSource)
		if err := convert_api_NFSVolumeSource_To_v1beta3_NFSVolumeSource(in.NFS, out.NFS, s); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	return nil
}

func convert_v1beta3_PersistentVolumeSpec_To_api_PersistentVolumeSpec(in *PersistentVolumeSpec, out *newer.PersistentVolumeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeSpec))(in)
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
	} else {
		out.Capacity = nil
	}
	if err := convert_v1beta3_PersistentVolumeSource_To_api_PersistentVolumeSource(&in.PersistentVolumeSource, &out.PersistentVolumeSource, s); err != nil {
		return err
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]newer.AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = newer.AccessModeType(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if in.ClaimRef != nil {
		out.ClaimRef = new(newer.ObjectReference)
		if err := convert_v1beta3_ObjectReference_To_api_ObjectReference(in.ClaimRef, out.ClaimRef, s); err != nil {
			return err
		}
	} else {
		out.ClaimRef = nil
	}
	return nil
}

func convert_api_PersistentVolumeSpec_To_v1beta3_PersistentVolumeSpec(in *newer.PersistentVolumeSpec, out *PersistentVolumeSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeSpec))(in)
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
	} else {
		out.Capacity = nil
	}
	if err := convert_api_PersistentVolumeSource_To_v1beta3_PersistentVolumeSource(&in.PersistentVolumeSource, &out.PersistentVolumeSource, s); err != nil {
		return err
	}
	if in.AccessModes != nil {
		out.AccessModes = make([]AccessModeType, len(in.AccessModes))
		for i := range in.AccessModes {
			out.AccessModes[i] = AccessModeType(in.AccessModes[i])
		}
	} else {
		out.AccessModes = nil
	}
	if in.ClaimRef != nil {
		out.ClaimRef = new(ObjectReference)
		if err := convert_api_ObjectReference_To_v1beta3_ObjectReference(in.ClaimRef, out.ClaimRef, s); err != nil {
			return err
		}
	} else {
		out.ClaimRef = nil
	}
	return nil
}

func convert_v1beta3_PersistentVolumeStatus_To_api_PersistentVolumeStatus(in *PersistentVolumeStatus, out *newer.PersistentVolumeStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PersistentVolumeStatus))(in)
	}
	out.Phase = newer.PersistentVolumePhase(in.Phase)
	return nil
}

func convert_api_PersistentVolumeStatus_To_v1beta3_PersistentVolumeStatus(in *newer.PersistentVolumeStatus, out *PersistentVolumeStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PersistentVolumeStatus))(in)
	}
	out.Phase = PersistentVolumePhase(in.Phase)
	return nil
}

func convert_v1beta3_Pod_To_api_Pod(in *Pod, out *newer.Pod, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Pod))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PodSpec_To_api_PodSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PodStatus_To_api_PodStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Pod_To_v1beta3_Pod(in *newer.Pod, out *Pod, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Pod))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PodSpec_To_v1beta3_PodSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_PodStatus_To_v1beta3_PodStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PodCondition_To_api_PodCondition(in *PodCondition, out *newer.PodCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodCondition))(in)
	}
	out.Type = newer.PodConditionType(in.Type)
	out.Status = newer.ConditionStatus(in.Status)
	return nil
}

func convert_api_PodCondition_To_v1beta3_PodCondition(in *newer.PodCondition, out *PodCondition, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodCondition))(in)
	}
	out.Type = PodConditionType(in.Type)
	out.Status = ConditionStatus(in.Status)
	return nil
}

func convert_v1beta3_PodExecOptions_To_api_PodExecOptions(in *PodExecOptions, out *newer.PodExecOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodExecOptions))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
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

func convert_api_PodExecOptions_To_v1beta3_PodExecOptions(in *newer.PodExecOptions, out *PodExecOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodExecOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
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

func convert_v1beta3_PodList_To_api_PodList(in *PodList, out *newer.PodList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Pod, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_Pod_To_api_Pod(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_PodList_To_v1beta3_PodList(in *newer.PodList, out *PodList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Pod, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Pod_To_v1beta3_Pod(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_PodLogOptions_To_api_PodLogOptions(in *PodLogOptions, out *newer.PodLogOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodLogOptions))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Container = in.Container
	out.Follow = in.Follow
	out.Previous = in.Previous
	return nil
}

func convert_api_PodLogOptions_To_v1beta3_PodLogOptions(in *newer.PodLogOptions, out *PodLogOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodLogOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Container = in.Container
	out.Follow = in.Follow
	out.Previous = in.Previous
	return nil
}

func convert_v1beta3_PodProxyOptions_To_api_PodProxyOptions(in *PodProxyOptions, out *newer.PodProxyOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodProxyOptions))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Path = in.Path
	return nil
}

func convert_api_PodProxyOptions_To_v1beta3_PodProxyOptions(in *newer.PodProxyOptions, out *PodProxyOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodProxyOptions))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	out.Path = in.Path
	return nil
}

func convert_v1beta3_PodSpec_To_api_PodSpec(in *PodSpec, out *newer.PodSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodSpec))(in)
	}
	if in.Volumes != nil {
		out.Volumes = make([]newer.Volume, len(in.Volumes))
		for i := range in.Volumes {
			if err := convert_v1beta3_Volume_To_api_Volume(&in.Volumes[i], &out.Volumes[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Volumes = nil
	}
	if in.Containers != nil {
		out.Containers = make([]newer.Container, len(in.Containers))
		for i := range in.Containers {
			if err := convert_v1beta3_Container_To_api_Container(&in.Containers[i], &out.Containers[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Containers = nil
	}
	out.RestartPolicy = newer.RestartPolicy(in.RestartPolicy)
	if in.TerminationGracePeriodSeconds != nil {
		out.TerminationGracePeriodSeconds = new(int64)
		*out.TerminationGracePeriodSeconds = *in.TerminationGracePeriodSeconds
	} else {
		out.TerminationGracePeriodSeconds = nil
	}
	out.DNSPolicy = newer.DNSPolicy(in.DNSPolicy)
	if in.NodeSelector != nil {
		out.NodeSelector = make(map[string]string)
		for key, val := range in.NodeSelector {
			out.NodeSelector[key] = val
		}
	} else {
		out.NodeSelector = nil
	}
	out.ServiceAccount = in.ServiceAccount
	out.Host = in.Host
	out.HostNetwork = in.HostNetwork
	return nil
}

func convert_api_PodSpec_To_v1beta3_PodSpec(in *newer.PodSpec, out *PodSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodSpec))(in)
	}
	if in.Volumes != nil {
		out.Volumes = make([]Volume, len(in.Volumes))
		for i := range in.Volumes {
			if err := convert_api_Volume_To_v1beta3_Volume(&in.Volumes[i], &out.Volumes[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Volumes = nil
	}
	if in.Containers != nil {
		out.Containers = make([]Container, len(in.Containers))
		for i := range in.Containers {
			if err := convert_api_Container_To_v1beta3_Container(&in.Containers[i], &out.Containers[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Containers = nil
	}
	out.RestartPolicy = RestartPolicy(in.RestartPolicy)
	if in.TerminationGracePeriodSeconds != nil {
		out.TerminationGracePeriodSeconds = new(int64)
		*out.TerminationGracePeriodSeconds = *in.TerminationGracePeriodSeconds
	} else {
		out.TerminationGracePeriodSeconds = nil
	}
	out.DNSPolicy = DNSPolicy(in.DNSPolicy)
	if in.NodeSelector != nil {
		out.NodeSelector = make(map[string]string)
		for key, val := range in.NodeSelector {
			out.NodeSelector[key] = val
		}
	} else {
		out.NodeSelector = nil
	}
	out.ServiceAccount = in.ServiceAccount
	out.Host = in.Host
	out.HostNetwork = in.HostNetwork
	return nil
}

func convert_v1beta3_PodStatus_To_api_PodStatus(in *PodStatus, out *newer.PodStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodStatus))(in)
	}
	out.Phase = newer.PodPhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]newer.PodCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_v1beta3_PodCondition_To_api_PodCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	out.Message = in.Message
	out.HostIP = in.HostIP
	out.PodIP = in.PodIP
	if in.ContainerStatuses != nil {
		out.ContainerStatuses = make([]newer.ContainerStatus, len(in.ContainerStatuses))
		for i := range in.ContainerStatuses {
			if err := convert_v1beta3_ContainerStatus_To_api_ContainerStatus(&in.ContainerStatuses[i], &out.ContainerStatuses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.ContainerStatuses = nil
	}
	return nil
}

func convert_api_PodStatus_To_v1beta3_PodStatus(in *newer.PodStatus, out *PodStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodStatus))(in)
	}
	out.Phase = PodPhase(in.Phase)
	if in.Conditions != nil {
		out.Conditions = make([]PodCondition, len(in.Conditions))
		for i := range in.Conditions {
			if err := convert_api_PodCondition_To_v1beta3_PodCondition(&in.Conditions[i], &out.Conditions[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Conditions = nil
	}
	out.Message = in.Message
	out.HostIP = in.HostIP
	out.PodIP = in.PodIP
	if in.ContainerStatuses != nil {
		out.ContainerStatuses = make([]ContainerStatus, len(in.ContainerStatuses))
		for i := range in.ContainerStatuses {
			if err := convert_api_ContainerStatus_To_v1beta3_ContainerStatus(&in.ContainerStatuses[i], &out.ContainerStatuses[i], s); err != nil {
				return err
			}
		}
	} else {
		out.ContainerStatuses = nil
	}
	return nil
}

func convert_v1beta3_PodStatusResult_To_api_PodStatusResult(in *PodStatusResult, out *newer.PodStatusResult, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodStatusResult))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PodStatus_To_api_PodStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PodStatusResult_To_v1beta3_PodStatusResult(in *newer.PodStatusResult, out *PodStatusResult, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodStatusResult))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PodStatus_To_v1beta3_PodStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PodTemplate_To_api_PodTemplate(in *PodTemplate, out *newer.PodTemplate, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodTemplate))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PodTemplateSpec_To_api_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PodTemplate_To_v1beta3_PodTemplate(in *newer.PodTemplate, out *PodTemplate, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodTemplate))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PodTemplateSpec_To_v1beta3_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_PodTemplateList_To_api_PodTemplateList(in *PodTemplateList, out *newer.PodTemplateList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodTemplateList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.PodTemplate, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_PodTemplate_To_api_PodTemplate(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_PodTemplateList_To_v1beta3_PodTemplateList(in *newer.PodTemplateList, out *PodTemplateList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodTemplateList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]PodTemplate, len(in.Items))
		for i := range in.Items {
			if err := convert_api_PodTemplate_To_v1beta3_PodTemplate(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_PodTemplateSpec_To_api_PodTemplateSpec(in *PodTemplateSpec, out *newer.PodTemplateSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*PodTemplateSpec))(in)
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_PodSpec_To_api_PodSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	return nil
}

func convert_api_PodTemplateSpec_To_v1beta3_PodTemplateSpec(in *newer.PodTemplateSpec, out *PodTemplateSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.PodTemplateSpec))(in)
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_PodSpec_To_v1beta3_PodSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_Probe_To_api_Probe(in *Probe, out *newer.Probe, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Probe))(in)
	}
	if err := convert_v1beta3_Handler_To_api_Handler(&in.Handler, &out.Handler, s); err != nil {
		return err
	}
	out.InitialDelaySeconds = in.InitialDelaySeconds
	out.TimeoutSeconds = in.TimeoutSeconds
	return nil
}

func convert_api_Probe_To_v1beta3_Probe(in *newer.Probe, out *Probe, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Probe))(in)
	}
	if err := convert_api_Handler_To_v1beta3_Handler(&in.Handler, &out.Handler, s); err != nil {
		return err
	}
	out.InitialDelaySeconds = in.InitialDelaySeconds
	out.TimeoutSeconds = in.TimeoutSeconds
	return nil
}

func convert_v1beta3_RangeAllocation_To_api_RangeAllocation(in *RangeAllocation, out *newer.RangeAllocation, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*RangeAllocation))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	out.Range = in.Range
	if err := s.Convert(&in.Data, &out.Data, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_RangeAllocation_To_v1beta3_RangeAllocation(in *newer.RangeAllocation, out *RangeAllocation, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.RangeAllocation))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	out.Range = in.Range
	if err := s.Convert(&in.Data, &out.Data, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ReplicationController_To_api_ReplicationController(in *ReplicationController, out *newer.ReplicationController, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ReplicationController))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ReplicationControllerSpec_To_api_ReplicationControllerSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ReplicationControllerStatus_To_api_ReplicationControllerStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_ReplicationController_To_v1beta3_ReplicationController(in *newer.ReplicationController, out *ReplicationController, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ReplicationController))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ReplicationControllerSpec_To_v1beta3_ReplicationControllerSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_ReplicationControllerStatus_To_v1beta3_ReplicationControllerStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ReplicationControllerList_To_api_ReplicationControllerList(in *ReplicationControllerList, out *newer.ReplicationControllerList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ReplicationControllerList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.ReplicationController, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_ReplicationController_To_api_ReplicationController(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ReplicationControllerList_To_v1beta3_ReplicationControllerList(in *newer.ReplicationControllerList, out *ReplicationControllerList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ReplicationControllerList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ReplicationController, len(in.Items))
		for i := range in.Items {
			if err := convert_api_ReplicationController_To_v1beta3_ReplicationController(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_ReplicationControllerSpec_To_api_ReplicationControllerSpec(in *ReplicationControllerSpec, out *newer.ReplicationControllerSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ReplicationControllerSpec))(in)
	}
	out.Replicas = in.Replicas
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	} else {
		out.Selector = nil
	}
	if in.TemplateRef != nil {
		out.TemplateRef = new(newer.ObjectReference)
		if err := convert_v1beta3_ObjectReference_To_api_ObjectReference(in.TemplateRef, out.TemplateRef, s); err != nil {
			return err
		}
	} else {
		out.TemplateRef = nil
	}
	if in.Template != nil {
		out.Template = new(newer.PodTemplateSpec)
		if err := convert_v1beta3_PodTemplateSpec_To_api_PodTemplateSpec(in.Template, out.Template, s); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}

func convert_api_ReplicationControllerSpec_To_v1beta3_ReplicationControllerSpec(in *newer.ReplicationControllerSpec, out *ReplicationControllerSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ReplicationControllerSpec))(in)
	}
	out.Replicas = in.Replicas
	if in.Selector != nil {
		out.Selector = make(map[string]string)
		for key, val := range in.Selector {
			out.Selector[key] = val
		}
	} else {
		out.Selector = nil
	}
	if in.TemplateRef != nil {
		out.TemplateRef = new(ObjectReference)
		if err := convert_api_ObjectReference_To_v1beta3_ObjectReference(in.TemplateRef, out.TemplateRef, s); err != nil {
			return err
		}
	} else {
		out.TemplateRef = nil
	}
	if in.Template != nil {
		out.Template = new(PodTemplateSpec)
		if err := convert_api_PodTemplateSpec_To_v1beta3_PodTemplateSpec(in.Template, out.Template, s); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}

func convert_v1beta3_ReplicationControllerStatus_To_api_ReplicationControllerStatus(in *ReplicationControllerStatus, out *newer.ReplicationControllerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ReplicationControllerStatus))(in)
	}
	out.Replicas = in.Replicas
	return nil
}

func convert_api_ReplicationControllerStatus_To_v1beta3_ReplicationControllerStatus(in *newer.ReplicationControllerStatus, out *ReplicationControllerStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ReplicationControllerStatus))(in)
	}
	out.Replicas = in.Replicas
	return nil
}

func convert_v1beta3_ResourceQuota_To_api_ResourceQuota(in *ResourceQuota, out *newer.ResourceQuota, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceQuota))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ResourceQuotaSpec_To_api_ResourceQuotaSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ResourceQuotaStatus_To_api_ResourceQuotaStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_ResourceQuota_To_v1beta3_ResourceQuota(in *newer.ResourceQuota, out *ResourceQuota, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ResourceQuota))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ResourceQuotaSpec_To_v1beta3_ResourceQuotaSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_ResourceQuotaStatus_To_v1beta3_ResourceQuotaStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ResourceQuotaList_To_api_ResourceQuotaList(in *ResourceQuotaList, out *newer.ResourceQuotaList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceQuotaList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.ResourceQuota, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_ResourceQuota_To_api_ResourceQuota(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ResourceQuotaList_To_v1beta3_ResourceQuotaList(in *newer.ResourceQuotaList, out *ResourceQuotaList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ResourceQuotaList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ResourceQuota, len(in.Items))
		for i := range in.Items {
			if err := convert_api_ResourceQuota_To_v1beta3_ResourceQuota(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_ResourceQuotaSpec_To_api_ResourceQuotaSpec(in *ResourceQuotaSpec, out *newer.ResourceQuotaSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceQuotaSpec))(in)
	}
	if in.Hard != nil {
		out.Hard = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[newer.ResourceName(key)] = newVal
		}
	} else {
		out.Hard = nil
	}
	return nil
}

func convert_api_ResourceQuotaSpec_To_v1beta3_ResourceQuotaSpec(in *newer.ResourceQuotaSpec, out *ResourceQuotaSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ResourceQuotaSpec))(in)
	}
	if in.Hard != nil {
		out.Hard = make(map[ResourceName]resource.Quantity)
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

func convert_v1beta3_ResourceQuotaStatus_To_api_ResourceQuotaStatus(in *ResourceQuotaStatus, out *newer.ResourceQuotaStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceQuotaStatus))(in)
	}
	if in.Hard != nil {
		out.Hard = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Hard {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Hard[newer.ResourceName(key)] = newVal
		}
	} else {
		out.Hard = nil
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
	} else {
		out.Used = nil
	}
	return nil
}

func convert_api_ResourceQuotaStatus_To_v1beta3_ResourceQuotaStatus(in *newer.ResourceQuotaStatus, out *ResourceQuotaStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ResourceQuotaStatus))(in)
	}
	if in.Hard != nil {
		out.Hard = make(map[ResourceName]resource.Quantity)
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
		out.Used = make(map[ResourceName]resource.Quantity)
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

func convert_v1beta3_ResourceRequirements_To_api_ResourceRequirements(in *ResourceRequirements, out *newer.ResourceRequirements, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ResourceRequirements))(in)
	}
	if in.Limits != nil {
		out.Limits = make(map[newer.ResourceName]resource.Quantity)
		for key, val := range in.Limits {
			newVal := resource.Quantity{}
			if err := s.Convert(&val, &newVal, 0); err != nil {
				return err
			}
			out.Limits[newer.ResourceName(key)] = newVal
		}
	} else {
		out.Limits = nil
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
	} else {
		out.Requests = nil
	}
	return nil
}

func convert_api_ResourceRequirements_To_v1beta3_ResourceRequirements(in *newer.ResourceRequirements, out *ResourceRequirements, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ResourceRequirements))(in)
	}
	if in.Limits != nil {
		out.Limits = make(map[ResourceName]resource.Quantity)
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
		out.Requests = make(map[ResourceName]resource.Quantity)
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

func convert_api_SELinuxOptions_To_v1beta3_SELinuxOptions(in *newer.SELinuxOptions, out *SELinuxOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.SELinuxOptions))(in)
	}
	out.User = in.User
	out.Role = in.Role
	out.Type = in.Type
	out.Level = in.Level
	return nil
}

func convert_v1beta3_SELinuxOptions_To_api_SELinuxOptions(in *SELinuxOptions, out *newer.SELinuxOptions, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SELinuxOptions))(in)
	}
	out.User = in.User
	out.Role = in.Role
	out.Type = in.Type
	out.Level = in.Level
	return nil
}

func convert_v1beta3_Secret_To_api_Secret(in *Secret, out *newer.Secret, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Secret))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
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
	out.Type = newer.SecretType(in.Type)
	return nil
}

func convert_api_Secret_To_v1beta3_Secret(in *newer.Secret, out *Secret, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Secret))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
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

func convert_v1beta3_SecretList_To_api_SecretList(in *SecretList, out *newer.SecretList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SecretList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Secret, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_Secret_To_api_Secret(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_SecretList_To_v1beta3_SecretList(in *newer.SecretList, out *SecretList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.SecretList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Secret, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Secret_To_v1beta3_Secret(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_SecretVolumeSource_To_api_SecretVolumeSource(in *SecretVolumeSource, out *newer.SecretVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SecretVolumeSource))(in)
	}
	out.SecretName = in.SecretName
	return nil
}

func convert_api_SecretVolumeSource_To_v1beta3_SecretVolumeSource(in *newer.SecretVolumeSource, out *SecretVolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.SecretVolumeSource))(in)
	}
	out.SecretName = in.SecretName
	return nil
}

func convert_api_SecurityContext_To_v1beta3_SecurityContext(in *newer.SecurityContext, out *SecurityContext, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.SecurityContext))(in)
	}
	if in.Capabilities != nil {
		out.Capabilities = new(Capabilities)
		if err := convert_api_Capabilities_To_v1beta3_Capabilities(in.Capabilities, out.Capabilities, s); err != nil {
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
		if err := convert_api_SELinuxOptions_To_v1beta3_SELinuxOptions(in.SELinuxOptions, out.SELinuxOptions, s); err != nil {
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

func convert_v1beta3_SecurityContext_To_api_SecurityContext(in *SecurityContext, out *newer.SecurityContext, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SecurityContext))(in)
	}
	if in.Capabilities != nil {
		out.Capabilities = new(newer.Capabilities)
		if err := convert_v1beta3_Capabilities_To_api_Capabilities(in.Capabilities, out.Capabilities, s); err != nil {
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
		out.SELinuxOptions = new(newer.SELinuxOptions)
		if err := convert_v1beta3_SELinuxOptions_To_api_SELinuxOptions(in.SELinuxOptions, out.SELinuxOptions, s); err != nil {
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

func convert_v1beta3_SerializedReference_To_api_SerializedReference(in *SerializedReference, out *newer.SerializedReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*SerializedReference))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectReference_To_api_ObjectReference(&in.Reference, &out.Reference, s); err != nil {
		return err
	}
	return nil
}

func convert_api_SerializedReference_To_v1beta3_SerializedReference(in *newer.SerializedReference, out *SerializedReference, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.SerializedReference))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectReference_To_v1beta3_ObjectReference(&in.Reference, &out.Reference, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_Service_To_api_Service(in *Service, out *newer.Service, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Service))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ServiceSpec_To_api_ServiceSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ServiceStatus_To_api_ServiceStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Service_To_v1beta3_Service(in *newer.Service, out *Service, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Service))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if err := convert_api_ServiceSpec_To_v1beta3_ServiceSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := convert_api_ServiceStatus_To_v1beta3_ServiceStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ServiceAccount_To_api_ServiceAccount(in *ServiceAccount, out *newer.ServiceAccount, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceAccount))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ObjectMeta_To_api_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Secrets != nil {
		out.Secrets = make([]newer.ObjectReference, len(in.Secrets))
		for i := range in.Secrets {
			if err := convert_v1beta3_ObjectReference_To_api_ObjectReference(&in.Secrets[i], &out.Secrets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Secrets = nil
	}
	return nil
}

func convert_api_ServiceAccount_To_v1beta3_ServiceAccount(in *newer.ServiceAccount, out *ServiceAccount, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ServiceAccount))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ObjectMeta_To_v1beta3_ObjectMeta(&in.ObjectMeta, &out.ObjectMeta, s); err != nil {
		return err
	}
	if in.Secrets != nil {
		out.Secrets = make([]ObjectReference, len(in.Secrets))
		for i := range in.Secrets {
			if err := convert_api_ObjectReference_To_v1beta3_ObjectReference(&in.Secrets[i], &out.Secrets[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Secrets = nil
	}
	return nil
}

func convert_v1beta3_ServiceAccountList_To_api_ServiceAccountList(in *ServiceAccountList, out *newer.ServiceAccountList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceAccountList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.ServiceAccount, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_ServiceAccount_To_api_ServiceAccount(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ServiceAccountList_To_v1beta3_ServiceAccountList(in *newer.ServiceAccountList, out *ServiceAccountList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ServiceAccountList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]ServiceAccount, len(in.Items))
		for i := range in.Items {
			if err := convert_api_ServiceAccount_To_v1beta3_ServiceAccount(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_ServiceList_To_api_ServiceList(in *ServiceList, out *newer.ServiceList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceList))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]newer.Service, len(in.Items))
		for i := range in.Items {
			if err := convert_v1beta3_Service_To_api_Service(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_api_ServiceList_To_v1beta3_ServiceList(in *newer.ServiceList, out *ServiceList, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ServiceList))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	if in.Items != nil {
		out.Items = make([]Service, len(in.Items))
		for i := range in.Items {
			if err := convert_api_Service_To_v1beta3_Service(&in.Items[i], &out.Items[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Items = nil
	}
	return nil
}

func convert_v1beta3_ServicePort_To_api_ServicePort(in *ServicePort, out *newer.ServicePort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServicePort))(in)
	}
	out.Name = in.Name
	out.Protocol = newer.Protocol(in.Protocol)
	out.Port = in.Port
	if err := s.Convert(&in.TargetPort, &out.TargetPort, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_ServicePort_To_v1beta3_ServicePort(in *newer.ServicePort, out *ServicePort, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ServicePort))(in)
	}
	out.Name = in.Name
	out.Protocol = Protocol(in.Protocol)
	out.Port = in.Port
	if err := s.Convert(&in.TargetPort, &out.TargetPort, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_ServiceSpec_To_api_ServiceSpec(in *ServiceSpec, out *newer.ServiceSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceSpec))(in)
	}
	if in.Ports != nil {
		out.Ports = make([]newer.ServicePort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_v1beta3_ServicePort_To_api_ServicePort(&in.Ports[i], &out.Ports[i], s); err != nil {
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
	out.PortalIP = in.PortalIP
	out.CreateExternalLoadBalancer = in.CreateExternalLoadBalancer
	if in.PublicIPs != nil {
		out.PublicIPs = make([]string, len(in.PublicIPs))
		for i := range in.PublicIPs {
			out.PublicIPs[i] = in.PublicIPs[i]
		}
	} else {
		out.PublicIPs = nil
	}
	out.SessionAffinity = newer.AffinityType(in.SessionAffinity)
	return nil
}

func convert_api_ServiceSpec_To_v1beta3_ServiceSpec(in *newer.ServiceSpec, out *ServiceSpec, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ServiceSpec))(in)
	}
	if in.Ports != nil {
		out.Ports = make([]ServicePort, len(in.Ports))
		for i := range in.Ports {
			if err := convert_api_ServicePort_To_v1beta3_ServicePort(&in.Ports[i], &out.Ports[i], s); err != nil {
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
	out.PortalIP = in.PortalIP
	out.CreateExternalLoadBalancer = in.CreateExternalLoadBalancer
	if in.PublicIPs != nil {
		out.PublicIPs = make([]string, len(in.PublicIPs))
		for i := range in.PublicIPs {
			out.PublicIPs[i] = in.PublicIPs[i]
		}
	} else {
		out.PublicIPs = nil
	}
	out.SessionAffinity = AffinityType(in.SessionAffinity)
	return nil
}

func convert_v1beta3_ServiceStatus_To_api_ServiceStatus(in *ServiceStatus, out *newer.ServiceStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*ServiceStatus))(in)
	}
	return nil
}

func convert_api_ServiceStatus_To_v1beta3_ServiceStatus(in *newer.ServiceStatus, out *ServiceStatus, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.ServiceStatus))(in)
	}
	return nil
}

func convert_v1beta3_Status_To_api_Status(in *Status, out *newer.Status, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Status))(in)
	}
	if err := convert_v1beta3_TypeMeta_To_api_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_v1beta3_ListMeta_To_api_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	out.Status = in.Status
	out.Message = in.Message
	out.Reason = newer.StatusReason(in.Reason)
	if in.Details != nil {
		out.Details = new(newer.StatusDetails)
		if err := convert_v1beta3_StatusDetails_To_api_StatusDetails(in.Details, out.Details, s); err != nil {
			return err
		}
	} else {
		out.Details = nil
	}
	out.Code = in.Code
	return nil
}

func convert_api_Status_To_v1beta3_Status(in *newer.Status, out *Status, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Status))(in)
	}
	if err := convert_api_TypeMeta_To_v1beta3_TypeMeta(&in.TypeMeta, &out.TypeMeta, s); err != nil {
		return err
	}
	if err := convert_api_ListMeta_To_v1beta3_ListMeta(&in.ListMeta, &out.ListMeta, s); err != nil {
		return err
	}
	out.Status = in.Status
	out.Message = in.Message
	out.Reason = StatusReason(in.Reason)
	if in.Details != nil {
		out.Details = new(StatusDetails)
		if err := convert_api_StatusDetails_To_v1beta3_StatusDetails(in.Details, out.Details, s); err != nil {
			return err
		}
	} else {
		out.Details = nil
	}
	out.Code = in.Code
	return nil
}

func convert_v1beta3_StatusCause_To_api_StatusCause(in *StatusCause, out *newer.StatusCause, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*StatusCause))(in)
	}
	out.Type = newer.CauseType(in.Type)
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func convert_api_StatusCause_To_v1beta3_StatusCause(in *newer.StatusCause, out *StatusCause, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.StatusCause))(in)
	}
	out.Type = CauseType(in.Type)
	out.Message = in.Message
	out.Field = in.Field
	return nil
}

func convert_v1beta3_StatusDetails_To_api_StatusDetails(in *StatusDetails, out *newer.StatusDetails, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*StatusDetails))(in)
	}
	out.ID = in.ID
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]newer.StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := convert_v1beta3_StatusCause_To_api_StatusCause(&in.Causes[i], &out.Causes[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Causes = nil
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func convert_api_StatusDetails_To_v1beta3_StatusDetails(in *newer.StatusDetails, out *StatusDetails, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.StatusDetails))(in)
	}
	out.ID = in.ID
	out.Kind = in.Kind
	if in.Causes != nil {
		out.Causes = make([]StatusCause, len(in.Causes))
		for i := range in.Causes {
			if err := convert_api_StatusCause_To_v1beta3_StatusCause(&in.Causes[i], &out.Causes[i], s); err != nil {
				return err
			}
		}
	} else {
		out.Causes = nil
	}
	out.RetryAfterSeconds = in.RetryAfterSeconds
	return nil
}

func convert_v1beta3_TCPSocketAction_To_api_TCPSocketAction(in *TCPSocketAction, out *newer.TCPSocketAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*TCPSocketAction))(in)
	}
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	return nil
}

func convert_api_TCPSocketAction_To_v1beta3_TCPSocketAction(in *newer.TCPSocketAction, out *TCPSocketAction, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.TCPSocketAction))(in)
	}
	if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_TypeMeta_To_api_TypeMeta(in *TypeMeta, out *newer.TypeMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*TypeMeta))(in)
	}
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	return nil
}

func convert_api_TypeMeta_To_v1beta3_TypeMeta(in *newer.TypeMeta, out *TypeMeta, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.TypeMeta))(in)
	}
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	return nil
}

func convert_v1beta3_Volume_To_api_Volume(in *Volume, out *newer.Volume, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*Volume))(in)
	}
	out.Name = in.Name
	if err := convert_v1beta3_VolumeSource_To_api_VolumeSource(&in.VolumeSource, &out.VolumeSource, s); err != nil {
		return err
	}
	return nil
}

func convert_api_Volume_To_v1beta3_Volume(in *newer.Volume, out *Volume, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.Volume))(in)
	}
	out.Name = in.Name
	if err := convert_api_VolumeSource_To_v1beta3_VolumeSource(&in.VolumeSource, &out.VolumeSource, s); err != nil {
		return err
	}
	return nil
}

func convert_v1beta3_VolumeMount_To_api_VolumeMount(in *VolumeMount, out *newer.VolumeMount, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*VolumeMount))(in)
	}
	out.Name = in.Name
	out.ReadOnly = in.ReadOnly
	out.MountPath = in.MountPath
	return nil
}

func convert_api_VolumeMount_To_v1beta3_VolumeMount(in *newer.VolumeMount, out *VolumeMount, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.VolumeMount))(in)
	}
	out.Name = in.Name
	out.ReadOnly = in.ReadOnly
	out.MountPath = in.MountPath
	return nil
}

func convert_v1beta3_VolumeSource_To_api_VolumeSource(in *VolumeSource, out *newer.VolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*VolumeSource))(in)
	}
	if in.HostPath != nil {
		out.HostPath = new(newer.HostPathVolumeSource)
		if err := convert_v1beta3_HostPathVolumeSource_To_api_HostPathVolumeSource(in.HostPath, out.HostPath, s); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.EmptyDir != nil {
		out.EmptyDir = new(newer.EmptyDirVolumeSource)
		if err := convert_v1beta3_EmptyDirVolumeSource_To_api_EmptyDirVolumeSource(in.EmptyDir, out.EmptyDir, s); err != nil {
			return err
		}
	} else {
		out.EmptyDir = nil
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(newer.GCEPersistentDiskVolumeSource)
		if err := convert_v1beta3_GCEPersistentDiskVolumeSource_To_api_GCEPersistentDiskVolumeSource(in.GCEPersistentDisk, out.GCEPersistentDisk, s); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(newer.AWSElasticBlockStoreVolumeSource)
		if err := convert_v1beta3_AWSElasticBlockStoreVolumeSource_To_api_AWSElasticBlockStoreVolumeSource(in.AWSElasticBlockStore, out.AWSElasticBlockStore, s); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.GitRepo != nil {
		out.GitRepo = new(newer.GitRepoVolumeSource)
		if err := convert_v1beta3_GitRepoVolumeSource_To_api_GitRepoVolumeSource(in.GitRepo, out.GitRepo, s); err != nil {
			return err
		}
	} else {
		out.GitRepo = nil
	}
	if in.Secret != nil {
		out.Secret = new(newer.SecretVolumeSource)
		if err := convert_v1beta3_SecretVolumeSource_To_api_SecretVolumeSource(in.Secret, out.Secret, s); err != nil {
			return err
		}
	} else {
		out.Secret = nil
	}
	if in.NFS != nil {
		out.NFS = new(newer.NFSVolumeSource)
		if err := convert_v1beta3_NFSVolumeSource_To_api_NFSVolumeSource(in.NFS, out.NFS, s); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	if in.ISCSI != nil {
		out.ISCSI = new(newer.ISCSIVolumeSource)
		if err := convert_v1beta3_ISCSIVolumeSource_To_api_ISCSIVolumeSource(in.ISCSI, out.ISCSI, s); err != nil {
			return err
		}
	} else {
		out.ISCSI = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(newer.GlusterfsVolumeSource)
		if err := convert_v1beta3_GlusterfsVolumeSource_To_api_GlusterfsVolumeSource(in.Glusterfs, out.Glusterfs, s); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.PersistentVolumeClaimVolumeSource != nil {
		out.PersistentVolumeClaimVolumeSource = new(newer.PersistentVolumeClaimVolumeSource)
		if err := convert_v1beta3_PersistentVolumeClaimVolumeSource_To_api_PersistentVolumeClaimVolumeSource(in.PersistentVolumeClaimVolumeSource, out.PersistentVolumeClaimVolumeSource, s); err != nil {
			return err
		}
	} else {
		out.PersistentVolumeClaimVolumeSource = nil
	}
	return nil
}

func convert_api_VolumeSource_To_v1beta3_VolumeSource(in *newer.VolumeSource, out *VolumeSource, s conversion.Scope) error {
	if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {
		defaulting.(func(*newer.VolumeSource))(in)
	}
	if in.HostPath != nil {
		out.HostPath = new(HostPathVolumeSource)
		if err := convert_api_HostPathVolumeSource_To_v1beta3_HostPathVolumeSource(in.HostPath, out.HostPath, s); err != nil {
			return err
		}
	} else {
		out.HostPath = nil
	}
	if in.EmptyDir != nil {
		out.EmptyDir = new(EmptyDirVolumeSource)
		if err := convert_api_EmptyDirVolumeSource_To_v1beta3_EmptyDirVolumeSource(in.EmptyDir, out.EmptyDir, s); err != nil {
			return err
		}
	} else {
		out.EmptyDir = nil
	}
	if in.GCEPersistentDisk != nil {
		out.GCEPersistentDisk = new(GCEPersistentDiskVolumeSource)
		if err := convert_api_GCEPersistentDiskVolumeSource_To_v1beta3_GCEPersistentDiskVolumeSource(in.GCEPersistentDisk, out.GCEPersistentDisk, s); err != nil {
			return err
		}
	} else {
		out.GCEPersistentDisk = nil
	}
	if in.AWSElasticBlockStore != nil {
		out.AWSElasticBlockStore = new(AWSElasticBlockStoreVolumeSource)
		if err := convert_api_AWSElasticBlockStoreVolumeSource_To_v1beta3_AWSElasticBlockStoreVolumeSource(in.AWSElasticBlockStore, out.AWSElasticBlockStore, s); err != nil {
			return err
		}
	} else {
		out.AWSElasticBlockStore = nil
	}
	if in.GitRepo != nil {
		out.GitRepo = new(GitRepoVolumeSource)
		if err := convert_api_GitRepoVolumeSource_To_v1beta3_GitRepoVolumeSource(in.GitRepo, out.GitRepo, s); err != nil {
			return err
		}
	} else {
		out.GitRepo = nil
	}
	if in.Secret != nil {
		out.Secret = new(SecretVolumeSource)
		if err := convert_api_SecretVolumeSource_To_v1beta3_SecretVolumeSource(in.Secret, out.Secret, s); err != nil {
			return err
		}
	} else {
		out.Secret = nil
	}
	if in.NFS != nil {
		out.NFS = new(NFSVolumeSource)
		if err := convert_api_NFSVolumeSource_To_v1beta3_NFSVolumeSource(in.NFS, out.NFS, s); err != nil {
			return err
		}
	} else {
		out.NFS = nil
	}
	if in.ISCSI != nil {
		out.ISCSI = new(ISCSIVolumeSource)
		if err := convert_api_ISCSIVolumeSource_To_v1beta3_ISCSIVolumeSource(in.ISCSI, out.ISCSI, s); err != nil {
			return err
		}
	} else {
		out.ISCSI = nil
	}
	if in.Glusterfs != nil {
		out.Glusterfs = new(GlusterfsVolumeSource)
		if err := convert_api_GlusterfsVolumeSource_To_v1beta3_GlusterfsVolumeSource(in.Glusterfs, out.Glusterfs, s); err != nil {
			return err
		}
	} else {
		out.Glusterfs = nil
	}
	if in.PersistentVolumeClaimVolumeSource != nil {
		out.PersistentVolumeClaimVolumeSource = new(PersistentVolumeClaimVolumeSource)
		if err := convert_api_PersistentVolumeClaimVolumeSource_To_v1beta3_PersistentVolumeClaimVolumeSource(in.PersistentVolumeClaimVolumeSource, out.PersistentVolumeClaimVolumeSource, s); err != nil {
			return err
		}
	} else {
		out.PersistentVolumeClaimVolumeSource = nil
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
		convert_api_RangeAllocation_To_v1beta3_RangeAllocation,
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
		convert_api_ServiceAccountList_To_v1beta3_ServiceAccountList,
		convert_api_ServiceAccount_To_v1beta3_ServiceAccount,
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
		convert_v1beta3_RangeAllocation_To_api_RangeAllocation,
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
		convert_v1beta3_ServiceAccountList_To_api_ServiceAccountList,
		convert_v1beta3_ServiceAccount_To_api_ServiceAccount,
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
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}
