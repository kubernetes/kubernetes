/*
Copyright 2015 Google Inc. All rights reserved.

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
	"fmt"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

func init() {
	err := newer.Scheme.AddGeneratedConversionFuncs(
		func(in *AWSElasticBlockStoreVolumeSource, out *newer.AWSElasticBlockStoreVolumeSource, s conversion.Scope) error {
			out.VolumeID = in.VolumeID
			out.FSType = in.FSType
			out.Partition = in.Partition
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *newer.AWSElasticBlockStoreVolumeSource, out *AWSElasticBlockStoreVolumeSource, s conversion.Scope) error {
			out.VolumeID = in.VolumeID
			out.FSType = in.FSType
			out.Partition = in.Partition
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *Binding, out *newer.Binding, s conversion.Scope) error {
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
		},
		func(in *newer.Binding, out *Binding, s conversion.Scope) error {
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
		},
		func(in *Capabilities, out *newer.Capabilities, s conversion.Scope) error {
			if in.Add != nil {
				out.Add = make([]newer.CapabilityType, len(in.Add))
				for i := range in.Add {
					if err := s.Convert(&in.Add[i], &out.Add[i], 0); err != nil {
						return err
					}
				}
			}
			if in.Drop != nil {
				out.Drop = make([]newer.CapabilityType, len(in.Drop))
				for i := range in.Drop {
					if err := s.Convert(&in.Drop[i], &out.Drop[i], 0); err != nil {
						return err
					}
				}
			}
			return nil
		},
		func(in *newer.Capabilities, out *Capabilities, s conversion.Scope) error {
			if in.Add != nil {
				out.Add = make([]CapabilityType, len(in.Add))
				for i := range in.Add {
					if err := s.Convert(&in.Add[i], &out.Add[i], 0); err != nil {
						return err
					}
				}
			}
			if in.Drop != nil {
				out.Drop = make([]CapabilityType, len(in.Drop))
				for i := range in.Drop {
					if err := s.Convert(&in.Drop[i], &out.Drop[i], 0); err != nil {
						return err
					}
				}
			}
			return nil
		},
		func(in *ComponentCondition, out *newer.ComponentCondition, s conversion.Scope) error {
			out.Type = newer.ComponentConditionType(in.Type)
			out.Status = newer.ConditionStatus(in.Status)
			out.Message = in.Message
			out.Error = in.Error
			return nil
		},
		func(in *newer.ComponentCondition, out *ComponentCondition, s conversion.Scope) error {
			out.Type = ComponentConditionType(in.Type)
			out.Status = ConditionStatus(in.Status)
			out.Message = in.Message
			out.Error = in.Error
			return nil
		},
		func(in *ComponentStatus, out *newer.ComponentStatus, s conversion.Scope) error {
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
		},
		func(in *newer.ComponentStatus, out *ComponentStatus, s conversion.Scope) error {
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
		},
		func(in *ComponentStatusList, out *newer.ComponentStatusList, s conversion.Scope) error {
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
		},
		func(in *newer.ComponentStatusList, out *ComponentStatusList, s conversion.Scope) error {
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
		},
		func(in *Container, out *newer.Container, s conversion.Scope) error {
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
			out.Privileged = in.Privileged
			out.ImagePullPolicy = newer.PullPolicy(in.ImagePullPolicy)
			if err := s.Convert(&in.Capabilities, &out.Capabilities, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.Container, out *Container, s conversion.Scope) error {
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
			out.Privileged = in.Privileged
			out.ImagePullPolicy = PullPolicy(in.ImagePullPolicy)
			if err := s.Convert(&in.Capabilities, &out.Capabilities, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ContainerPort, out *newer.ContainerPort, s conversion.Scope) error {
			out.Name = in.Name
			out.HostPort = in.HostPort
			out.ContainerPort = in.ContainerPort
			out.Protocol = newer.Protocol(in.Protocol)
			out.HostIP = in.HostIP
			return nil
		},
		func(in *newer.ContainerPort, out *ContainerPort, s conversion.Scope) error {
			out.Name = in.Name
			out.HostPort = in.HostPort
			out.ContainerPort = in.ContainerPort
			out.Protocol = Protocol(in.Protocol)
			out.HostIP = in.HostIP
			return nil
		},
		func(in *ContainerState, out *newer.ContainerState, s conversion.Scope) error {
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
		},
		func(in *newer.ContainerState, out *ContainerState, s conversion.Scope) error {
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
		},
		func(in *ContainerStateRunning, out *newer.ContainerStateRunning, s conversion.Scope) error {
			if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.ContainerStateRunning, out *ContainerStateRunning, s conversion.Scope) error {
			if err := s.Convert(&in.StartedAt, &out.StartedAt, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ContainerStateTerminated, out *newer.ContainerStateTerminated, s conversion.Scope) error {
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
		},
		func(in *newer.ContainerStateTerminated, out *ContainerStateTerminated, s conversion.Scope) error {
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
		},
		func(in *ContainerStateWaiting, out *newer.ContainerStateWaiting, s conversion.Scope) error {
			out.Reason = in.Reason
			return nil
		},
		func(in *newer.ContainerStateWaiting, out *ContainerStateWaiting, s conversion.Scope) error {
			out.Reason = in.Reason
			return nil
		},
		func(in *ContainerStatus, out *newer.ContainerStatus, s conversion.Scope) error {
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
		},
		func(in *newer.ContainerStatus, out *ContainerStatus, s conversion.Scope) error {
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
		},
		func(in *DeleteOptions, out *newer.DeleteOptions, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if in.GracePeriodSeconds != nil {
				out.GracePeriodSeconds = new(int64)
				*out.GracePeriodSeconds = *in.GracePeriodSeconds
			}
			return nil
		},
		func(in *newer.DeleteOptions, out *DeleteOptions, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if in.GracePeriodSeconds != nil {
				out.GracePeriodSeconds = new(int64)
				*out.GracePeriodSeconds = *in.GracePeriodSeconds
			}
			return nil
		},
		func(in *EmptyDirVolumeSource, out *newer.EmptyDirVolumeSource, s conversion.Scope) error {
			out.Medium = newer.StorageType(in.Medium)
			return nil
		},
		func(in *newer.EmptyDirVolumeSource, out *EmptyDirVolumeSource, s conversion.Scope) error {
			out.Medium = StorageType(in.Medium)
			return nil
		},
		func(in *EndpointAddress, out *newer.EndpointAddress, s conversion.Scope) error {
			out.IP = in.IP
			if err := s.Convert(&in.TargetRef, &out.TargetRef, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.EndpointAddress, out *EndpointAddress, s conversion.Scope) error {
			out.IP = in.IP
			if err := s.Convert(&in.TargetRef, &out.TargetRef, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *EndpointPort, out *newer.EndpointPort, s conversion.Scope) error {
			out.Name = in.Name
			out.Port = in.Port
			out.Protocol = newer.Protocol(in.Protocol)
			return nil
		},
		func(in *newer.EndpointPort, out *EndpointPort, s conversion.Scope) error {
			out.Name = in.Name
			out.Port = in.Port
			out.Protocol = Protocol(in.Protocol)
			return nil
		},
		func(in *EndpointSubset, out *newer.EndpointSubset, s conversion.Scope) error {
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
		},
		func(in *newer.EndpointSubset, out *EndpointSubset, s conversion.Scope) error {
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
		},
		func(in *Endpoints, out *newer.Endpoints, s conversion.Scope) error {
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
		},
		func(in *newer.Endpoints, out *Endpoints, s conversion.Scope) error {
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
		},
		func(in *EndpointsList, out *newer.EndpointsList, s conversion.Scope) error {
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
		},
		func(in *newer.EndpointsList, out *EndpointsList, s conversion.Scope) error {
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
		},
		func(in *EnvVar, out *newer.EnvVar, s conversion.Scope) error {
			out.Name = in.Name
			out.Value = in.Value
			return nil
		},
		func(in *newer.EnvVar, out *EnvVar, s conversion.Scope) error {
			out.Name = in.Name
			out.Value = in.Value
			return nil
		},
		func(in *Event, out *newer.Event, s conversion.Scope) error {
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
		},
		func(in *newer.Event, out *Event, s conversion.Scope) error {
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
		},
		func(in *EventList, out *newer.EventList, s conversion.Scope) error {
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
		},
		func(in *newer.EventList, out *EventList, s conversion.Scope) error {
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
		},
		func(in *EventSource, out *newer.EventSource, s conversion.Scope) error {
			out.Component = in.Component
			out.Host = in.Host
			return nil
		},
		func(in *newer.EventSource, out *EventSource, s conversion.Scope) error {
			out.Component = in.Component
			out.Host = in.Host
			return nil
		},
		func(in *ExecAction, out *newer.ExecAction, s conversion.Scope) error {
			if in.Command != nil {
				out.Command = make([]string, len(in.Command))
				for i := range in.Command {
					out.Command[i] = in.Command[i]
				}
			}
			return nil
		},
		func(in *newer.ExecAction, out *ExecAction, s conversion.Scope) error {
			if in.Command != nil {
				out.Command = make([]string, len(in.Command))
				for i := range in.Command {
					out.Command[i] = in.Command[i]
				}
			}
			return nil
		},
		func(in *GCEPersistentDiskVolumeSource, out *newer.GCEPersistentDiskVolumeSource, s conversion.Scope) error {
			out.PDName = in.PDName
			out.FSType = in.FSType
			out.Partition = in.Partition
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *newer.GCEPersistentDiskVolumeSource, out *GCEPersistentDiskVolumeSource, s conversion.Scope) error {
			out.PDName = in.PDName
			out.FSType = in.FSType
			out.Partition = in.Partition
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *GitRepoVolumeSource, out *newer.GitRepoVolumeSource, s conversion.Scope) error {
			out.Repository = in.Repository
			out.Revision = in.Revision
			return nil
		},
		func(in *newer.GitRepoVolumeSource, out *GitRepoVolumeSource, s conversion.Scope) error {
			out.Repository = in.Repository
			out.Revision = in.Revision
			return nil
		},
		func(in *GlusterfsVolumeSource, out *newer.GlusterfsVolumeSource, s conversion.Scope) error {
			out.EndpointsName = in.EndpointsName
			out.Path = in.Path
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *newer.GlusterfsVolumeSource, out *GlusterfsVolumeSource, s conversion.Scope) error {
			out.EndpointsName = in.EndpointsName
			out.Path = in.Path
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *HTTPGetAction, out *newer.HTTPGetAction, s conversion.Scope) error {
			out.Path = in.Path
			if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
				return err
			}
			out.Host = in.Host
			return nil
		},
		func(in *newer.HTTPGetAction, out *HTTPGetAction, s conversion.Scope) error {
			out.Path = in.Path
			if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
				return err
			}
			out.Host = in.Host
			return nil
		},
		func(in *Handler, out *newer.Handler, s conversion.Scope) error {
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
		},
		func(in *newer.Handler, out *Handler, s conversion.Scope) error {
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
		},
		func(in *HostPathVolumeSource, out *newer.HostPathVolumeSource, s conversion.Scope) error {
			out.Path = in.Path
			return nil
		},
		func(in *newer.HostPathVolumeSource, out *HostPathVolumeSource, s conversion.Scope) error {
			out.Path = in.Path
			return nil
		},
		func(in *ISCSIVolumeSource, out *newer.ISCSIVolumeSource, s conversion.Scope) error {
			out.TargetPortal = in.TargetPortal
			out.IQN = in.IQN
			out.Lun = in.Lun
			out.FSType = in.FSType
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *newer.ISCSIVolumeSource, out *ISCSIVolumeSource, s conversion.Scope) error {
			out.TargetPortal = in.TargetPortal
			out.IQN = in.IQN
			out.Lun = in.Lun
			out.FSType = in.FSType
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *Lifecycle, out *newer.Lifecycle, s conversion.Scope) error {
			if err := s.Convert(&in.PostStart, &out.PostStart, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.PreStop, &out.PreStop, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.Lifecycle, out *Lifecycle, s conversion.Scope) error {
			if err := s.Convert(&in.PostStart, &out.PostStart, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.PreStop, &out.PreStop, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *LimitRange, out *newer.LimitRange, s conversion.Scope) error {
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
		},
		func(in *newer.LimitRange, out *LimitRange, s conversion.Scope) error {
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
		},
		func(in *LimitRangeItem, out *newer.LimitRangeItem, s conversion.Scope) error {
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
		},
		func(in *newer.LimitRangeItem, out *LimitRangeItem, s conversion.Scope) error {
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
		},
		func(in *LimitRangeList, out *newer.LimitRangeList, s conversion.Scope) error {
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
		},
		func(in *newer.LimitRangeList, out *LimitRangeList, s conversion.Scope) error {
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
		},
		func(in *LimitRangeSpec, out *newer.LimitRangeSpec, s conversion.Scope) error {
			if in.Limits != nil {
				out.Limits = make([]newer.LimitRangeItem, len(in.Limits))
				for i := range in.Limits {
					if err := s.Convert(&in.Limits[i], &out.Limits[i], 0); err != nil {
						return err
					}
				}
			}
			return nil
		},
		func(in *newer.LimitRangeSpec, out *LimitRangeSpec, s conversion.Scope) error {
			if in.Limits != nil {
				out.Limits = make([]LimitRangeItem, len(in.Limits))
				for i := range in.Limits {
					if err := s.Convert(&in.Limits[i], &out.Limits[i], 0); err != nil {
						return err
					}
				}
			}
			return nil
		},
		func(in *ListMeta, out *newer.ListMeta, s conversion.Scope) error {
			out.SelfLink = in.SelfLink
			out.ResourceVersion = in.ResourceVersion
			return nil
		},
		func(in *newer.ListMeta, out *ListMeta, s conversion.Scope) error {
			out.SelfLink = in.SelfLink
			out.ResourceVersion = in.ResourceVersion
			return nil
		},
		func(in *NFSVolumeSource, out *newer.NFSVolumeSource, s conversion.Scope) error {
			out.Server = in.Server
			out.Path = in.Path
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *newer.NFSVolumeSource, out *NFSVolumeSource, s conversion.Scope) error {
			out.Server = in.Server
			out.Path = in.Path
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *Namespace, out *newer.Namespace, s conversion.Scope) error {
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
		},
		func(in *newer.Namespace, out *Namespace, s conversion.Scope) error {
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
		},
		func(in *NamespaceList, out *newer.NamespaceList, s conversion.Scope) error {
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
		},
		func(in *newer.NamespaceList, out *NamespaceList, s conversion.Scope) error {
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
		},
		func(in *NamespaceSpec, out *newer.NamespaceSpec, s conversion.Scope) error {
			if in.Finalizers != nil {
				out.Finalizers = make([]newer.FinalizerName, len(in.Finalizers))
				for i := range in.Finalizers {
					if err := s.Convert(&in.Finalizers[i], &out.Finalizers[i], 0); err != nil {
						return err
					}
				}
			}
			return nil
		},
		func(in *newer.NamespaceSpec, out *NamespaceSpec, s conversion.Scope) error {
			if in.Finalizers != nil {
				out.Finalizers = make([]FinalizerName, len(in.Finalizers))
				for i := range in.Finalizers {
					if err := s.Convert(&in.Finalizers[i], &out.Finalizers[i], 0); err != nil {
						return err
					}
				}
			}
			return nil
		},
		func(in *NamespaceStatus, out *newer.NamespaceStatus, s conversion.Scope) error {
			out.Phase = newer.NamespacePhase(in.Phase)
			return nil
		},
		func(in *newer.NamespaceStatus, out *NamespaceStatus, s conversion.Scope) error {
			out.Phase = NamespacePhase(in.Phase)
			return nil
		},
		func(in *Node, out *newer.Node, s conversion.Scope) error {
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
		},
		func(in *newer.Node, out *Node, s conversion.Scope) error {
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
		},
		func(in *NodeAddress, out *newer.NodeAddress, s conversion.Scope) error {
			out.Type = newer.NodeAddressType(in.Type)
			out.Address = in.Address
			return nil
		},
		func(in *newer.NodeAddress, out *NodeAddress, s conversion.Scope) error {
			out.Type = NodeAddressType(in.Type)
			out.Address = in.Address
			return nil
		},
		func(in *NodeCondition, out *newer.NodeCondition, s conversion.Scope) error {
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
		},
		func(in *newer.NodeCondition, out *NodeCondition, s conversion.Scope) error {
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
		},
		func(in *NodeList, out *newer.NodeList, s conversion.Scope) error {
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
		},
		func(in *newer.NodeList, out *NodeList, s conversion.Scope) error {
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
		},
		func(in *NodeSpec, out *newer.NodeSpec, s conversion.Scope) error {
			out.PodCIDR = in.PodCIDR
			out.ExternalID = in.ExternalID
			out.Unschedulable = in.Unschedulable
			return nil
		},
		func(in *newer.NodeSpec, out *NodeSpec, s conversion.Scope) error {
			out.PodCIDR = in.PodCIDR
			out.ExternalID = in.ExternalID
			out.Unschedulable = in.Unschedulable
			return nil
		},
		func(in *NodeStatus, out *newer.NodeStatus, s conversion.Scope) error {
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
		},
		func(in *newer.NodeStatus, out *NodeStatus, s conversion.Scope) error {
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
		},
		func(in *NodeSystemInfo, out *newer.NodeSystemInfo, s conversion.Scope) error {
			out.MachineID = in.MachineID
			out.SystemUUID = in.SystemUUID
			out.BootID = in.BootID
			out.KernelVersion = in.KernelVersion
			out.OsImage = in.OsImage
			out.ContainerRuntimeVersion = in.ContainerRuntimeVersion
			out.KubeletVersion = in.KubeletVersion
			out.KubeProxyVersion = in.KubeProxyVersion
			return nil
		},
		func(in *newer.NodeSystemInfo, out *NodeSystemInfo, s conversion.Scope) error {
			out.MachineID = in.MachineID
			out.SystemUUID = in.SystemUUID
			out.BootID = in.BootID
			out.KernelVersion = in.KernelVersion
			out.OsImage = in.OsImage
			out.ContainerRuntimeVersion = in.ContainerRuntimeVersion
			out.KubeletVersion = in.KubeletVersion
			out.KubeProxyVersion = in.KubeProxyVersion
			return nil
		},
		func(in *ObjectMeta, out *newer.ObjectMeta, s conversion.Scope) error {
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
		},
		func(in *newer.ObjectMeta, out *ObjectMeta, s conversion.Scope) error {
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
		},
		func(in *ObjectReference, out *newer.ObjectReference, s conversion.Scope) error {
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.Name = in.Name
			out.UID = in.UID
			out.APIVersion = in.APIVersion
			out.ResourceVersion = in.ResourceVersion
			out.FieldPath = in.FieldPath
			return nil
		},
		func(in *newer.ObjectReference, out *ObjectReference, s conversion.Scope) error {
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.Name = in.Name
			out.UID = in.UID
			out.APIVersion = in.APIVersion
			out.ResourceVersion = in.ResourceVersion
			out.FieldPath = in.FieldPath
			return nil
		},
		func(in *PersistentVolume, out *newer.PersistentVolume, s conversion.Scope) error {
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
		},
		func(in *newer.PersistentVolume, out *PersistentVolume, s conversion.Scope) error {
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
		},
		func(in *PersistentVolumeClaim, out *newer.PersistentVolumeClaim, s conversion.Scope) error {
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
		},
		func(in *newer.PersistentVolumeClaim, out *PersistentVolumeClaim, s conversion.Scope) error {
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
		},
		func(in *PersistentVolumeClaimList, out *newer.PersistentVolumeClaimList, s conversion.Scope) error {
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
		},
		func(in *newer.PersistentVolumeClaimList, out *PersistentVolumeClaimList, s conversion.Scope) error {
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
		},
		func(in *PersistentVolumeClaimSpec, out *newer.PersistentVolumeClaimSpec, s conversion.Scope) error {
			if in.AccessModes != nil {
				out.AccessModes = make([]newer.AccessModeType, len(in.AccessModes))
				for i := range in.AccessModes {
					if err := s.Convert(&in.AccessModes[i], &out.AccessModes[i], 0); err != nil {
						return err
					}
				}
			}
			if err := s.Convert(&in.Resources, &out.Resources, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.PersistentVolumeClaimSpec, out *PersistentVolumeClaimSpec, s conversion.Scope) error {
			if in.AccessModes != nil {
				out.AccessModes = make([]AccessModeType, len(in.AccessModes))
				for i := range in.AccessModes {
					if err := s.Convert(&in.AccessModes[i], &out.AccessModes[i], 0); err != nil {
						return err
					}
				}
			}
			if err := s.Convert(&in.Resources, &out.Resources, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *PersistentVolumeClaimStatus, out *newer.PersistentVolumeClaimStatus, s conversion.Scope) error {
			out.Phase = newer.PersistentVolumeClaimPhase(in.Phase)
			if in.AccessModes != nil {
				out.AccessModes = make([]newer.AccessModeType, len(in.AccessModes))
				for i := range in.AccessModes {
					if err := s.Convert(&in.AccessModes[i], &out.AccessModes[i], 0); err != nil {
						return err
					}
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
		},
		func(in *newer.PersistentVolumeClaimStatus, out *PersistentVolumeClaimStatus, s conversion.Scope) error {
			out.Phase = PersistentVolumeClaimPhase(in.Phase)
			if in.AccessModes != nil {
				out.AccessModes = make([]AccessModeType, len(in.AccessModes))
				for i := range in.AccessModes {
					if err := s.Convert(&in.AccessModes[i], &out.AccessModes[i], 0); err != nil {
						return err
					}
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
		},
		func(in *PersistentVolumeClaimVolumeSource, out *newer.PersistentVolumeClaimVolumeSource, s conversion.Scope) error {
			out.ClaimName = in.ClaimName
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *newer.PersistentVolumeClaimVolumeSource, out *PersistentVolumeClaimVolumeSource, s conversion.Scope) error {
			out.ClaimName = in.ClaimName
			out.ReadOnly = in.ReadOnly
			return nil
		},
		func(in *PersistentVolumeList, out *newer.PersistentVolumeList, s conversion.Scope) error {
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
		},
		func(in *newer.PersistentVolumeList, out *PersistentVolumeList, s conversion.Scope) error {
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
		},
		func(in *PersistentVolumeSource, out *newer.PersistentVolumeSource, s conversion.Scope) error {
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
		},
		func(in *newer.PersistentVolumeSource, out *PersistentVolumeSource, s conversion.Scope) error {
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
		},
		func(in *PersistentVolumeSpec, out *newer.PersistentVolumeSpec, s conversion.Scope) error {
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
					if err := s.Convert(&in.AccessModes[i], &out.AccessModes[i], 0); err != nil {
						return err
					}
				}
			}
			if err := s.Convert(&in.ClaimRef, &out.ClaimRef, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.PersistentVolumeSpec, out *PersistentVolumeSpec, s conversion.Scope) error {
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
					if err := s.Convert(&in.AccessModes[i], &out.AccessModes[i], 0); err != nil {
						return err
					}
				}
			}
			if err := s.Convert(&in.ClaimRef, &out.ClaimRef, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *PersistentVolumeStatus, out *newer.PersistentVolumeStatus, s conversion.Scope) error {
			out.Phase = newer.PersistentVolumePhase(in.Phase)
			return nil
		},
		func(in *newer.PersistentVolumeStatus, out *PersistentVolumeStatus, s conversion.Scope) error {
			out.Phase = PersistentVolumePhase(in.Phase)
			return nil
		},
		func(in *Pod, out *newer.Pod, s conversion.Scope) error {
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
		},
		func(in *newer.Pod, out *Pod, s conversion.Scope) error {
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
		},
		func(in *PodCondition, out *newer.PodCondition, s conversion.Scope) error {
			out.Type = newer.PodConditionType(in.Type)
			out.Status = newer.ConditionStatus(in.Status)
			return nil
		},
		func(in *newer.PodCondition, out *PodCondition, s conversion.Scope) error {
			out.Type = PodConditionType(in.Type)
			out.Status = ConditionStatus(in.Status)
			return nil
		},
		func(in *PodExecOptions, out *newer.PodExecOptions, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Stdin = in.Stdin
			out.Stdout = in.Stdout
			out.Stderr = in.Stderr
			out.TTY = in.TTY
			out.Container = in.Container
			out.Command = in.Command
			return nil
		},
		func(in *newer.PodExecOptions, out *PodExecOptions, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Stdin = in.Stdin
			out.Stdout = in.Stdout
			out.Stderr = in.Stderr
			out.TTY = in.TTY
			out.Container = in.Container
			out.Command = in.Command
			return nil
		},
		func(in *PodList, out *newer.PodList, s conversion.Scope) error {
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
		},
		func(in *newer.PodList, out *PodList, s conversion.Scope) error {
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
		},
		func(in *PodLogOptions, out *newer.PodLogOptions, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Container = in.Container
			out.Follow = in.Follow
			return nil
		},
		func(in *newer.PodLogOptions, out *PodLogOptions, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Container = in.Container
			out.Follow = in.Follow
			return nil
		},
		func(in *PodProxyOptions, out *newer.PodProxyOptions, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Path = in.Path
			return nil
		},
		func(in *newer.PodProxyOptions, out *PodProxyOptions, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Path = in.Path
			return nil
		},
		func(in *EnvVar, out *newer.EnvVar, s conversion.Scope) error {
			out.Name = in.Name
			out.Value = in.Value
			if err := s.Convert(&in.ValueFrom, &out.ValueFrom, 0); err != nil {
				return err
			}

			return nil
		},
		func(in *newer.EnvVar, out *EnvVar, s conversion.Scope) error {
			out.Name = in.Name
			out.Value = in.Value
			if err := s.Convert(&in.ValueFrom, &out.ValueFrom, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *PodSpec, out *newer.PodSpec, s conversion.Scope) error {
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
		},
		func(in *newer.PodSpec, out *PodSpec, s conversion.Scope) error {
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
		},
		func(in *PodStatus, out *newer.PodStatus, s conversion.Scope) error {
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
		},
		func(in *newer.PodStatus, out *PodStatus, s conversion.Scope) error {
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
		},
		func(in *PodStatusResult, out *newer.PodStatusResult, s conversion.Scope) error {
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
		},
		func(in *newer.PodStatusResult, out *PodStatusResult, s conversion.Scope) error {
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
		},
		func(in *PodTemplate, out *newer.PodTemplate, s conversion.Scope) error {
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
		},
		func(in *newer.PodTemplate, out *PodTemplate, s conversion.Scope) error {
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
		},
		func(in *PodTemplateList, out *newer.PodTemplateList, s conversion.Scope) error {
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
		},
		func(in *newer.PodTemplateList, out *PodTemplateList, s conversion.Scope) error {
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
		},
		func(in *PodTemplateSpec, out *newer.PodTemplateSpec, s conversion.Scope) error {
			if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.PodTemplateSpec, out *PodTemplateSpec, s conversion.Scope) error {
			if err := s.Convert(&in.ObjectMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *Probe, out *newer.Probe, s conversion.Scope) error {
			if err := s.Convert(&in.Handler, &out.Handler, 0); err != nil {
				return err
			}
			out.InitialDelaySeconds = in.InitialDelaySeconds
			out.TimeoutSeconds = in.TimeoutSeconds
			return nil
		},
		func(in *newer.Probe, out *Probe, s conversion.Scope) error {
			if err := s.Convert(&in.Handler, &out.Handler, 0); err != nil {
				return err
			}
			out.InitialDelaySeconds = in.InitialDelaySeconds
			out.TimeoutSeconds = in.TimeoutSeconds
			return nil
		},
		func(in *ReplicationController, out *newer.ReplicationController, s conversion.Scope) error {
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
		},
		func(in *newer.ReplicationController, out *ReplicationController, s conversion.Scope) error {
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
		},
		func(in *ReplicationControllerList, out *newer.ReplicationControllerList, s conversion.Scope) error {
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
		},
		func(in *newer.ReplicationControllerList, out *ReplicationControllerList, s conversion.Scope) error {
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
		},
		func(in *ReplicationControllerSpec, out *newer.ReplicationControllerSpec, s conversion.Scope) error {
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
		},
		func(in *newer.ReplicationControllerSpec, out *ReplicationControllerSpec, s conversion.Scope) error {
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
		},
		func(in *ReplicationControllerStatus, out *newer.ReplicationControllerStatus, s conversion.Scope) error {
			out.Replicas = in.Replicas
			return nil
		},
		func(in *newer.ReplicationControllerStatus, out *ReplicationControllerStatus, s conversion.Scope) error {
			out.Replicas = in.Replicas
			return nil
		},
		func(in *ResourceQuota, out *newer.ResourceQuota, s conversion.Scope) error {
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
		},
		func(in *newer.ResourceQuota, out *ResourceQuota, s conversion.Scope) error {
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
		},
		func(in *ResourceQuotaList, out *newer.ResourceQuotaList, s conversion.Scope) error {
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
		},
		func(in *newer.ResourceQuotaList, out *ResourceQuotaList, s conversion.Scope) error {
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
		},
		func(in *ResourceQuotaSpec, out *newer.ResourceQuotaSpec, s conversion.Scope) error {
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
		},
		func(in *newer.ResourceQuotaSpec, out *ResourceQuotaSpec, s conversion.Scope) error {
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
		},
		func(in *ResourceQuotaStatus, out *newer.ResourceQuotaStatus, s conversion.Scope) error {
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
		},
		func(in *newer.ResourceQuotaStatus, out *ResourceQuotaStatus, s conversion.Scope) error {
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
		},
		func(in *ResourceRequirements, out *newer.ResourceRequirements, s conversion.Scope) error {
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
		},
		func(in *newer.ResourceRequirements, out *ResourceRequirements, s conversion.Scope) error {
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
		},
		func(in *Secret, out *newer.Secret, s conversion.Scope) error {
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
		},
		func(in *newer.Secret, out *Secret, s conversion.Scope) error {
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
		},
		func(in *SecretList, out *newer.SecretList, s conversion.Scope) error {
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
		},
		func(in *newer.SecretList, out *SecretList, s conversion.Scope) error {
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
		},
		func(in *SecretVolumeSource, out *newer.SecretVolumeSource, s conversion.Scope) error {
			out.SecretName = in.SecretName
			return nil
		},
		func(in *newer.SecretVolumeSource, out *SecretVolumeSource, s conversion.Scope) error {
			out.SecretName = in.SecretName
			return nil
		},
		func(in *Service, out *newer.Service, s conversion.Scope) error {
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
		},
		func(in *newer.Service, out *Service, s conversion.Scope) error {
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
		},
		func(in *ServiceList, out *newer.ServiceList, s conversion.Scope) error {
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
		},
		func(in *newer.ServiceList, out *ServiceList, s conversion.Scope) error {
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
		},
		func(in *ServicePort, out *newer.ServicePort, s conversion.Scope) error {
			out.Name = in.Name
			out.Protocol = newer.Protocol(in.Protocol)
			out.Port = in.Port
			if err := s.Convert(&in.TargetPort, &out.TargetPort, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.ServicePort, out *ServicePort, s conversion.Scope) error {
			out.Name = in.Name
			out.Protocol = Protocol(in.Protocol)
			out.Port = in.Port
			if err := s.Convert(&in.TargetPort, &out.TargetPort, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ServiceSpec, out *newer.ServiceSpec, s conversion.Scope) error {
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
		},
		func(in *newer.ServiceSpec, out *ServiceSpec, s conversion.Scope) error {
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
		},
		func(in *ServiceStatus, out *newer.ServiceStatus, s conversion.Scope) error {
			return nil
		},
		func(in *newer.ServiceStatus, out *ServiceStatus, s conversion.Scope) error {
			return nil
		},
		func(in *Status, out *newer.Status, s conversion.Scope) error {
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
		},
		func(in *newer.Status, out *Status, s conversion.Scope) error {
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
		},
		func(in *StatusCause, out *newer.StatusCause, s conversion.Scope) error {
			out.Type = newer.CauseType(in.Type)
			out.Message = in.Message
			out.Field = in.Field
			return nil
		},
		func(in *newer.StatusCause, out *StatusCause, s conversion.Scope) error {
			out.Type = CauseType(in.Type)
			out.Message = in.Message
			out.Field = in.Field
			return nil
		},
		func(in *StatusDetails, out *newer.StatusDetails, s conversion.Scope) error {
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
		},
		func(in *newer.StatusDetails, out *StatusDetails, s conversion.Scope) error {
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
		},
		func(in *TCPSocketAction, out *newer.TCPSocketAction, s conversion.Scope) error {
			if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.TCPSocketAction, out *TCPSocketAction, s conversion.Scope) error {
			if err := s.Convert(&in.Port, &out.Port, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *TypeMeta, out *newer.TypeMeta, s conversion.Scope) error {
			out.Kind = in.Kind
			out.APIVersion = in.APIVersion
			return nil
		},
		func(in *newer.TypeMeta, out *TypeMeta, s conversion.Scope) error {
			out.Kind = in.Kind
			out.APIVersion = in.APIVersion
			return nil
		},
		func(in *Volume, out *newer.Volume, s conversion.Scope) error {
			out.Name = in.Name
			if err := s.Convert(&in.VolumeSource, &out.VolumeSource, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *newer.Volume, out *Volume, s conversion.Scope) error {
			out.Name = in.Name
			if err := s.Convert(&in.VolumeSource, &out.VolumeSource, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *VolumeMount, out *newer.VolumeMount, s conversion.Scope) error {
			out.Name = in.Name
			out.ReadOnly = in.ReadOnly
			out.MountPath = in.MountPath
			return nil
		},
		func(in *newer.VolumeMount, out *VolumeMount, s conversion.Scope) error {
			out.Name = in.Name
			out.ReadOnly = in.ReadOnly
			out.MountPath = in.MountPath
			return nil
		},
		func(in *VolumeSource, out *newer.VolumeSource, s conversion.Scope) error {
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
		},
		func(in *newer.VolumeSource, out *VolumeSource, s conversion.Scope) error {
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
		},
	)

	// Add field conversion funcs.
	err = newer.Scheme.AddFieldLabelConversionFunc("v1", "Pod",
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
	err = newer.Scheme.AddFieldLabelConversionFunc("v1", "Node",
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
	err = newer.Scheme.AddFieldLabelConversionFunc("v1", "ReplicationController",
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
	err = newer.Scheme.AddFieldLabelConversionFunc("v1", "Event",
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
	err = newer.Scheme.AddFieldLabelConversionFunc("v1", "Namespace",
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
}
