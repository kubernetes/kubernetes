/*
Copyright 2014 Google Inc. All rights reserved.

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

package v1beta1

import (
	"strconv"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

func init() {
	newer.Scheme.AddConversionFuncs(
		// TypeMeta must be converted to JSONBase
		func(in *newer.TypeMeta, out *JSONBase, s conversion.Scope) error {
			out.Kind = in.Kind
			out.APIVersion = in.APIVersion
			return nil
		},
		func(in *JSONBase, out *newer.TypeMeta, s conversion.Scope) error {
			out.Kind = in.Kind
			out.APIVersion = in.APIVersion
			return nil
		},

		// ListMeta must be converted to JSONBase
		func(in *newer.ListMeta, out *JSONBase, s conversion.Scope) error {
			out.SelfLink = in.SelfLink
			if len(in.ResourceVersion) > 0 {
				v, err := strconv.ParseUint(in.ResourceVersion, 10, 64)
				if err != nil {
					return err
				}
				out.ResourceVersion = v
			}
			return nil
		},
		func(in *JSONBase, out *newer.ListMeta, s conversion.Scope) error {
			out.SelfLink = in.SelfLink
			if in.ResourceVersion != 0 {
				out.ResourceVersion = strconv.FormatUint(in.ResourceVersion, 10)
			} else {
				out.ResourceVersion = ""
			}
			return nil
		},

		// ObjectMeta must be converted to JSONBase
		func(in *newer.ObjectMeta, out *JSONBase, s conversion.Scope) error {
			out.Namespace = in.Namespace
			out.ID = in.Name
			out.UID = in.UID
			out.CreationTimestamp = in.CreationTimestamp
			out.SelfLink = in.SelfLink
			if len(in.ResourceVersion) > 0 {
				v, err := strconv.ParseUint(in.ResourceVersion, 10, 64)
				if err != nil {
					return err
				}
				out.ResourceVersion = v
			}
			return nil
		},
		func(in *JSONBase, out *newer.ObjectMeta, s conversion.Scope) error {
			out.Namespace = in.Namespace
			out.Name = in.ID
			out.UID = in.UID
			out.CreationTimestamp = in.CreationTimestamp
			out.SelfLink = in.SelfLink
			if in.ResourceVersion != 0 {
				out.ResourceVersion = strconv.FormatUint(in.ResourceVersion, 10)
			} else {
				out.ResourceVersion = ""
			}
			return nil
		},

		// EnvVar's Key is deprecated in favor of Name.
		func(in *newer.EnvVar, out *EnvVar, s conversion.Scope) error {
			out.Value = in.Value
			out.Key = in.Name
			out.Name = in.Name
			return nil
		},
		func(in *EnvVar, out *newer.EnvVar, s conversion.Scope) error {
			out.Value = in.Value
			if in.Name != "" {
				out.Name = in.Name
			} else {
				out.Name = in.Key
			}
			return nil
		},

		// Path & MountType are deprecated.
		func(in *newer.VolumeMount, out *VolumeMount, s conversion.Scope) error {
			out.Name = in.Name
			out.ReadOnly = in.ReadOnly
			out.MountPath = in.MountPath
			out.Path = in.MountPath
			out.MountType = "" // MountType is ignored.
			return nil
		},
		func(in *VolumeMount, out *newer.VolumeMount, s conversion.Scope) error {
			out.Name = in.Name
			out.ReadOnly = in.ReadOnly
			if in.MountPath == "" {
				out.MountPath = in.Path
			} else {
				out.MountPath = in.MountPath
			}
			return nil
		},

		// MinionList.Items had a wrong name in v1beta1
		func(in *newer.NodeList, out *MinionList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
				return err
			}
			out.Minions = out.Items
			return nil
		},
		func(in *MinionList, out *newer.NodeList, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			if len(in.Items) == 0 {
				s.Convert(&in.Minions, &out.Items, 0)
			} else {
				s.Convert(&in.Items, &out.Items, 0)
			}
			return nil
		},

		// Convert all the standard objects
		func(in *newer.Pod, out *Pod, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			out.Labels = in.Metadata.Labels

			out.DesiredState.Manifest.Version = "v1beta2"
			out.DesiredState.Manifest.ID = in.Metadata.Name
			out.DesiredState.Manifest.UUID = in.Metadata.UID
			if err := s.Convert(&in.Spec.Containers, &out.DesiredState.Manifest.Containers, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec.RestartPolicy, &out.DesiredState.Manifest.RestartPolicy, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec.Volumes, &out.DesiredState.Manifest.Volumes, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.Status.Info, &out.CurrentState.Info, 0); err != nil {
				return err
			}
			out.CurrentState.Host = in.Status.Host
			out.CurrentState.HostIP = in.Status.HostIP
			out.CurrentState.Status = PodStatus(in.Status.Condition)
			out.CurrentState.PodIP = in.Status.PodIP
			return nil
		},
		func(in *Pod, out *newer.Pod, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			out.Metadata.Labels = in.Labels

			if err := s.Convert(&in.DesiredState.Manifest.Containers, &out.Spec.Containers, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.DesiredState.Manifest.RestartPolicy, &out.Spec.RestartPolicy, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.DesiredState.Manifest.Volumes, &out.Spec.Volumes, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.CurrentState.Info, &out.Status.Info, 0); err != nil {
				return err
			}
			out.Status.Host = in.CurrentState.Host
			out.Status.HostIP = in.CurrentState.HostIP
			out.Status.Condition = newer.PodCondition(in.CurrentState.Status)
			out.Status.PodIP = in.CurrentState.PodIP

			return nil
		},

		// Added Type field
		func(in *newer.LivenessProbe, out *LivenessProbe, s conversion.Scope) error {
			if err := s.Convert(&in.HTTPGet, &out.HTTPGet, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TCPSocket, &out.TCPSocket, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Exec, &out.Exec, 0); err != nil {
				return err
			}
			out.InitialDelaySeconds = in.InitialDelaySeconds
			return nil
		},
		func(in *LivenessProbe, out *newer.LivenessProbe, s conversion.Scope) error {
			if err := s.Convert(&in.TCPSocket, &out.TCPSocket, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.HTTPGet, &out.HTTPGet, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Exec, &out.Exec, 0); err != nil {
				return err
			}
			switch {
			case in.HTTPGet != nil:
				out.Type = "HTTP"
			case in.TCPSocket != nil:
				out.Type = "TCP"
			case in.Exec != nil:
				out.Type = "Exec"
			}
			out.InitialDelaySeconds = in.InitialDelaySeconds
			return nil
		},

		func(in *newer.Operation, out *ServerOp, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ServerOp, out *newer.Operation, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			return nil
		},

		// Convert all the standard lists
		func(in *newer.PodList, out *PodList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *PodList, out *newer.PodList, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.ReplicationControllerList, out *ReplicationControllerList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *ReplicationControllerList, out *newer.ReplicationControllerList, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.ServiceList, out *ServiceList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *ServiceList, out *newer.ServiceList, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.EndpointsList, out *EndpointsList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *EndpointsList, out *newer.EndpointsList, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		// Renamed ServerOpList -> OperationList
		func(in *newer.OperationList, out *ServerOpList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *ServerOpList, out *newer.OperationList, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.EventList, out *EventList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.JSONBase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Metadata, &out.JSONBase, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *EventList, out *newer.EventList, s conversion.Scope) error {
			if err := s.Convert(&in.JSONBase, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.JSONBase, &out.Metadata, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
	)
}
