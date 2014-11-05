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
	"errors"
	"strconv"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

func init() {
	newer.Scheme.AddConversionFuncs(
		// TypeMeta must be split into two objects
		func(in *newer.TypeMeta, out *TypeMeta, s conversion.Scope) error {
			out.Kind = in.Kind
			out.APIVersion = in.APIVersion
			return nil
		},
		func(in *TypeMeta, out *newer.TypeMeta, s conversion.Scope) error {
			out.Kind = in.Kind
			out.APIVersion = in.APIVersion
			return nil
		},

		// ListMeta must be converted to TypeMeta
		func(in *newer.ListMeta, out *TypeMeta, s conversion.Scope) error {
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
		func(in *TypeMeta, out *newer.ListMeta, s conversion.Scope) error {
			out.SelfLink = in.SelfLink
			if in.ResourceVersion != 0 {
				out.ResourceVersion = strconv.FormatUint(in.ResourceVersion, 10)
			} else {
				out.ResourceVersion = ""
			}
			return nil
		},

		// ObjectMeta must be converted to TypeMeta
		func(in *newer.ObjectMeta, out *TypeMeta, s conversion.Scope) error {
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
			return s.Convert(&in.Annotations, &out.Annotations, 0)
		},
		func(in *TypeMeta, out *newer.ObjectMeta, s conversion.Scope) error {
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
			return s.Convert(&in.Annotations, &out.Annotations, 0)
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
		func(in *newer.MinionList, out *MinionList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ListMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
				return err
			}
			out.Minions = out.Items
			return nil
		},
		func(in *MinionList, out *newer.MinionList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ListMeta, 0); err != nil {
				return err
			}
			if len(in.Items) == 0 {
				if err := s.Convert(&in.Minions, &out.Items, 0); err != nil {
					return err
				}
			} else {
				if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
					return err
				}
			}
			return nil
		},

		// Convert all to the new PodCondition constants
		func(in *newer.PodCondition, out *PodStatus, s conversion.Scope) error {
			switch *in {
			case "":
				*out = ""
			case newer.PodPending:
				*out = PodWaiting
			case newer.PodRunning:
				*out = PodRunning
			case newer.PodSucceeded:
				*out = PodTerminated
			case newer.PodFailed:
				*out = PodTerminated
			default:
				return errors.New("The string provided is not a valid PodCondition constant value")
			}

			return nil
		},

		func(in *PodStatus, out *newer.PodCondition, s conversion.Scope) error {
			switch *in {
			case "":
				*out = ""
			case PodWaiting:
				*out = newer.PodPending
			case PodRunning:
				*out = newer.PodRunning
			case PodTerminated:
				// Older API versions did not contain enough info to map to PodFailed
				*out = newer.PodFailed
			default:
				return errors.New("The string provided is not a valid PodCondition constant value")
			}
			return nil
		},

		// Convert all the standard objects
		func(in *newer.Pod, out *Pod, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.DesiredState, &out.DesiredState, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.CurrentState, &out.CurrentState, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.NodeSelector, &out.NodeSelector, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *Pod, out *newer.Pod, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.DesiredState, &out.DesiredState, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.CurrentState, &out.CurrentState, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.NodeSelector, &out.NodeSelector, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *newer.ReplicationController, out *ReplicationController, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.DesiredState, &out.DesiredState, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.CurrentState, &out.CurrentState, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ReplicationController, out *newer.ReplicationController, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.DesiredState, &out.DesiredState, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.CurrentState, &out.CurrentState, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *newer.Service, out *Service, s conversion.Scope) error {

			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			out.Port = in.Spec.Port
			out.Protocol = Protocol(in.Spec.Protocol)
			if err := s.Convert(&in.Spec.Selector, &out.Selector, 0); err != nil {
				return err
			}
			out.CreateExternalLoadBalancer = in.Spec.CreateExternalLoadBalancer
			out.ContainerPort = in.Spec.ContainerPort
			out.PortalIP = in.Spec.PortalIP
			out.ProxyPort = in.Spec.ProxyPort
			return nil
		},
		func(in *Service, out *newer.Service, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			out.Spec.Port = in.Port
			out.Spec.Protocol = newer.Protocol(in.Protocol)
			if err := s.Convert(&in.Selector, &out.Spec.Selector, 0); err != nil {
				return err
			}
			out.Spec.CreateExternalLoadBalancer = in.CreateExternalLoadBalancer
			out.Spec.ContainerPort = in.ContainerPort
			out.Spec.PortalIP = in.PortalIP
			out.Spec.ProxyPort = in.ProxyPort
			return nil
		},

		func(in *newer.Binding, out *Binding, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}

			out.PodID = in.PodID
			out.Host = in.Host

			return nil
		},
		func(in *Binding, out *newer.Binding, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}

			out.PodID = in.PodID
			out.Host = in.Host

			return nil
		},

		func(in *newer.Status, out *Status, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}

			out.Code = in.Code
			out.Message = in.Message
			out.Reason = StatusReason(in.Reason)
			out.Status = in.Status
			return s.Convert(&in.Details, &out.Details, 0)
		},
		func(in *Status, out *newer.Status, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}

			out.Code = in.Code
			out.Message = in.Message
			out.Reason = newer.StatusReason(in.Reason)
			out.Status = in.Status
			return s.Convert(&in.Details, &out.Details, 0)
		},

		func(in *newer.Minion, out *Minion, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			out.HostIP = in.HostIP
			return s.Convert(&in.NodeResources, &out.NodeResources, 0)
		},
		func(in *Minion, out *newer.Minion, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			out.HostIP = in.HostIP
			return s.Convert(&in.NodeResources, &out.NodeResources, 0)
		},

		func(in *newer.BoundPod, out *BoundPod, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}

			return s.Convert(&in.Spec, &out.Spec, 0)
		},
		func(in *BoundPod, out *newer.BoundPod, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}

			return s.Convert(&in.Spec, &out.Spec, 0)
		},

		func(in *newer.BoundPods, out *BoundPods, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Host = in.Host
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *BoundPods, out *newer.BoundPods, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			out.Host = in.Host
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.Endpoints, out *Endpoints, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}

			return s.Convert(&in.Endpoints, &out.Endpoints, 0)
		},
		func(in *Endpoints, out *newer.Endpoints, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}

			return s.Convert(&in.Endpoints, &out.Endpoints, 0)
		},

		func(in *newer.ServerOp, out *ServerOp, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ServerOp, out *newer.ServerOp, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *newer.Event, out *Event, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}

			out.Message = in.Message
			out.Reason = in.Reason
			out.Source = in.Source
			out.Status = in.Status
			out.Timestamp = in.Timestamp
			return s.Convert(&in.InvolvedObject, &out.InvolvedObject, 0)
		},
		func(in *Event, out *newer.Event, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}

			out.Message = in.Message
			out.Reason = in.Reason
			out.Source = in.Source
			out.Status = in.Status
			out.Timestamp = in.Timestamp
			return s.Convert(&in.InvolvedObject, &out.InvolvedObject, 0)
		},

		// Convert all the standard lists
		func(in *newer.PodList, out *PodList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ListMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *PodList, out *newer.PodList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ListMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.ReplicationControllerList, out *ReplicationControllerList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ListMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *ReplicationControllerList, out *newer.ReplicationControllerList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ListMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.ServiceList, out *ServiceList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ListMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *ServiceList, out *newer.ServiceList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ListMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.EndpointsList, out *EndpointsList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ListMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *EndpointsList, out *newer.EndpointsList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ListMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.EventList, out *EventList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ListMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *EventList, out *newer.EventList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ListMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.ServerOpList, out *ServerOpList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ListMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *ServerOpList, out *newer.ServerOpList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ListMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		func(in *newer.ContainerManifestList, out *ContainerManifestList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ListMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},
		func(in *ContainerManifestList, out *newer.ContainerManifestList, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ListMeta, 0); err != nil {
				return err
			}
			return s.Convert(&in.Items, &out.Items, 0)
		},

		// Object ID <-> Name
		// TODO: amend the conversion package to allow overriding specific fields.
		func(in *ObjectReference, out *newer.ObjectReference, s conversion.Scope) error {
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.Name = in.ID
			out.UID = in.UID
			out.APIVersion = in.APIVersion
			out.ResourceVersion = in.ResourceVersion
			out.FieldPath = in.FieldPath
			return nil
		},
		func(in *newer.ObjectReference, out *ObjectReference, s conversion.Scope) error {
			out.ID = in.Name
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.ID = in.Name
			out.UID = in.UID
			out.APIVersion = in.APIVersion
			out.ResourceVersion = in.ResourceVersion
			out.FieldPath = in.FieldPath
			return nil
		},
	)
}
