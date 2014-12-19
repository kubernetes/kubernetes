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
	// Our TypeMeta was split into two different structs.
	newer.Scheme.AddStructFieldConversion(TypeMeta{}, "TypeMeta", newer.TypeMeta{}, "TypeMeta")
	newer.Scheme.AddStructFieldConversion(TypeMeta{}, "TypeMeta", newer.ObjectMeta{}, "ObjectMeta")
	newer.Scheme.AddStructFieldConversion(TypeMeta{}, "TypeMeta", newer.ListMeta{}, "ListMeta")

	newer.Scheme.AddStructFieldConversion(newer.TypeMeta{}, "TypeMeta", TypeMeta{}, "TypeMeta")
	newer.Scheme.AddStructFieldConversion(newer.ObjectMeta{}, "ObjectMeta", TypeMeta{}, "TypeMeta")
	newer.Scheme.AddStructFieldConversion(newer.ListMeta{}, "ListMeta", TypeMeta{}, "TypeMeta")

	// TODO: scope this to a specific type once that becomes available and remove the Event conversion functions below
	// newer.Scheme.AddStructFieldConversion(string(""), "Status", string(""), "Condition")
	// newer.Scheme.AddStructFieldConversion(string(""), "Condition", string(""), "Status")

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
		func(in *newer.NodeList, out *MinionList, s conversion.Scope) error {
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
		func(in *MinionList, out *newer.NodeList, s conversion.Scope) error {
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

		func(in *newer.PodStatus, out *PodState, s conversion.Scope) error {
			if err := s.Convert(&in.Phase, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Info, &out.Info, 0); err != nil {
				return err
			}
			out.Message = in.Message
			out.Host = in.Host
			out.HostIP = in.HostIP
			out.PodIP = in.PodIP
			return nil
		},
		func(in *PodState, out *newer.PodStatus, s conversion.Scope) error {
			if err := s.Convert(&in.Status, &out.Phase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Info, &out.Info, 0); err != nil {
				return err
			}

			out.Message = in.Message
			out.Host = in.Host
			out.HostIP = in.HostIP
			out.PodIP = in.PodIP
			return nil
		},
		func(in *newer.PodSpec, out *PodState, s conversion.Scope) error {
			if err := s.Convert(&in, &out.Manifest, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *PodState, out *newer.PodSpec, s conversion.Scope) error {
			if err := s.Convert(&in.Manifest, &out, 0); err != nil {
				return err
			}
			return nil
		},

		// Convert all to the new PodPhase constants
		func(in *newer.PodPhase, out *PodStatus, s conversion.Scope) error {
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
			case newer.PodUnknown:
				*out = PodUnknown
			default:
				return errors.New("The string provided is not a valid PodPhase constant value")
			}

			return nil
		},

		func(in *PodStatus, out *newer.PodPhase, s conversion.Scope) error {
			switch *in {
			case "":
				*out = ""
			case PodWaiting:
				*out = newer.PodPending
			case PodRunning:
				*out = newer.PodRunning
			case PodTerminated:
				// Older API versions did not contain enough info to map to PodSucceeded
				*out = newer.PodFailed
			case PodUnknown:
				*out = newer.PodUnknown
			default:
				return errors.New("The string provided is not a valid PodPhase constant value")
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
			// TODO: Change this to use in.ObjectMeta.Labels.
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec, &out.DesiredState.Manifest, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.CurrentState, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec.NodeSelector, &out.NodeSelector, 0); err != nil {
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
			if err := s.Convert(&in.DesiredState.Manifest, &out.Spec, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.CurrentState, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.NodeSelector, &out.Spec.NodeSelector, 0); err != nil {
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

			if err := s.Convert(&in.Spec, &out.DesiredState, 0); err != nil {
				return err
			}
			out.CurrentState.Replicas = in.Status.Replicas
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

			if err := s.Convert(&in.DesiredState, &out.Spec, 0); err != nil {
				return err
			}
			out.Status.Replicas = in.CurrentState.Replicas
			return nil
		},

		func(in *newer.ReplicationControllerSpec, out *ReplicationControllerState, s conversion.Scope) error {
			out.Replicas = in.Replicas
			if err := s.Convert(&in.Selector, &out.ReplicaSelector, 0); err != nil {
				return err
			}
			if in.TemplateRef != nil && in.Template == nil {
				return errors.New("objects with a template ref cannot be converted to older objects, must populate template")
			}
			if in.Template != nil {
				if err := s.Convert(in.Template, &out.PodTemplate, 0); err != nil {
					return err
				}
			}
			return nil
		},
		func(in *ReplicationControllerState, out *newer.ReplicationControllerSpec, s conversion.Scope) error {
			out.Replicas = in.Replicas
			if err := s.Convert(&in.ReplicaSelector, &out.Selector, 0); err != nil {
				return err
			}
			out.Template = &newer.PodTemplateSpec{}
			if err := s.Convert(&in.PodTemplate, out.Template, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *newer.PodTemplateSpec, out *PodTemplate, s conversion.Scope) error {
			if err := s.Convert(&in.Spec, &out.DesiredState.Manifest, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *PodTemplate, out *newer.PodTemplateSpec, s conversion.Scope) error {
			if err := s.Convert(&in.DesiredState.Manifest, &out.Spec, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.ObjectMeta.Labels, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *newer.PodSpec, out *BoundPod, s conversion.Scope) error {
			if err := s.Convert(&in, &out.Spec, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *BoundPod, out *newer.PodSpec, s conversion.Scope) error {
			if err := s.Convert(&in.Spec, &out, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *newer.PodSpec, out *ContainerManifest, s conversion.Scope) error {
			if err := s.Convert(&in.Volumes, &out.Volumes, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Containers, &out.Containers, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.RestartPolicy, &out.RestartPolicy, 0); err != nil {
				return err
			}
			out.Version = "v1beta2"
			return nil
		},
		func(in *ContainerManifest, out *newer.PodSpec, s conversion.Scope) error {
			if err := s.Convert(&in.Volumes, &out.Volumes, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Containers, &out.Containers, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.RestartPolicy, &out.RestartPolicy, 0); err != nil {
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
			out.PublicIPs = in.Spec.PublicIPs
			out.ContainerPort = in.Spec.ContainerPort
			out.PortalIP = in.Spec.PortalIP
			out.ProxyPort = in.Spec.ProxyPort
			if err := s.Convert(&in.Spec.SessionAffinity, &out.SessionAffinity, 0); err != nil {
				return err
			}

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
			out.Spec.PublicIPs = in.PublicIPs
			out.Spec.ContainerPort = in.ContainerPort
			out.Spec.PortalIP = in.PortalIP
			out.Spec.ProxyPort = in.ProxyPort
			if err := s.Convert(&in.SessionAffinity, &out.Spec.SessionAffinity, 0); err != nil {
				return err
			}

			return nil
		},

		func(in *newer.Node, out *Minion, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta.Labels, &out.Labels, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status.Phase, &out.Status.Phase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status.Conditions, &out.Status.Conditions, 0); err != nil {
				return err
			}

			out.HostIP = in.Status.HostIP
			return s.Convert(&in.Spec.Capacity, &out.NodeResources.Capacity, 0)
		},
		func(in *Minion, out *newer.Node, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.ObjectMeta.Labels, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status.Phase, &out.Status.Phase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status.Conditions, &out.Status.Conditions, 0); err != nil {
				return err
			}

			out.Status.HostIP = in.HostIP
			return s.Convert(&in.NodeResources.Capacity, &out.Spec.Capacity, 0)
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
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.ID = in.Name
			out.UID = in.UID
			out.APIVersion = in.APIVersion
			out.ResourceVersion = in.ResourceVersion
			out.FieldPath = in.FieldPath
			return nil
		},

		// Event Status -> Condition
		// TODO: remove this when it becomes possible to specify a field name conversion on a specific type
		func(in *newer.Event, out *Event, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Status = in.Condition
			out.Reason = in.Reason
			out.Message = in.Message
			out.Source = in.Source
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
			out.Condition = in.Status
			out.Reason = in.Reason
			out.Message = in.Message
			out.Source = in.Source
			out.Timestamp = in.Timestamp
			return s.Convert(&in.InvolvedObject, &out.InvolvedObject, 0)
		},
	)
}
