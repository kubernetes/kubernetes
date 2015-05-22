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

package v1beta2

import (
	"fmt"
	"net"
	"reflect"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func addConversionFuncs() {
	// Our TypeMeta was split into two different structs.
	api.Scheme.AddStructFieldConversion(TypeMeta{}, "TypeMeta", api.TypeMeta{}, "TypeMeta")
	api.Scheme.AddStructFieldConversion(TypeMeta{}, "TypeMeta", api.ObjectMeta{}, "ObjectMeta")
	api.Scheme.AddStructFieldConversion(TypeMeta{}, "TypeMeta", api.ListMeta{}, "ListMeta")

	api.Scheme.AddStructFieldConversion(api.TypeMeta{}, "TypeMeta", TypeMeta{}, "TypeMeta")
	api.Scheme.AddStructFieldConversion(api.ObjectMeta{}, "ObjectMeta", TypeMeta{}, "TypeMeta")
	api.Scheme.AddStructFieldConversion(api.ListMeta{}, "ListMeta", TypeMeta{}, "TypeMeta")
	api.Scheme.AddStructFieldConversion(api.Endpoints{}, "Endpoints", Endpoints{}, "Endpoints")

	// TODO: scope this to a specific type once that becomes available and remove the Event conversion functions below
	// api.Scheme.AddStructFieldConversion(string(""), "Status", string(""), "Condition")
	// api.Scheme.AddStructFieldConversion(string(""), "Condition", string(""), "Status")

	err := api.Scheme.AddConversionFuncs(
		// TypeMeta must be split into two objects
		func(in *api.TypeMeta, out *TypeMeta, s conversion.Scope) error {
			out.Kind = in.Kind
			out.APIVersion = in.APIVersion
			return nil
		},
		func(in *TypeMeta, out *api.TypeMeta, s conversion.Scope) error {
			out.Kind = in.Kind
			out.APIVersion = in.APIVersion
			return nil
		},

		// ListMeta must be converted to TypeMeta
		func(in *api.ListMeta, out *TypeMeta, s conversion.Scope) error {
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
		func(in *TypeMeta, out *api.ListMeta, s conversion.Scope) error {
			out.SelfLink = in.SelfLink
			if in.ResourceVersion != 0 {
				out.ResourceVersion = strconv.FormatUint(in.ResourceVersion, 10)
			} else {
				out.ResourceVersion = ""
			}
			return nil
		},

		// ObjectMeta must be converted to TypeMeta
		func(in *api.ObjectMeta, out *TypeMeta, s conversion.Scope) error {
			out.Namespace = in.Namespace
			out.ID = in.Name
			out.GenerateName = in.GenerateName
			out.UID = in.UID
			out.CreationTimestamp = in.CreationTimestamp
			out.DeletionTimestamp = in.DeletionTimestamp
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
		func(in *TypeMeta, out *api.ObjectMeta, s conversion.Scope) error {
			out.Namespace = in.Namespace
			out.Name = in.ID
			out.GenerateName = in.GenerateName
			out.UID = in.UID
			out.CreationTimestamp = in.CreationTimestamp
			out.DeletionTimestamp = in.DeletionTimestamp
			out.SelfLink = in.SelfLink
			if in.ResourceVersion != 0 {
				out.ResourceVersion = strconv.FormatUint(in.ResourceVersion, 10)
			} else {
				out.ResourceVersion = ""
			}
			return s.Convert(&in.Annotations, &out.Annotations, 0)
		},

		// Convert all to the new PodPhase constants
		func(in *api.PodPhase, out *PodStatus, s conversion.Scope) error {
			switch *in {
			case "":
				*out = ""
			case api.PodPending:
				*out = PodWaiting
			case api.PodRunning:
				*out = PodRunning
			case api.PodSucceeded:
				*out = PodSucceeded
			case api.PodFailed:
				*out = PodTerminated
			case api.PodUnknown:
				*out = PodUnknown
			default:
				return &api.ConversionError{
					In:      in,
					Out:     out,
					Message: "The string provided is not a valid PodPhase constant value",
				}
			}

			return nil
		},

		func(in *PodStatus, out *api.PodPhase, s conversion.Scope) error {
			switch *in {
			case "":
				*out = ""
			case PodWaiting:
				*out = api.PodPending
			case PodRunning:
				*out = api.PodRunning
			case PodTerminated:
				// Older API versions did not contain enough info to map to PodSucceeded
				*out = api.PodFailed
			case PodSucceeded:
				*out = api.PodSucceeded
			case PodUnknown:
				*out = api.PodUnknown
			default:
				return &api.ConversionError{
					In:      in,
					Out:     out,
					Message: "The string provided is not a valid PodPhase constant value",
				}
			}
			return nil
		},

		// Convert all the standard objects
		func(in *api.Pod, out *Pod, s conversion.Scope) error {
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
			out.DesiredState.Host = in.Spec.Host
			out.CurrentState.Host = in.Spec.Host
			out.ServiceAccount = in.Spec.ServiceAccount
			if err := s.Convert(&in.Status, &out.CurrentState, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec.NodeSelector, &out.NodeSelector, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *Pod, out *api.Pod, s conversion.Scope) error {
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
			out.Spec.ServiceAccount = in.ServiceAccount
			out.Spec.Host = in.DesiredState.Host
			if err := s.Convert(&in.CurrentState, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.NodeSelector, &out.Spec.NodeSelector, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.ReplicationController, out *ReplicationController, s conversion.Scope) error {
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
		func(in *ReplicationController, out *api.ReplicationController, s conversion.Scope) error {
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

		func(in *api.ReplicationControllerSpec, out *ReplicationControllerState, s conversion.Scope) error {
			out.Replicas = in.Replicas
			if err := s.Convert(&in.Selector, &out.ReplicaSelector, 0); err != nil {
				return err
			}
			if in.TemplateRef != nil && in.Template == nil {
				return &api.ConversionError{
					In:      in,
					Out:     out,
					Message: "objects with a template ref cannot be converted to older objects, must populate template",
				}
			}
			if in.Template != nil {
				if err := s.Convert(in.Template, &out.PodTemplate, 0); err != nil {
					return err
				}
			}
			return nil
		},
		func(in *ReplicationControllerState, out *api.ReplicationControllerSpec, s conversion.Scope) error {
			out.Replicas = in.Replicas
			if err := s.Convert(&in.ReplicaSelector, &out.Selector, 0); err != nil {
				return err
			}
			out.Template = &api.PodTemplateSpec{}
			if err := s.Convert(&in.PodTemplate, out.Template, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.PodTemplateSpec, out *PodTemplate, s conversion.Scope) error {
			if err := s.Convert(&in.Spec, &out.DesiredState.Manifest, 0); err != nil {
				return err
			}
			out.DesiredState.Host = in.Spec.Host
			out.ServiceAccount = in.Spec.ServiceAccount
			if err := s.Convert(&in.Spec.NodeSelector, &out.NodeSelector, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta.Labels, &out.Labels, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta.Annotations, &out.Annotations, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *PodTemplate, out *api.PodTemplateSpec, s conversion.Scope) error {
			if err := s.Convert(&in.DesiredState.Manifest, &out.Spec, 0); err != nil {
				return err
			}
			out.Spec.Host = in.DesiredState.Host
			out.Spec.ServiceAccount = in.ServiceAccount
			if err := s.Convert(&in.NodeSelector, &out.Spec.NodeSelector, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.ObjectMeta.Labels, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Annotations, &out.ObjectMeta.Annotations, 0); err != nil {
				return err
			}
			return nil
		},
		// Converts internal Container to v1beta2.Container.
		// Fields 'CPU' and 'Memory' are not present in the internal Container object.
		// Hence the need for a custom conversion function.
		func(in *api.Container, out *Container, s conversion.Scope) error {
			if err := s.Convert(&in.Name, &out.Name, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Image, &out.Image, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Command, &out.Entrypoint, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Args, &out.Command, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.WorkingDir, &out.WorkingDir, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Ports, &out.Ports, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Env, &out.Env, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Resources, &out.Resources, 0); err != nil {
				return err
			}
			if err := s.Convert(in.Resources.Limits.Cpu(), &out.CPU, 0); err != nil {
				return err
			}
			if err := s.Convert(in.Resources.Limits.Memory(), &out.Memory, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.VolumeMounts, &out.VolumeMounts, 0); err != nil {
				return err
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
			if err := s.Convert(&in.TerminationMessagePath, &out.TerminationMessagePath, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ImagePullPolicy, &out.ImagePullPolicy, 0); err != nil {
				return err
			}
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
		},
		// Internal API does not support CPU to be specified via an explicit field.
		// Hence it must be stored in Container.Resources.
		func(in *int, out *api.ResourceList, s conversion.Scope) error {
			if *in == 0 {
				return nil
			}
			quantity := resource.Quantity{}
			if err := s.Convert(in, &quantity, 0); err != nil {
				return err
			}
			(*out)[api.ResourceCPU] = quantity

			return nil
		},
		// Internal API does not support Memory to be specified via an explicit field.
		// Hence it must be stored in Container.Resources.
		func(in *int64, out *api.ResourceList, s conversion.Scope) error {
			if *in == 0 {
				return nil
			}
			quantity := resource.Quantity{}
			if err := s.Convert(in, &quantity, 0); err != nil {
				return err
			}
			(*out)[api.ResourceMemory] = quantity

			return nil
		},
		// Converts v1beta2.Container to internal api.Container.
		// Fields 'CPU' and 'Memory' are not present in the internal api.Container object.
		// Hence the need for a custom conversion function.
		func(in *Container, out *api.Container, s conversion.Scope) error {
			if err := s.Convert(&in.Name, &out.Name, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Image, &out.Image, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Command, &out.Args, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Entrypoint, &out.Command, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.WorkingDir, &out.WorkingDir, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Ports, &out.Ports, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Env, &out.Env, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Resources, &out.Resources, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.CPU, &out.Resources.Limits, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Memory, &out.Resources.Limits, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.VolumeMounts, &out.VolumeMounts, 0); err != nil {
				return err
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
			if err := s.Convert(&in.TerminationMessagePath, &out.TerminationMessagePath, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ImagePullPolicy, &out.ImagePullPolicy, 0); err != nil {
				return err
			}
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
		},
		func(in *api.PodSpec, out *ContainerManifest, s conversion.Scope) error {
			if err := s.Convert(&in.Volumes, &out.Volumes, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Containers, &out.Containers, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.RestartPolicy, &out.RestartPolicy, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ImagePullSecrets, &out.ImagePullSecrets, 0); err != nil {
				return err
			}
			if in.TerminationGracePeriodSeconds != nil {
				out.TerminationGracePeriodSeconds = new(int64)
				*out.TerminationGracePeriodSeconds = *in.TerminationGracePeriodSeconds
			}
			if in.ActiveDeadlineSeconds != nil {
				out.ActiveDeadlineSeconds = new(int64)
				*out.ActiveDeadlineSeconds = *in.ActiveDeadlineSeconds
			}
			out.DNSPolicy = DNSPolicy(in.DNSPolicy)
			out.Version = "v1beta2"
			out.HostNetwork = in.HostNetwork
			return nil
		},
		func(in *ContainerManifest, out *api.PodSpec, s conversion.Scope) error {
			if err := s.Convert(&in.Volumes, &out.Volumes, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Containers, &out.Containers, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.RestartPolicy, &out.RestartPolicy, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ImagePullSecrets, &out.ImagePullSecrets, 0); err != nil {
				return err
			}
			if in.TerminationGracePeriodSeconds != nil {
				out.TerminationGracePeriodSeconds = new(int64)
				*out.TerminationGracePeriodSeconds = *in.TerminationGracePeriodSeconds
			}
			if in.ActiveDeadlineSeconds != nil {
				out.ActiveDeadlineSeconds = new(int64)
				*out.ActiveDeadlineSeconds = *in.ActiveDeadlineSeconds
			}
			out.DNSPolicy = api.DNSPolicy(in.DNSPolicy)
			out.HostNetwork = in.HostNetwork
			return nil
		},

		func(in *api.PodStatus, out *PodState, s conversion.Scope) error {
			if err := s.Convert(&in.Phase, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ContainerStatuses, &out.Info, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Conditions, &out.Conditions, 0); err != nil {
				return err
			}
			out.Message = in.Message
			out.HostIP = in.HostIP
			out.PodIP = in.PodIP
			return nil
		},
		func(in *PodState, out *api.PodStatus, s conversion.Scope) error {
			if err := s.Convert(&in.Status, &out.Phase, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Info, &out.ContainerStatuses, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Conditions, &out.Conditions, 0); err != nil {
				return err
			}
			out.Message = in.Message
			out.HostIP = in.HostIP
			out.PodIP = in.PodIP
			return nil
		},

		func(in *[]api.ContainerStatus, out *PodInfo, s conversion.Scope) error {
			*out = make(map[string]ContainerStatus)
			for _, st := range *in {
				v := ContainerStatus{}
				if err := s.Convert(&st, &v, 0); err != nil {
					return err
				}
				(*out)[st.Name] = v
			}
			return nil
		},
		func(in *PodInfo, out *[]api.ContainerStatus, s conversion.Scope) error {
			for k, v := range *in {
				st := api.ContainerStatus{}
				if err := s.Convert(&v, &st, 0); err != nil {
					return err
				}
				st.Name = k
				*out = append(*out, st)
			}
			return nil
		},

		func(in *api.ContainerStatus, out *ContainerStatus, s conversion.Scope) error {
			if err := s.Convert(&in.State, &out.State, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.LastTerminationState, &out.LastTerminationState, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Ready, &out.Ready, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.RestartCount, &out.RestartCount, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Image, &out.Image, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ImageID, &out.ImageID, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ContainerID, &out.ContainerID, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ContainerStatus, out *api.ContainerStatus, s conversion.Scope) error {
			if err := s.Convert(&in.State, &out.State, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.LastTerminationState, &out.LastTerminationState, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Ready, &out.Ready, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.RestartCount, &out.RestartCount, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Image, &out.Image, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ImageID, &out.ImageID, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ContainerID, &out.ContainerID, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.PodStatusResult, out *PodStatusResult, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.State, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *PodStatusResult, out *api.PodStatusResult, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.State, &out.Status, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.PodSpec, out *PodState, s conversion.Scope) error {
			if err := s.Convert(&in, &out.Manifest, 0); err != nil {
				return err
			}
			out.Host = in.Host
			return nil
		},
		func(in *PodState, out *api.PodSpec, s conversion.Scope) error {
			if err := s.Convert(&in.Manifest, &out, 0); err != nil {
				return err
			}
			out.Host = in.Host
			return nil
		},
		func(in *api.Service, out *Service, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			// Produce legacy fields.
			out.Protocol = ProtocolTCP
			if len(in.Spec.Ports) > 0 {
				out.PortName = in.Spec.Ports[0].Name
				out.Port = in.Spec.Ports[0].Port
				out.Protocol = Protocol(in.Spec.Ports[0].Protocol)
				out.ContainerPort = in.Spec.Ports[0].TargetPort
			}
			// Copy modern fields.
			for i := range in.Spec.Ports {
				out.Ports = append(out.Ports, ServicePort{
					Name:          in.Spec.Ports[i].Name,
					Port:          in.Spec.Ports[i].Port,
					Protocol:      Protocol(in.Spec.Ports[i].Protocol),
					ContainerPort: in.Spec.Ports[i].TargetPort,
					NodePort:      in.Spec.Ports[i].NodePort,
				})
			}

			if err := s.Convert(&in.Spec.Selector, &out.Selector, 0); err != nil {
				return err
			}
			out.PublicIPs = in.Spec.DeprecatedPublicIPs
			out.PortalIP = in.Spec.PortalIP
			if err := s.Convert(&in.Spec.SessionAffinity, &out.SessionAffinity, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.Status.LoadBalancer, &out.LoadBalancerStatus, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.Spec.Type, &out.Type, 0); err != nil {
				return err
			}
			out.CreateExternalLoadBalancer = in.Spec.Type == api.ServiceTypeLoadBalancer

			return nil
		},
		func(in *Service, out *api.Service, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}

			if len(in.Ports) == 0 && in.Port != 0 {
				// Use legacy fields to produce modern fields.
				out.Spec.Ports = append(out.Spec.Ports, api.ServicePort{
					Name:       in.PortName,
					Port:       in.Port,
					Protocol:   api.Protocol(in.Protocol),
					TargetPort: in.ContainerPort,
				})
			} else {
				// Use modern fields, ignore legacy.
				for i := range in.Ports {
					out.Spec.Ports = append(out.Spec.Ports, api.ServicePort{
						Name:       in.Ports[i].Name,
						Port:       in.Ports[i].Port,
						Protocol:   api.Protocol(in.Ports[i].Protocol),
						TargetPort: in.Ports[i].ContainerPort,
						NodePort:   in.Ports[i].NodePort,
					})
				}
			}

			if err := s.Convert(&in.Selector, &out.Spec.Selector, 0); err != nil {
				return err
			}
			out.Spec.DeprecatedPublicIPs = in.PublicIPs
			out.Spec.PortalIP = in.PortalIP
			if err := s.Convert(&in.SessionAffinity, &out.Spec.SessionAffinity, 0); err != nil {
				return err
			}

			if err := s.Convert(&in.LoadBalancerStatus, &out.Status.LoadBalancer, 0); err != nil {
				return err
			}

			typeIn := in.Type
			if typeIn == "" {
				if in.CreateExternalLoadBalancer {
					typeIn = ServiceTypeLoadBalancer
				} else {
					typeIn = ServiceTypeClusterIP
				}
			}
			if err := s.Convert(&typeIn, &out.Spec.Type, 0); err != nil {
				return err
			}

			return nil
		},

		func(in *api.Node, out *Minion, s conversion.Scope) error {
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
			if err := s.Convert(&in.Status.Addresses, &out.Status.Addresses, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status.NodeInfo, &out.Status.NodeInfo, 0); err != nil {
				return err
			}

			for _, address := range in.Status.Addresses {
				if address.Type == api.NodeLegacyHostIP {
					out.HostIP = address.Address
				}
			}
			out.PodCIDR = in.Spec.PodCIDR
			out.ExternalID = in.Spec.ExternalID
			out.Unschedulable = in.Spec.Unschedulable
			return s.Convert(&in.Status.Capacity, &out.NodeResources.Capacity, 0)
		},
		func(in *Minion, out *api.Node, s conversion.Scope) error {
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
			if err := s.Convert(&in.Status.Addresses, &out.Status.Addresses, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status.NodeInfo, &out.Status.NodeInfo, 0); err != nil {
				return err
			}

			if in.HostIP != "" {
				api.AddToNodeAddresses(&out.Status.Addresses,
					api.NodeAddress{Type: api.NodeLegacyHostIP, Address: in.HostIP})
			}
			out.Spec.PodCIDR = in.PodCIDR
			out.Spec.ExternalID = in.ExternalID
			out.Spec.Unschedulable = in.Unschedulable
			return s.Convert(&in.NodeResources.Capacity, &out.Status.Capacity, 0)
		},

		func(in *api.LimitRange, out *LimitRange, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *LimitRange, out *api.LimitRange, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *Namespace, out *api.Namespace, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.ObjectMeta.Labels, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.LimitRangeSpec, out *LimitRangeSpec, s conversion.Scope) error {
			*out = LimitRangeSpec{}
			out.Limits = make([]LimitRangeItem, len(in.Limits), len(in.Limits))
			for i := range in.Limits {
				if err := s.Convert(&in.Limits[i], &out.Limits[i], 0); err != nil {
					return err
				}
			}
			return nil
		},
		func(in *LimitRangeSpec, out *api.LimitRangeSpec, s conversion.Scope) error {
			*out = api.LimitRangeSpec{}
			out.Limits = make([]api.LimitRangeItem, len(in.Limits), len(in.Limits))
			for i := range in.Limits {
				if err := s.Convert(&in.Limits[i], &out.Limits[i], 0); err != nil {
					return err
				}
			}
			return nil
		},

		func(in *api.LimitRangeItem, out *LimitRangeItem, s conversion.Scope) error {
			*out = LimitRangeItem{}
			out.Type = LimitType(in.Type)
			if err := s.Convert(&in.Max, &out.Max, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Min, &out.Min, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Default, &out.Default, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *LimitRangeItem, out *api.LimitRangeItem, s conversion.Scope) error {
			*out = api.LimitRangeItem{}
			out.Type = api.LimitType(in.Type)
			if err := s.Convert(&in.Max, &out.Max, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Min, &out.Min, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Default, &out.Default, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.ResourceQuota, out *ResourceQuota, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.Labels, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ResourceQuota, out *api.ResourceQuota, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Spec, &out.Spec, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Labels, &out.ObjectMeta.Labels, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.ResourceQuotaSpec, out *ResourceQuotaSpec, s conversion.Scope) error {
			*out = ResourceQuotaSpec{}
			if err := s.Convert(&in.Hard, &out.Hard, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ResourceQuotaSpec, out *api.ResourceQuotaSpec, s conversion.Scope) error {
			*out = api.ResourceQuotaSpec{}
			if err := s.Convert(&in.Hard, &out.Hard, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.ResourceQuotaStatus, out *ResourceQuotaStatus, s conversion.Scope) error {
			*out = ResourceQuotaStatus{}
			if err := s.Convert(&in.Hard, &out.Hard, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Used, &out.Used, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *ResourceQuotaStatus, out *api.ResourceQuotaStatus, s conversion.Scope) error {
			*out = api.ResourceQuotaStatus{}
			if err := s.Convert(&in.Hard, &out.Hard, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Used, &out.Used, 0); err != nil {
				return err
			}
			return nil
		},

		// Object ID <-> Name
		// TODO: amend the conversion package to allow overriding specific fields.
		func(in *ObjectReference, out *api.ObjectReference, s conversion.Scope) error {
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.Name = in.ID
			out.UID = in.UID
			out.APIVersion = in.APIVersion
			out.ResourceVersion = in.ResourceVersion
			out.FieldPath = in.FieldPath
			return nil
		},
		func(in *api.ObjectReference, out *ObjectReference, s conversion.Scope) error {
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.ID = in.Name
			out.UID = in.UID
			out.APIVersion = in.APIVersion
			out.ResourceVersion = in.ResourceVersion
			out.FieldPath = in.FieldPath
			return nil
		},

		// Event Status <-> Condition
		// Event Source <-> Source.Component
		// Event Host <-> Source.Host
		// TODO: remove this when it becomes possible to specify a field name conversion on a specific type
		func(in *api.Event, out *Event, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			out.Reason = in.Reason
			out.Message = in.Message
			out.Source = in.Source.Component
			out.Host = in.Source.Host
			out.Timestamp = in.FirstTimestamp
			out.FirstTimestamp = in.FirstTimestamp
			out.LastTimestamp = in.LastTimestamp
			out.Count = in.Count
			return s.Convert(&in.InvolvedObject, &out.InvolvedObject, 0)
		},
		func(in *Event, out *api.Event, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			out.Reason = in.Reason
			out.Message = in.Message
			out.Source.Component = in.Source
			out.Source.Host = in.Host
			if in.FirstTimestamp.IsZero() {
				// Assume this is an old event that does not specify FirstTimestamp/LastTimestamp/Count
				out.FirstTimestamp = in.Timestamp
				out.LastTimestamp = in.Timestamp
				out.Count = 1
			} else {
				out.FirstTimestamp = in.FirstTimestamp
				out.LastTimestamp = in.LastTimestamp
				out.Count = in.Count
			}
			return s.Convert(&in.InvolvedObject, &out.InvolvedObject, 0)
		},

		// This is triggered for the Memory field of Container.
		func(in *int64, out *resource.Quantity, s conversion.Scope) error {
			out.Set(*in)
			out.Format = resource.BinarySI
			return nil
		},
		func(in *resource.Quantity, out *int64, s conversion.Scope) error {
			*out = in.Value()
			return nil
		},

		// This is triggered by the CPU field of Container.
		// Note that if we add other int/Quantity conversions my
		// simple hack (int64=Value(), int=MilliValue()) here won't work.
		func(in *int, out *resource.Quantity, s conversion.Scope) error {
			out.SetMilli(int64(*in))
			out.Format = resource.DecimalSI
			return nil
		},
		func(in *resource.Quantity, out *int, s conversion.Scope) error {
			*out = int(in.MilliValue())
			return nil
		},

		// Convert resource lists.
		func(in *ResourceList, out *api.ResourceList, s conversion.Scope) error {
			*out = api.ResourceList{}
			for k, v := range *in {
				fv, err := strconv.ParseFloat(v.String(), 64)
				if err != nil {
					return &api.ConversionError{
						In: in, Out: out,
						Message: fmt.Sprintf("value '%v' of '%v': %v", v, k, err),
					}
				}
				if k == ResourceCPU {
					(*out)[api.ResourceCPU] = *resource.NewMilliQuantity(int64(fv*1000), resource.DecimalSI)
				} else {
					(*out)[api.ResourceName(k)] = *resource.NewQuantity(int64(fv), resource.BinarySI)
				}
			}
			return nil
		},
		func(in *api.ResourceList, out *ResourceList, s conversion.Scope) error {
			*out = ResourceList{}
			for k, v := range *in {
				if k == api.ResourceCPU {
					(*out)[ResourceCPU] = util.NewIntOrStringFromString(fmt.Sprintf("%v", float64(v.MilliValue())/1000))
				} else {
					(*out)[ResourceName(k)] = util.NewIntOrStringFromInt(int(v.Value()))
				}
			}
			return nil
		},

		func(in *api.Volume, out *Volume, s conversion.Scope) error {
			if err := s.Convert(&in.VolumeSource, &out.Source, 0); err != nil {
				return err
			}
			out.Name = in.Name
			return nil
		},
		func(in *Volume, out *api.Volume, s conversion.Scope) error {
			if err := s.Convert(&in.Source, &out.VolumeSource, 0); err != nil {
				return err
			}
			out.Name = in.Name
			return nil
		},

		func(in *api.VolumeSource, out *VolumeSource, s conversion.Scope) error {
			if err := s.Convert(&in.EmptyDir, &out.EmptyDir, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.GitRepo, &out.GitRepo, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ISCSI, &out.ISCSI, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.GCEPersistentDisk, &out.GCEPersistentDisk, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.AWSElasticBlockStore, &out.AWSElasticBlockStore, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.HostPath, &out.HostDir, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Secret, &out.Secret, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.NFS, &out.NFS, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Glusterfs, &out.Glusterfs, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.PersistentVolumeClaimVolumeSource, &out.PersistentVolumeClaimVolumeSource, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.RBD, &out.RBD, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *VolumeSource, out *api.VolumeSource, s conversion.Scope) error {
			if err := s.Convert(&in.EmptyDir, &out.EmptyDir, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.GitRepo, &out.GitRepo, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.GCEPersistentDisk, &out.GCEPersistentDisk, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.AWSElasticBlockStore, &out.AWSElasticBlockStore, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ISCSI, &out.ISCSI, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.HostDir, &out.HostPath, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Secret, &out.Secret, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.NFS, &out.NFS, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.PersistentVolumeClaimVolumeSource, &out.PersistentVolumeClaimVolumeSource, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Glusterfs, &out.Glusterfs, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.RBD, &out.RBD, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.PullPolicy, out *PullPolicy, s conversion.Scope) error {
			switch *in {
			case api.PullAlways:
				*out = PullAlways
			case api.PullNever:
				*out = PullNever
			case api.PullIfNotPresent:
				*out = PullIfNotPresent
			case "":
				*out = ""
			default:
				// Let unknown values through - they will get caught by validation
				*out = PullPolicy(*in)
			}
			return nil
		},
		func(in *PullPolicy, out *api.PullPolicy, s conversion.Scope) error {
			switch *in {
			case PullAlways:
				*out = api.PullAlways
			case PullNever:
				*out = api.PullNever
			case PullIfNotPresent:
				*out = api.PullIfNotPresent
			case "":
				*out = ""
			default:
				// Let unknown values through - they will get caught by validation
				*out = api.PullPolicy(*in)
			}
			return nil
		},

		func(in *api.RestartPolicy, out *RestartPolicy, s conversion.Scope) error {
			switch *in {
			case api.RestartPolicyAlways:
				*out = RestartPolicy{Always: &RestartPolicyAlways{}}
			case api.RestartPolicyNever:
				*out = RestartPolicy{Never: &RestartPolicyNever{}}
			case api.RestartPolicyOnFailure:
				*out = RestartPolicy{OnFailure: &RestartPolicyOnFailure{}}
			default:
				*out = RestartPolicy{}
			}
			return nil
		},
		func(in *RestartPolicy, out *api.RestartPolicy, s conversion.Scope) error {
			switch {
			case in.Always != nil:
				*out = api.RestartPolicyAlways
			case in.Never != nil:
				*out = api.RestartPolicyNever
			case in.OnFailure != nil:
				*out = api.RestartPolicyOnFailure
			default:
				*out = ""
			}
			return nil
		},

		func(in *api.Probe, out *LivenessProbe, s conversion.Scope) error {
			if err := s.Convert(&in.Exec, &out.Exec, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.HTTPGet, &out.HTTPGet, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TCPSocket, &out.TCPSocket, 0); err != nil {
				return err
			}
			out.InitialDelaySeconds = in.InitialDelaySeconds
			out.TimeoutSeconds = in.TimeoutSeconds
			return nil
		},
		func(in *LivenessProbe, out *api.Probe, s conversion.Scope) error {
			if err := s.Convert(&in.Exec, &out.Exec, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.HTTPGet, &out.HTTPGet, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TCPSocket, &out.TCPSocket, 0); err != nil {
				return err
			}
			out.InitialDelaySeconds = in.InitialDelaySeconds
			out.TimeoutSeconds = in.TimeoutSeconds
			return nil
		},

		func(in *api.Endpoints, out *Endpoints, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.ObjectMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Subsets, &out.Subsets, 0); err != nil {
				return err
			}
			// Produce back-compat fields.
			firstPortName := ""
			if len(in.Subsets) > 0 {
				if len(in.Subsets[0].Ports) > 0 {
					if err := s.Convert(&in.Subsets[0].Ports[0].Protocol, &out.Protocol, 0); err != nil {
						return err
					}
					firstPortName = in.Subsets[0].Ports[0].Name
				}
			} else {
				out.Protocol = ProtocolTCP
			}
			for i := range in.Subsets {
				ss := &in.Subsets[i]
				for j := range ss.Ports {
					ssp := &ss.Ports[j]
					if ssp.Name != firstPortName {
						continue
					}
					for k := range ss.Addresses {
						ssa := &ss.Addresses[k]
						hostPort := net.JoinHostPort(ssa.IP, strconv.Itoa(ssp.Port))
						out.Endpoints = append(out.Endpoints, hostPort)
						if ssa.TargetRef != nil {
							target := EndpointObjectReference{
								Endpoint: hostPort,
							}
							if err := s.Convert(ssa.TargetRef, &target.ObjectReference, 0); err != nil {
								return err
							}
							out.TargetRefs = append(out.TargetRefs, target)
						}
					}
				}
			}
			return nil
		},
		func(in *Endpoints, out *api.Endpoints, s conversion.Scope) error {
			if err := s.Convert(&in.TypeMeta, &out.TypeMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.TypeMeta, &out.ObjectMeta, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Subsets, &out.Subsets, 0); err != nil {
				return err
			}
			// Back-compat fields are handled in the defaulting phase.
			return nil
		},

		func(in *api.NodeCondition, out *NodeCondition, s conversion.Scope) error {
			if err := s.Convert(&in.Type, &out.Kind, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.LastHeartbeatTime, &out.LastProbeTime, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.LastTransitionTime, &out.LastTransitionTime, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Reason, &out.Reason, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Message, &out.Message, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *NodeCondition, out *api.NodeCondition, s conversion.Scope) error {
			if err := s.Convert(&in.Kind, &out.Type, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.LastProbeTime, &out.LastHeartbeatTime, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.LastTransitionTime, &out.LastTransitionTime, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Reason, &out.Reason, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Message, &out.Message, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.NodeConditionType, out *NodeConditionKind, s conversion.Scope) error {
			switch *in {
			case api.NodeReady:
				*out = NodeReady
				break
			case "":
				*out = ""
			default:
				*out = NodeConditionKind(*in)
				break
			}
			return nil
		},
		func(in *NodeConditionKind, out *api.NodeConditionType, s conversion.Scope) error {
			switch *in {
			case NodeReady:
				*out = api.NodeReady
				break
			case "":
				*out = ""
			default:
				*out = api.NodeConditionType(*in)
				break
			}
			return nil
		},

		func(in *api.ConditionStatus, out *ConditionStatus, s conversion.Scope) error {
			switch *in {
			case api.ConditionTrue:
				*out = ConditionFull
				break
			case api.ConditionFalse:
				*out = ConditionNone
				break
			default:
				*out = ConditionStatus(*in)
				break
			}
			return nil
		},
		func(in *ConditionStatus, out *api.ConditionStatus, s conversion.Scope) error {
			switch *in {
			case ConditionFull:
				*out = api.ConditionTrue
				break
			case ConditionNone:
				*out = api.ConditionFalse
				break
			default:
				*out = api.ConditionStatus(*in)
				break
			}
			return nil
		},

		func(in *api.PodCondition, out *PodCondition, s conversion.Scope) error {
			if err := s.Convert(&in.Type, &out.Kind, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
				return err
			}
			return nil
		},
		func(in *PodCondition, out *api.PodCondition, s conversion.Scope) error {
			if err := s.Convert(&in.Kind, &out.Type, 0); err != nil {
				return err
			}
			if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
				return err
			}
			return nil
		},

		func(in *api.PodConditionType, out *PodConditionKind, s conversion.Scope) error {
			switch *in {
			case api.PodReady:
				*out = PodReady
				break
			case "":
				*out = ""
			default:
				*out = PodConditionKind(*in)
				break
			}
			return nil
		},
		func(in *PodConditionKind, out *api.PodConditionType, s conversion.Scope) error {
			switch *in {
			case PodReady:
				*out = api.PodReady
				break
			case "":
				*out = ""
			default:
				*out = api.PodConditionType(*in)
				break
			}
			return nil
		},

		func(in *Binding, out *api.Binding, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			out.Target = api.ObjectReference{
				Name: in.Host,
			}
			out.Name = in.PodID
			return nil
		},
		func(in *api.Binding, out *Binding, s conversion.Scope) error {
			if err := s.DefaultConvert(in, out, conversion.IgnoreMissingFields); err != nil {
				return err
			}
			out.Host = in.Target.Name
			out.PodID = in.Name
			return nil
		},
		func(in *api.SecretVolumeSource, out *SecretVolumeSource, s conversion.Scope) error {
			out.Target.ID = in.SecretName
			return nil
		},
		func(in *SecretVolumeSource, out *api.SecretVolumeSource, s conversion.Scope) error {
			out.SecretName = in.Target.ID
			return nil
		},
	)
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}

	// Add field conversion funcs.
	err = api.Scheme.AddFieldLabelConversionFunc("v1beta2", "Pod",
		func(label, value string) (string, string, error) {
			switch label {
			case "name":
				return "metadata.name", value, nil
			case "DesiredState.Host":
				return "spec.host", value, nil
			case "DesiredState.Status":
				podStatus := PodStatus(value)
				var internalValue api.PodPhase
				api.Scheme.Convert(&podStatus, &internalValue)
				return "status.phase", string(internalValue), nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = api.Scheme.AddFieldLabelConversionFunc("v1beta2", "Node",
		func(label, value string) (string, string, error) {
			switch label {
			case "name":
				return "metadata.name", value, nil
			case "unschedulable":
				return "spec.unschedulable", value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// if one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = api.Scheme.AddFieldLabelConversionFunc("v1beta2", "ReplicationController",
		func(label, value string) (string, string, error) {
			switch label {
			case "name":
				return "metadata.name", value, nil
			case "currentState.replicas":
				return "status.replicas", value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = api.Scheme.AddFieldLabelConversionFunc("v1beta2", "Event",
		func(label, value string) (string, string, error) {
			switch label {
			case "involvedObject.kind",
				"involvedObject.namespace",
				"involvedObject.uid",
				"involvedObject.apiVersion",
				"involvedObject.resourceVersion",
				"involvedObject.fieldPath",
				"reason",
				"source":
				return label, value, nil
			case "involvedObject.id":
				return "involvedObject.name", value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
	err = api.Scheme.AddFieldLabelConversionFunc("v1beta2", "Namespace",
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
	err = api.Scheme.AddFieldLabelConversionFunc("v1beta2", "Secret",
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
	err = api.Scheme.AddFieldLabelConversionFunc("v1beta2", "ServiceAccount",
		func(label, value string) (string, string, error) {
			switch label {
			case "name":
				return "metadata.name", value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}
