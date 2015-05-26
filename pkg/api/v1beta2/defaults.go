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
	"net"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

func addDefaultingFuncs() {
	api.Scheme.AddDefaultingFuncs(
		func(obj *ReplicationController) {
			if len(obj.DesiredState.ReplicaSelector) == 0 {
				obj.DesiredState.ReplicaSelector = obj.DesiredState.PodTemplate.Labels
			}
			if len(obj.Labels) == 0 {
				obj.Labels = obj.DesiredState.PodTemplate.Labels
			}
		},
		func(obj *Volume) {
			if util.AllPtrFieldsNil(&obj.Source) {
				glog.Errorf("Defaulting volume source for %v", obj)
				obj.Source = VolumeSource{
					EmptyDir: &EmptyDirVolumeSource{},
				}
			}
		},
		func(obj *ContainerPort) {
			if obj.Protocol == "" {
				obj.Protocol = ProtocolTCP
			}
		},
		func(obj *Container) {
			if obj.ImagePullPolicy == "" {
				// TODO(dchen1107): Move ParseImageName code to pkg/util
				parts := strings.Split(obj.Image, ":")
				// Check image tag
				if parts[len(parts)-1] == "latest" {
					obj.ImagePullPolicy = PullAlways
				} else {
					obj.ImagePullPolicy = PullIfNotPresent
				}
			}
			if obj.TerminationMessagePath == "" {
				obj.TerminationMessagePath = TerminationMessagePathDefault
			}
			defaultSecurityContext(obj)
		},
		func(obj *RestartPolicy) {
			if util.AllPtrFieldsNil(obj) {
				obj.Always = &RestartPolicyAlways{}
			}
		},
		func(obj *Service) {
			if obj.Protocol == "" {
				obj.Protocol = ProtocolTCP
			}
			if obj.SessionAffinity == "" {
				obj.SessionAffinity = ServiceAffinityNone
			}
			if obj.Type == "" {
				if obj.CreateExternalLoadBalancer {
					obj.Type = ServiceTypeLoadBalancer
				} else {
					obj.Type = ServiceTypeClusterIP
				}
			} else if obj.Type == ServiceTypeLoadBalancer {
				obj.CreateExternalLoadBalancer = true
			}
			for i := range obj.Ports {
				sp := &obj.Ports[i]
				if sp.Protocol == "" {
					sp.Protocol = ProtocolTCP
				}
				if sp.ContainerPort == util.NewIntOrStringFromInt(0) || sp.ContainerPort == util.NewIntOrStringFromString("") {
					sp.ContainerPort = util.NewIntOrStringFromInt(sp.Port)
				}
			}
		},
		func(obj *PodSpec) {
			if obj.DNSPolicy == "" {
				obj.DNSPolicy = DNSClusterFirst
			}
			if obj.HostNetwork {
				defaultHostNetworkPorts(&obj.Containers)
			}
		},
		func(obj *ContainerManifest) {
			if obj.DNSPolicy == "" {
				obj.DNSPolicy = DNSClusterFirst
			}
			if obj.HostNetwork {
				defaultHostNetworkPorts(&obj.Containers)
			}
		},
		func(obj *LivenessProbe) {
			if obj.TimeoutSeconds == 0 {
				obj.TimeoutSeconds = 1
			}
		},
		func(obj *Secret) {
			if obj.Type == "" {
				obj.Type = SecretTypeOpaque
			}
		},
		func(obj *PersistentVolume) {
			if obj.Status.Phase == "" {
				obj.Status.Phase = VolumePending
			}
		},
		func(obj *PersistentVolumeClaim) {
			if obj.Status.Phase == "" {
				obj.Status.Phase = ClaimPending
			}
		},
		func(obj *Endpoints) {
			if obj.Protocol == "" {
				obj.Protocol = ProtocolTCP
			}
			if len(obj.Subsets) == 0 && len(obj.Endpoints) > 0 {
				// Must be a legacy-style object - populate
				// Subsets from the older fields.  Do this the
				// simplest way, which is dumb (but valid).
				for i := range obj.Endpoints {
					host, portStr, err := net.SplitHostPort(obj.Endpoints[i])
					if err != nil {
						glog.Errorf("failed to SplitHostPort(%q)", obj.Endpoints[i])
					}
					var tgtRef *ObjectReference
					for j := range obj.TargetRefs {
						if obj.TargetRefs[j].Endpoint == obj.Endpoints[i] {
							tgtRef = &ObjectReference{}
							*tgtRef = obj.TargetRefs[j].ObjectReference
						}
					}
					port, err := strconv.Atoi(portStr)
					if err != nil {
						glog.Errorf("failed to Atoi(%q)", portStr)
					}
					obj.Subsets = append(obj.Subsets, EndpointSubset{
						Addresses: []EndpointAddress{{IP: host, TargetRef: tgtRef}},
						Ports:     []EndpointPort{{Protocol: obj.Protocol, Port: port}},
					})
				}
			}
			for i := range obj.Subsets {
				ss := &obj.Subsets[i]
				for i := range ss.Ports {
					ep := &ss.Ports[i]
					if ep.Protocol == "" {
						ep.Protocol = ProtocolTCP
					}
				}
			}
		},
		func(obj *HTTPGetAction) {
			if obj.Path == "" {
				obj.Path = "/"
			}
		},
		func(obj *NamespaceStatus) {
			if obj.Phase == "" {
				obj.Phase = NamespaceActive
			}
		},
		func(obj *Minion) {
			if obj.ExternalID == "" {
				obj.ExternalID = obj.ID
			}
		},
		func(obj *ObjectFieldSelector) {
			if obj.APIVersion == "" {
				obj.APIVersion = "v1beta2"
			}
		},
	)
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

// defaultSecurityContext performs the downward and upward merges of a pod definition
func defaultSecurityContext(container *Container) {
	if container.SecurityContext == nil {
		glog.V(4).Infof("creating security context for container %s", container.Name)
		container.SecurityContext = &SecurityContext{}
	}
	// if there are no capabilities defined on the SecurityContext then copy the container settings
	if container.SecurityContext.Capabilities == nil {
		glog.V(4).Infof("downward merge of container.Capabilities for container %s", container.Name)
		container.SecurityContext.Capabilities = &container.Capabilities
	} else {
		// if there are capabilities defined on the security context and the container setting is
		// empty then assume that it was left off the pod definition and ensure that the container
		// settings match the security context settings (checked by the convert functions).  If
		// there are settings in both then don't touch it, the converter will error if they don't
		// match
		if len(container.Capabilities.Add) == 0 {
			glog.V(4).Infof("upward merge of container.Capabilities.Add for container %s", container.Name)
			container.Capabilities.Add = container.SecurityContext.Capabilities.Add
		}
		if len(container.Capabilities.Drop) == 0 {
			glog.V(4).Infof("upward merge of container.Capabilities.Drop for container %s", container.Name)
			container.Capabilities.Drop = container.SecurityContext.Capabilities.Drop
		}
	}
	// if there are no privileged settings on the security context then copy the container settings
	if container.SecurityContext.Privileged == nil {
		glog.V(4).Infof("downward merge of container.Privileged for container %s", container.Name)
		container.SecurityContext.Privileged = &container.Privileged
	} else {
		// we don't have a good way to know if container.Privileged was set or just defaulted to false
		// so the best we can do here is check if the securityContext is set to true and the
		// container is set to false and assume that the Privileged field was left off the container
		// definition and not an intentional mismatch
		if *container.SecurityContext.Privileged && !container.Privileged {
			glog.V(4).Infof("upward merge of container.Privileged for container %s", container.Name)
			container.Privileged = *container.SecurityContext.Privileged
		}
	}
}
