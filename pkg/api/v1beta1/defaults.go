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
	"net"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

func init() {
	api.Scheme.AddDefaultingFuncs(
		func(obj *Volume) {
			if util.AllPtrFieldsNil(&obj.Source) {
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
				obj.SessionAffinity = AffinityTypeNone
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
	)
}

// With host networking default all host ports to container ports.
func defaultHostNetworkPorts(containers *[]Container) {
	for i := range *containers {
		for j := range (*containers)[i].Ports {
			if (*containers)[i].Ports[j].HostPort == 0 {
				(*containers)[i].Ports[j].HostPort = (*containers)[i].Ports[j].ContainerPort
			}
		}
	}
}
