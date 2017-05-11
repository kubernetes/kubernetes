/*
Copyright 2017 The Kubernetes Authors.

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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient=true
// +nonNamespaced=true

// Group describes an API group with all its versions and resources.
type Group struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Status of the group.
	Status GroupStatus `json:"spec,omitempty" protobuf:"bytes,2,opt,name=status"`
}

// GroupStatus describes the API group with all its versions and resources.
type GroupStatus struct {
	// versions are the versions supported in this group, order by preference. The first one is the preferred
	// version for the group.
	Versions []GroupVersion `json:"versions" protobuf:"bytes,1,rep,name=versions"`
	// a map of client CIDR to server address that is serving this group.
	// This is to help clients reach servers in the most network-efficient way possible.
	// Clients can use the appropriate server address as per the CIDR that they match.
	// In case of multiple matches, clients should use the longest matching CIDR.
	// The server returns only those CIDRs that it thinks that the client can match.
	// For example: the master will return an internal IP CIDR only, if the client reaches the server using an internal IP.
	// Server looks at X-Forwarded-For header or X-Real-Ip header or request.RemoteAddr (in that order) to get the client IP.
	ServerAddressByClientCIDRs []metav1.ServerAddressByClientCIDR `json:"serverAddressByClientCIDRs" protobuf:"bytes,2,rep,name=serverAddressByClientCIDRs"`
}

// GroupVersion describes one version of an API group, including all resources available in this version.
type GroupVersion struct {
	// the version name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// resources contains the name of the resources and if they are namespaced.
	Resources []metav1.APIResource `json:"resources" protobuf:"bytes,2,rep,name=resources"`
}

// GroupList is a list of API groups.
type GroupList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// List of group.
	Items []Group `json:"items" protobuf:"bytes,2,rep,name=items"`
}
