/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	kapi "k8s.io/kubernetes/pkg/api/v1"
)

// APIServerList is a list of APIServer objects.
type APIServerList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Items []APIServer `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// APIServerSpec contains information for locating and communicating with a server.
// Only https is supported, though you are able to disable host verification.
type APIServerSpec struct {
	// InternalHost is the host:port for locating the server inside the pod/service network.
	InternalHost string `json:"internalHost,omitempty" protobuf:"bytes,1,opt,name=internalHost"`
	// Group is the API group name this server hosts
	Group string `json:"group,omitempty" protobuf:"bytes,2,opt,name=group"`
	// Version is the API version this server hosts
	Version string `json:"version,omitempty" protobuf:"bytes,3,opt,name=version"`

	// InsecureSkipTLSVerify disables TLS certificate verification when communicating with this server.
	// This is strongly discouraged.  You should use the CABundle instead.
	InsecureSkipTLSVerify bool `json:"insecureSkipTLSVerify,omitempty" protobuf:"varint,4,opt,name=insecureSkipTLSVerify"`
	// CABundle is a PEM encoded CA bundle which be used to validate an API server's serving certificate.
	CABundle []byte `json:"caBundle" protobuf:"bytes,5,opt,name=caBundle"`

	// Priority controls the ordering of this API group in the overall discovery document that gets served.
	// Client tools like `kubectl` use this ordering to derive preference, so this ordering mechanism is important.
	// Secondary ordering is performed based on name.
	Priority int64 `json:"priority" protobuf:"varint,6,opt,name=priority"`
}

// APIServerStatus contains derived information about an API server
type APIServerStatus struct {
}

// +genclient=true
// +nonNamespaced=true

// APIServer represents a server for a particular GroupVersion.
// Name must be "version.group".
type APIServer struct {
	unversioned.TypeMeta `json:",inline"`
	kapi.ObjectMeta      `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec contains information for locating and communicating with a server
	Spec APIServerSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// Status contains derived information about an API server
	Status APIServerStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}
