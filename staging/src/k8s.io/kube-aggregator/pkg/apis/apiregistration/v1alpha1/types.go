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

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// APIServiceList is a list of APIService objects.
type APIServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Items []APIService `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// ServiceReference holds a reference to Service.legacy.k8s.io
type ServiceReference struct {
	// Namespace is the namespace of the service
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,1,opt,name=namespace"`
	// Name is the name of the service
	Name string `json:"name,omitempty" protobuf:"bytes,2,opt,name=name"`
}

// APIServiceSpec contains information for locating and communicating with a server.
// Only https is supported, though you are able to disable certificate verification.
type APIServiceSpec struct {
	// Service is a reference to the service for this API server.  It must communicate
	// on port 443
	Service ServiceReference `json:"service" protobuf:"bytes,1,opt,name=service"`
	// Group is the API group name this server hosts
	Group string `json:"group,omitempty" protobuf:"bytes,2,opt,name=group"`
	// Version is the API version this server hosts.  For example, "v1"
	Version string `json:"version,omitempty" protobuf:"bytes,3,opt,name=version"`

	// InsecureSkipTLSVerify disables TLS certificate verification when communicating with this server.
	// This is strongly discouraged.  You should use the CABundle instead.
	InsecureSkipTLSVerify bool `json:"insecureSkipTLSVerify,omitempty" protobuf:"varint,4,opt,name=insecureSkipTLSVerify"`
	// CABundle is a PEM encoded CA bundle which will be used to validate an API server's serving certificate.
	CABundle []byte `json:"caBundle" protobuf:"bytes,5,opt,name=caBundle"`

	// Priority controls the ordering of this API group in the overall discovery document that gets served.
	// Client tools like `kubectl` use this ordering to derive preference, so this ordering mechanism is important.
	// Values must be between 1 and 1000
	// The primary sort is based on priority, ordered lowest number to highest (10 before 20).
	// The secondary sort is based on the alphabetical comparison of the name of the object.  (v1.bar before v1.foo)
	// We'd recommend something like: *.k8s.io (except extensions) at 100, extensions at 150
	// PaaSes (OpenShift, Deis) are recommended to be in the 200s
	Priority int64 `json:"priority" protobuf:"varint,6,opt,name=priority"`
}

// APIServiceStatus contains derived information about an API server
type APIServiceStatus struct {
}

// +genclient=true
// +nonNamespaced=true

// APIService represents a server for a particular GroupVersion.
// Name must be "version.group".
type APIService struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec contains information for locating and communicating with a server
	Spec APIServiceSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// Status contains derived information about an API server
	Status APIServiceStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}
