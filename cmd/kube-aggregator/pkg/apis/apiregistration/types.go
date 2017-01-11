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

package apiregistration

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kapi "k8s.io/kubernetes/pkg/api"
)

// APIServiceList is a list of APIService objects.
type APIServiceList struct {
	metav1.TypeMeta
	metav1.ListMeta

	Items []APIService
}

// ServiceReference holds a reference to Service.legacy.k8s.io
type ServiceReference struct {
	// Namespace is the namespace of the service
	Namespace string
	// Name is the name of the service
	Name string
}

// APIServiceSpec contains information for locating and communicating with a server.
// Only https is supported, though you are able to disable certificate verification.
type APIServiceSpec struct {
	// Service is a reference to the service for this API server.  It must communicate
	// on port 443
	Service ServiceReference
	// Group is the API group name this server hosts
	Group string
	// Version is the API version this server hosts.  For example, "v1"
	Version string

	// InsecureSkipTLSVerify disables TLS certificate verification when communicating with this server.
	// This is strongly discouraged.  You should use the CABundle instead.
	InsecureSkipTLSVerify bool
	// CABundle is a PEM encoded CA bundle which will be used to validate an API server's serving certificate.
	CABundle []byte

	// Priority controls the ordering of this API group in the overall discovery document that gets served.
	// Client tools like `kubectl` use this ordering to derive preference, so this ordering mechanism is important.
	// Values must be between 1 and 1000
	// The primary sort is based on priority, ordered lowest number to highest (10 before 20).
	// The secondary sort is based on the alphabetical comparison of the name of the object.  (v1.bar before v1.foo)
	// We'd recommend something like: *.k8s.io (except extensions) at 100, extensions at 150
	// PaaSes (OpenShift, Deis) are recommended to be in the 200s
	Priority int64
}

// APIServiceStatus contains derived information about an API server
type APIServiceStatus struct {
}

// +genclient=true
// +nonNamespaced=true

// APIService represents a server for a particular GroupVersion.
// Name must be "version.group".
type APIService struct {
	metav1.TypeMeta
	kapi.ObjectMeta

	// Spec contains information for locating and communicating with a server
	Spec APIServiceSpec
	// Status contains derived information about an API server
	Status APIServiceStatus
}
