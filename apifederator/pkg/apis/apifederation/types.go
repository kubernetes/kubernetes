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

package apifederation

import (
	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// APIServerList is a list of APIServer objects.
type APIServerList struct {
	unversioned.TypeMeta
	unversioned.ListMeta

	Items []APIServer
}

// APIServerSpec contains information for locating and communicating with a server.
// Only https is supported, though you are able to disable host verification.
type APIServerSpec struct {
	// InternalHost is the host:port for locating the server inside the pod/service network.
	InternalHost string
	// Group is the API group name this server hosts
	Group string
	// Version is the API version this server hosts
	Version string

	// InsecureSkipTLSVerify disables TLS certificate verification when communicating with this server.
	// This is strongly discouraged.  You should use the CABundle instead.
	InsecureSkipTLSVerify bool
	// CABundle is a PEM encoded CA bundle which be used to validate an API server's serving certificate.
	CABundle []byte

	// Priority controls the ordering of this API group in the overall discovery document that gets served.
	// Client tools like `kubectl` use this ordering to derive preference, so this ordering mechanism is important.
	// Secondary ordering is performed based on name.
	Priority int64
}

// APIServerStatus contains derived information about an API server
type APIServerStatus struct {
}

// +genclient=true
// +nonNamespaced=true

// APIServer represents a server for a particular GroupVersion.
// Name must be "version.group".
type APIServer struct {
	unversioned.TypeMeta
	kapi.ObjectMeta

	// Spec contains information for locating and communicating with a server
	Spec APIServerSpec
	// Status contains derived information about an API server
	Status APIServerStatus
}
