/*
Copyright 2018 The Kubernetes Authors.

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

package topology

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodLocator represents information about where a pod exists in arbitrary space.  This is useful for things like
// being able to reverse-map pod IPs to topology labels, without needing to watch all Pods or all Nodes.
type PodLocator struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta
	// IP addresses of the running Pod.
	// May not be loopback (127.0.0.0/8), link-local (169.254.0.0/16),
	// or link-local multicast ((224.0.0.0/24).
	// Both IPv4 and IPv6 are also accepted.
	// +optional
	Addresses []string
	// Name of Node where the Pod is running on.
	// +optional
	NodeName string
	// Labels of Node where the Pod is running on, being added for identifying Pod's topological information.
	// +optional
	NodeLabels map[string]string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodLocatorList is a list of PodLocator objects.
type PodLocatorList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is a list of schema objects.
	Items []PodLocator
}
