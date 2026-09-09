/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// BootstrapToken describes one bootstrap token, stored as a Secret in the cluster
// +k8s:deepcopy-gen=true
type BootstrapToken struct {
	// Token is used for establishing bidirectional trust between nodes and control-planes.
	// Used for joining nodes in the cluster.
	Token *BootstrapTokenString `json:"token" datapolicy:"token"`
	// Description sets a human-friendly message why this token exists and what it's used
	// for, so other administrators can know its purpose.
	// +optional
	Description string `json:"description,omitempty"`
	// TTL defines the time to live for this token. Defaults to 24h.
	// Expires and TTL are mutually exclusive.
	// +optional
	TTL *metav1.Duration `json:"ttl,omitempty"`
	// Expires specifies the timestamp when this token expires. Defaults to being set
	// dynamically at runtime based on the TTL. Expires and TTL are mutually exclusive.
	// +optional
	Expires *metav1.Time `json:"expires,omitempty"`
	// Usages describes the ways in which this token can be used. Can by default be used
	// for establishing bidirectional trust, but that can be changed here.
	// +optional
	Usages []string `json:"usages,omitempty"`
	// Groups specifies the extra groups that this token will authenticate as when/if
	// used for authentication
	// +optional
	Groups []string `json:"groups,omitempty"`
}

// BootstrapTokenString is a token of the format abcdef.abcdef0123456789 that is used
// for both validation of the practically of the API server from a joining node's point
// of view and as an authentication method for the node in the bootstrap phase of
// "kubeadm join". This token is and should be short-lived
type BootstrapTokenString struct {
	ID     string `json:"-"`
	Secret string `json:"-" datapolicy:"token"`
}
