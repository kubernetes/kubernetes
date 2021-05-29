// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// Inventory records all objects touched in a build operation.
type Inventory struct {
	Type      string   `json:"type,omitempty" yaml:"type,omitempty"`
	ConfigMap NameArgs `json:"configMap,omitempty" yaml:"configMap,omitempty"`
}

// NameArgs holds both namespace and name.
type NameArgs struct {
	Name      string `json:"name,omitempty" yaml:"name,omitempty"`
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
}
