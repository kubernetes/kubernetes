// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// ObjectMeta partially copies apimachinery/pkg/apis/meta/v1.ObjectMeta
// No need for a direct dependence; the fields are stable.
type ObjectMeta struct {
	Name      string `json:"name,omitempty" yaml:"name,omitempty"`
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
}
