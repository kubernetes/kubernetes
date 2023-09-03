// Copyright 2022 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

type ReplacementField struct {
	Replacement `json:",inline,omitempty" yaml:",inline,omitempty"`
	Path        string `json:"path,omitempty" yaml:"path,omitempty"`
}
