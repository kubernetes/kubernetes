// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"sigs.k8s.io/kustomize/api/resid"
)

// Selector specifies a set of resources.
// Any resource that matches intersection of all conditions
// is included in this set.
type Selector struct {
	resid.Gvk `json:",inline,omitempty" yaml:",inline,omitempty"`
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	Name      string `json:"name,omitempty" yaml:"name,omitempty"`

	// AnnotationSelector is a string that follows the label selection expression
	// https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#api
	// It matches with the resource annotations.
	AnnotationSelector string `json:"annotationSelector,omitempty" yaml:"annotationSelector,omitempty"`

	// LabelSelector is a string that follows the label selection expression
	// https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#api
	// It matches with the resource labels.
	LabelSelector string `json:"labelSelector,omitempty" yaml:"labelSelector,omitempty"`
}
