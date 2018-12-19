/*
Copyright 2017 The Kubernetes Authors.

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

package types

import (
	"sigs.k8s.io/kustomize/pkg/gvk"
	"strings"
)

// Var represents a variable whose value will be sourced
// from a field in a Kubernetes object.
type Var struct {
	// Value of identifier name e.g. FOO used in container args, annotations
	// Appears in pod template as $(FOO)
	Name string `json:"name" yaml:"name"`

	// ObjRef must refer to a Kubernetes resource under the
	// purview of this kustomization. ObjRef should use the
	// raw name of the object (the name specified in its YAML,
	// before addition of a namePrefix).
	ObjRef Target `json:"objref" yaml:"objref"`

	// FieldRef refers to the field of the object referred to by
	// ObjRef whose value will be extracted for use in
	// replacing $(FOO).
	// If unspecified, this defaults to fieldPath: metadata.name
	FieldRef FieldSelector `json:"fieldref,omitempty" yaml:"fieldref,omitempty"`
}

// Target refers to a kubernetes object by Group, Version, Kind and Name
// gvk.Gvk contains Group, Version and Kind
// APIVersion is added to keep the backward compatibility of using ObjectReference
// for Var.ObjRef
type Target struct {
	APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	gvk.Gvk    `json:",inline,omitempty" yaml:",inline,omitempty"`
	Name       string `json:"name" yaml:"name"`
}

// FieldSelector contains the fieldPath to an object field, such as metadata.name
// This struct is added to keep the backward compatibility of using ObjectFieldSelector
// for Var.FieldRef
type FieldSelector struct {
	FieldPath string `json:"fieldPath,omitempty" yaml:"fieldPath,omitempty"`
}

// Defaulting sets reference to field used by default.
func (v *Var) Defaulting() {
	if v.FieldRef.FieldPath == "" {
		v.FieldRef.FieldPath = "metadata.name"
	}
}

// GVK returns the Gvk object in Target
func (t *Target) GVK() gvk.Gvk {
	if t.APIVersion == "" {
		return t.Gvk
	}
	versions := strings.Split(t.APIVersion, "/")
	if len(versions) == 2 {
		t.Group = versions[0]
		t.Version = versions[1]
	}
	if len(versions) == 1 {
		t.Version = versions[0]
	}
	return t.Gvk
}
