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
	"fmt"
	"sort"
	"strings"

	"sigs.k8s.io/kustomize/pkg/gvk"
)

const defaultFieldPath = "metadata.name"

// Var represents a variable whose value will be sourced
// from a field in a Kubernetes object.
type Var struct {
	// Value of identifier name e.g. FOO used in container args, annotations
	// Appears in pod template as $(FOO)
	Name string `json:"name" yaml:"name"`

	// ObjRef must refer to a Kubernetes resource under the
	// purview of this kustomization. ObjRef should use the
	// raw name of the object (the name specified in its YAML,
	// before addition of a namePrefix and a nameSuffix).
	ObjRef Target `json:"objref" yaml:"objref"`

	// FieldRef refers to the field of the object referred to by
	// ObjRef whose value will be extracted for use in
	// replacing $(FOO).
	// If unspecified, this defaults to fieldPath: $defaultFieldPath
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

// FieldSelector contains the fieldPath to an object field.
// This struct is added to keep the backward compatibility of using ObjectFieldSelector
// for Var.FieldRef
type FieldSelector struct {
	FieldPath string `json:"fieldPath,omitempty" yaml:"fieldPath,omitempty"`
}

// defaulting sets reference to field used by default.
func (v *Var) defaulting() {
	if v.FieldRef.FieldPath == "" {
		v.FieldRef.FieldPath = defaultFieldPath
	}
}

// VarSet is a slice of Vars where no var.Name is repeated.
type VarSet struct {
	set []Var
}

// Set returns a copy of the var set.
func (vs *VarSet) Set() []Var {
	s := make([]Var, len(vs.set))
	copy(s, vs.set)
	return s
}

// MergeSet absorbs other vars with error on name collision.
func (vs *VarSet) MergeSet(incoming *VarSet) error {
	return vs.MergeSlice(incoming.set)
}

// MergeSlice absorbs other vars with error on name collision.
// Empty fields in incoming vars are defaulted.
func (vs *VarSet) MergeSlice(incoming []Var) error {
	for _, v := range incoming {
		if vs.Contains(v) {
			return fmt.Errorf(
				"var %s already encountered", v.Name)
		}
		v.defaulting()
		vs.insert(v)
	}
	return nil
}

func (vs *VarSet) insert(v Var) {
	index := sort.Search(
		len(vs.set),
		func(i int) bool { return vs.set[i].Name > v.Name })
	// make room
	vs.set = append(vs.set, Var{})
	// shift right at index.
	// copy will not increase size of destination.
	copy(vs.set[index+1:], vs.set[index:])
	vs.set[index] = v
}

// Contains is true if the set has the other var.
func (vs *VarSet) Contains(other Var) bool {
	return vs.Get(other.Name) != nil
}

// Get returns the var with the given name, else nil.
func (vs *VarSet) Get(name string) *Var {
	for _, v := range vs.set {
		if v.Name == name {
			return &v
		}
	}
	return nil
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
