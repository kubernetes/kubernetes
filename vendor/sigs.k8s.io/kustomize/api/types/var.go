// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"sigs.k8s.io/kustomize/api/resid"
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
	resid.Gvk  `json:",inline,omitempty" yaml:",inline,omitempty"`
	Name       string `json:"name" yaml:"name"`
	Namespace  string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
}

// GVK returns the Gvk object in Target
func (t *Target) GVK() resid.Gvk {
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

// FieldSelector contains the fieldPath to an object field.
// This struct is added to keep the backward compatibility of using ObjectFieldSelector
// for Var.FieldRef
type FieldSelector struct {
	FieldPath string `json:"fieldPath,omitempty" yaml:"fieldPath,omitempty"`
}

// defaulting sets reference to field used by default.
func (v *Var) Defaulting() {
	if v.FieldRef.FieldPath == "" {
		v.FieldRef.FieldPath = DefaultReplacementFieldPath
	}
	v.ObjRef.GVK()
}

// DeepEqual returns true if var a and b are Equals.
// Note 1: The objects are unchanged by the VarEqual
// Note 2: Should be normalize be FieldPath before doing
// the DeepEqual. spec.a[b] is supposed to be the same
// as spec.a.b
func (v Var) DeepEqual(other Var) bool {
	v.Defaulting()
	other.Defaulting()
	return reflect.DeepEqual(v, other)
}

// VarSet is a set of Vars where no var.Name is repeated.
type VarSet struct {
	set map[string]Var
}

// NewVarSet returns an initialized VarSet
func NewVarSet() VarSet {
	return VarSet{set: map[string]Var{}}
}

// AsSlice returns the vars as a slice.
func (vs *VarSet) AsSlice() []Var {
	s := make([]Var, len(vs.set))
	i := 0
	for _, v := range vs.set {
		s[i] = v
		i++
	}
	sort.Sort(byName(s))
	return s
}

// Copy returns a copy of the var set.
func (vs *VarSet) Copy() VarSet {
	newSet := make(map[string]Var, len(vs.set))
	for k, v := range vs.set {
		newSet[k] = v
	}
	return VarSet{set: newSet}
}

// MergeSet absorbs other vars with error on name collision.
func (vs *VarSet) MergeSet(incoming VarSet) error {
	for _, incomingVar := range incoming.set {
		if err := vs.Merge(incomingVar); err != nil {
			return err
		}
	}
	return nil
}

// MergeSlice absorbs a Var slice with error on name collision.
// Empty fields in incoming vars are defaulted.
func (vs *VarSet) MergeSlice(incoming []Var) error {
	for _, v := range incoming {
		if err := vs.Merge(v); err != nil {
			return err
		}
	}
	return nil
}

// Merge absorbs another Var with error on name collision.
// Empty fields in incoming Var is defaulted.
func (vs *VarSet) Merge(v Var) error {
	if vs.Contains(v) {
		return fmt.Errorf(
			"var '%s' already encountered", v.Name)
	}
	v.Defaulting()
	vs.set[v.Name] = v
	return nil
}

// AbsorbSet absorbs other vars with error on (name,value) collision.
func (vs *VarSet) AbsorbSet(incoming VarSet) error {
	for _, v := range incoming.set {
		if err := vs.Absorb(v); err != nil {
			return err
		}
	}
	return nil
}

// AbsorbSlice absorbs a Var slice with error on (name,value) collision.
// Empty fields in incoming vars are defaulted.
func (vs *VarSet) AbsorbSlice(incoming []Var) error {
	for _, v := range incoming {
		if err := vs.Absorb(v); err != nil {
			return err
		}
	}
	return nil
}

// Absorb absorbs another Var with error on (name,value) collision.
// Empty fields in incoming Var is defaulted.
func (vs *VarSet) Absorb(v Var) error {
	conflicting := vs.Get(v.Name)
	if conflicting == nil {
		// no conflict. The var is valid.
		v.Defaulting()
		vs.set[v.Name] = v
		return nil
	}

	if !reflect.DeepEqual(v, *conflicting) {
		// two vars with the same name are pointing at two
		// different resources.
		return fmt.Errorf(
			"var '%s' already encountered", v.Name)
	}
	return nil
}

// Contains is true if the set has the other var.
func (vs *VarSet) Contains(other Var) bool {
	return vs.Get(other.Name) != nil
}

// Get returns the var with the given name, else nil.
func (vs *VarSet) Get(name string) *Var {
	if v, found := vs.set[name]; found {
		return &v
	}
	return nil
}

// byName is a sort interface which sorts Vars by name alphabetically
type byName []Var

func (v byName) Len() int           { return len(v) }
func (v byName) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v byName) Less(i, j int) bool { return v[i].Name < v[j].Name }
