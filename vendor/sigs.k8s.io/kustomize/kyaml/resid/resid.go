// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resid

import (
	"reflect"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// ResId is an identifier of a k8s resource object.
type ResId struct {
	// Gvk of the resource.
	Gvk `json:",inline,omitempty" yaml:",inline,omitempty"`

	// Name of the resource.
	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	// Namespace the resource belongs to, if it can belong to a namespace.
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
}

// NewResIdWithNamespace creates new ResId
// in a given namespace.
func NewResIdWithNamespace(k Gvk, n, ns string) ResId {
	return ResId{Gvk: k, Name: n, Namespace: ns}
}

// NewResId creates new ResId.
func NewResId(k Gvk, n string) ResId {
	return NewResIdWithNamespace(k, n, "")
}

// NewResIdKindOnly creates a new ResId.
func NewResIdKindOnly(k string, n string) ResId {
	return NewResId(FromKind(k), n)
}

const (
	noNamespace          = "[noNs]"
	noName               = "[noName]"
	separator            = "/"
	TotallyNotANamespace = "_non_namespaceable_"
	DefaultNamespace     = "default"
)

// String of ResId based on GVK, name and prefix
func (id ResId) String() string {
	ns := id.Namespace
	if ns == "" {
		ns = noNamespace
	}
	nm := id.Name
	if nm == "" {
		nm = noName
	}
	return strings.Join(
		[]string{id.Gvk.String(), strings.Join([]string{nm, ns}, fieldSep)}, separator)
}

func FromString(s string) ResId {
	values := strings.Split(s, separator)
	gvk := GvkFromString(values[0])

	values = strings.Split(values[1], fieldSep)
	last := len(values) - 1

	ns := values[last]
	if ns == noNamespace {
		ns = ""
	}
	nm := strings.Join(values[:last], fieldSep)
	if nm == noName {
		nm = ""
	}
	return ResId{
		Gvk:       gvk,
		Namespace: ns,
		Name:      nm,
	}
}

// FromRNode returns the ResId for the RNode
func FromRNode(rn *yaml.RNode) ResId {
	group, version := ParseGroupVersion(rn.GetApiVersion())
	return NewResIdWithNamespace(
		Gvk{Group: group, Version: version, Kind: rn.GetKind()}, rn.GetName(), rn.GetNamespace())
}

// GvknEquals returns true if the other id matches
// Group/Version/Kind/name.
func (id ResId) GvknEquals(o ResId) bool {
	return id.Name == o.Name && id.Gvk.Equals(o.Gvk)
}

// IsSelectedBy returns true if self is selected by the argument.
func (id ResId) IsSelectedBy(selector ResId) bool {
	return (selector.Name == "" || selector.Name == id.Name) &&
		(selector.Namespace == "" || selector.IsNsEquals(id)) &&
		id.Gvk.IsSelected(&selector.Gvk)
}

// Equals returns true if the other id matches
// namespace/Group/Version/Kind/name.
func (id ResId) Equals(o ResId) bool {
	return id.IsNsEquals(o) && id.GvknEquals(o)
}

// IsNsEquals returns true if the id is in
// the same effective namespace.
func (id ResId) IsNsEquals(o ResId) bool {
	return id.EffectiveNamespace() == o.EffectiveNamespace()
}

// IsInDefaultNs returns true if id is a namespaceable
// ResId and the Namespace is either not set or set
// to DefaultNamespace.
func (id ResId) IsInDefaultNs() bool {
	return !id.IsClusterScoped() && id.isPutativelyDefaultNs()
}

func (id ResId) isPutativelyDefaultNs() bool {
	return id.Namespace == "" || id.Namespace == DefaultNamespace
}

// EffectiveNamespace returns a non-ambiguous, non-empty
// namespace for use in reporting and equality tests.
func (id ResId) EffectiveNamespace() string {
	// The order of these checks matters.
	if id.IsClusterScoped() {
		return TotallyNotANamespace
	}
	if id.isPutativelyDefaultNs() {
		return DefaultNamespace
	}
	return id.Namespace
}

// IsEmpty returns true of all of the id's fields are
// empty strings
func (id ResId) IsEmpty() bool {
	return reflect.DeepEqual(id, ResId{})
}
