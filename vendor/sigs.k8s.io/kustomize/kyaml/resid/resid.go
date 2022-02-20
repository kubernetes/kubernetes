// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resid

import (
	"reflect"
	"strings"
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
	noNamespace          = "~X"
	noName               = "~N"
	separator            = "|"
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
		[]string{id.Gvk.String(), ns, nm}, separator)
}

func FromString(s string) ResId {
	values := strings.Split(s, separator)
	g := GvkFromString(values[0])

	ns := values[1]
	if ns == noNamespace {
		ns = ""
	}
	nm := values[2]
	if nm == noName {
		nm = ""
	}
	return ResId{
		Gvk:       g,
		Namespace: ns,
		Name:      nm,
	}
}

// GvknString of ResId based on GVK and name
func (id ResId) GvknString() string {
	return id.Gvk.String() + separator + id.Name
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
