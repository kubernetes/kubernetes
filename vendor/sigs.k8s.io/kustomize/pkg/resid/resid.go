/*
Copyright 2018 The Kubernetes Authors.

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

package resid

import (
	"strings"

	"sigs.k8s.io/kustomize/pkg/gvk"
)

// ResId conflates GroupVersionKind with a textual name to uniquely identify a kubernetes resource (object).
type ResId struct {
	// Gvk of the resource.
	gvKind gvk.Gvk
	// original name of the resource before transformation.
	name string
	// namePrefix of the resource
	// an untransformed resource has no prefix, fully transformed resource has an arbitrary number of prefixes
	// concatenated together.
	prefix string
	// namespace the resource belongs to
	// an untransformed resource has no namespace, fully transformed resource has the namespace from
	// the top most overlay
	namespace string
}

// NewResIdWithPrefixNamespace creates new resource identifier with a prefix and a namespace
func NewResIdWithPrefixNamespace(k gvk.Gvk, n, p, ns string) ResId {
	return ResId{gvKind: k, name: n, prefix: p, namespace: ns}
}

// NewResIdWithPrefix creates new resource identifier with a prefix
func NewResIdWithPrefix(k gvk.Gvk, n, p string) ResId {
	return ResId{gvKind: k, name: n, prefix: p}
}

// NewResId creates new resource identifier
func NewResId(k gvk.Gvk, n string) ResId {
	return ResId{gvKind: k, name: n}
}

// NewResIdKindOnly creates new resource identifier
func NewResIdKindOnly(k string, n string) ResId {
	return ResId{gvKind: gvk.FromKind(k), name: n}
}

const (
	noNamespace = "noNamespace"
	noPrefix    = "noPrefix"
	noName      = "noName"
	separator   = "|"
)

// String of ResId based on GVK, name and prefix
func (n ResId) String() string {
	ns := n.namespace
	if ns == "" {
		ns = noNamespace
	}
	p := n.prefix
	if p == "" {
		p = noPrefix
	}
	nm := n.name
	if nm == "" {
		nm = noName
	}

	return strings.Join(
		[]string{n.gvKind.String(), ns, p, nm}, separator)
}

// GvknString of ResId based on GVK and name
func (n ResId) GvknString() string {
	return n.gvKind.String() + separator + n.name
}

// GvknEquals return if two ResId have the same Group/Version/Kind and name
// The comparison excludes prefix
func (n ResId) GvknEquals(id ResId) bool {
	return n.gvKind.Equals(id.gvKind) && n.name == id.name
}

// Gvk returns Group/Version/Kind of the resource.
func (n ResId) Gvk() gvk.Gvk {
	return n.gvKind
}

// Name returns resource name.
func (n ResId) Name() string {
	return n.name
}

// Prefix returns name prefix.
func (n ResId) Prefix() string {
	return n.prefix
}

// Namespace returns resource namespace.
func (n ResId) Namespace() string {
	return n.namespace
}

// CopyWithNewPrefix make a new copy from current ResId and append a new prefix
func (n ResId) CopyWithNewPrefix(p string) ResId {
	return ResId{gvKind: n.gvKind, name: n.name, prefix: n.concatPrefix(p), namespace: n.namespace}
}

// CopyWithNewNamespace make a new copy from current ResId and set a new namespace
func (n ResId) CopyWithNewNamespace(ns string) ResId {
	return ResId{gvKind: n.gvKind, name: n.name, prefix: n.prefix, namespace: ns}
}

// HasSameLeftmostPrefix check if two ResIds have the same
// left most prefix.
func (n ResId) HasSameLeftmostPrefix(id ResId) bool {
	prefixes1 := n.prefixList()
	prefixes2 := id.prefixList()
	return prefixes1[0] == prefixes2[0]
}

func (n ResId) concatPrefix(p string) string {
	if p == "" {
		return n.prefix
	}
	if n.prefix == "" {
		return p
	}
	return p + ":" + n.prefix
}

func (n ResId) prefixList() []string {
	return strings.Split(n.prefix, ":")
}
