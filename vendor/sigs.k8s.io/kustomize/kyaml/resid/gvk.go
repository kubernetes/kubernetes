// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resid

import (
	"strings"

	"sigs.k8s.io/kustomize/kyaml/openapi"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Gvk identifies a Kubernetes API type.
// https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/api-group.md
type Gvk struct {
	Group   string `json:"group,omitempty" yaml:"group,omitempty"`
	Version string `json:"version,omitempty" yaml:"version,omitempty"`
	Kind    string `json:"kind,omitempty" yaml:"kind,omitempty"`
	// isClusterScoped is true if the object is known, per the openapi
	// data in use, to be cluster scoped, and false otherwise.
	isClusterScoped bool
}

func NewGvk(g, v, k string) Gvk {
	result := Gvk{Group: g, Version: v, Kind: k}
	result.isClusterScoped =
		openapi.IsCertainlyClusterScoped(result.AsTypeMeta())
	return result
}

func GvkFromNode(r *yaml.RNode) Gvk {
	g, v := ParseGroupVersion(r.GetApiVersion())
	return NewGvk(g, v, r.GetKind())
}

// FromKind makes a Gvk with only the kind specified.
func FromKind(k string) Gvk {
	return NewGvk("", "", k)
}

// ParseGroupVersion parses a KRM metadata apiVersion field.
func ParseGroupVersion(apiVersion string) (group, version string) {
	if i := strings.Index(apiVersion, "/"); i > -1 {
		return apiVersion[:i], apiVersion[i+1:]
	}
	return "", apiVersion
}

// GvkFromString makes a Gvk from the output of Gvk.String().
func GvkFromString(s string) Gvk {
	values := strings.Split(s, fieldSep)
	if len(values) < 3 {
		// ...then the string didn't come from Gvk.String().
		return Gvk{
			Group:   noGroup,
			Version: noVersion,
			Kind:    noKind,
		}
	}
	k := values[0]
	if k == noKind {
		k = ""
	}
	v := values[1]
	if v == noVersion {
		v = ""
	}
	g := strings.Join(values[2:], fieldSep)
	if g == noGroup {
		g = ""
	}
	return NewGvk(g, v, k)
}

// Values that are brief but meaningful in logs.
const (
	noGroup   = "[noGrp]"
	noVersion = "[noVer]"
	noKind    = "[noKind]"
	fieldSep  = "."
)

// String returns a string representation of the GVK.
func (x Gvk) String() string {
	g := x.Group
	if g == "" {
		g = noGroup
	}
	v := x.Version
	if v == "" {
		v = noVersion
	}
	k := x.Kind
	if k == "" {
		k = noKind
	}
	return strings.Join([]string{k, v, g}, fieldSep)
}

// legacySortString returns an older version of String() that LegacyOrderTransformer depends on
// to keep its ordering stable across Kustomize versions
func (x Gvk) legacySortString() string {
	legacyNoGroup := "~G"
	legacyNoVersion := "~V"
	legacyNoKind := "~K"
	legacyFieldSeparator := "_"

	g := x.Group
	if g == "" {
		g = legacyNoGroup
	}
	v := x.Version
	if v == "" {
		v = legacyNoVersion
	}
	k := x.Kind
	if k == "" {
		k = legacyNoKind
	}
	return strings.Join([]string{g, v, k}, legacyFieldSeparator)
}

// ApiVersion returns the combination of Group and Version
func (x Gvk) ApiVersion() string {
	var sb strings.Builder
	if x.Group != "" {
		sb.WriteString(x.Group)
		sb.WriteString("/")
	}
	sb.WriteString(x.Version)
	return sb.String()
}

// StringWoEmptyField returns a string representation of the GVK. Non-exist
// fields will be omitted. This is called when generating a filename for the
// resource.
func (x Gvk) StringWoEmptyField() string {
	var s []string
	if x.Group != "" {
		s = append(s, x.Group)
	}
	if x.Version != "" {
		s = append(s, x.Version)
	}
	if x.Kind != "" {
		s = append(s, x.Kind)
	}
	return strings.Join(s, "_")
}

// Equals returns true if the Gvk's have equal fields.
func (x Gvk) Equals(o Gvk) bool {
	return x.Group == o.Group && x.Version == o.Version && x.Kind == o.Kind
}

// An attempt to order things to help k8s, e.g.
// a Service should come before things that refer to it.
// Namespace should be first.
// In some cases order just specified to provide determinism.
var orderFirst = []string{
	"Namespace",
	"ResourceQuota",
	"StorageClass",
	"CustomResourceDefinition",
	"ServiceAccount",
	"PodSecurityPolicy",
	"Role",
	"ClusterRole",
	"RoleBinding",
	"ClusterRoleBinding",
	"ConfigMap",
	"Secret",
	"Endpoints",
	"Service",
	"LimitRange",
	"PriorityClass",
	"PersistentVolume",
	"PersistentVolumeClaim",
	"Deployment",
	"StatefulSet",
	"CronJob",
	"PodDisruptionBudget",
}
var orderLast = []string{
	"MutatingWebhookConfiguration",
	"ValidatingWebhookConfiguration",
}
var typeOrders = func() map[string]int {
	m := map[string]int{}
	for i, n := range orderFirst {
		m[n] = -len(orderFirst) + i
	}
	for i, n := range orderLast {
		m[n] = 1 + i
	}
	return m
}()

// IsLessThan returns true if self is less than the argument.
func (x Gvk) IsLessThan(o Gvk) bool {
	indexI := typeOrders[x.Kind]
	indexJ := typeOrders[o.Kind]
	if indexI != indexJ {
		return indexI < indexJ
	}
	return x.legacySortString() < o.legacySortString()
}

// IsSelected returns true if `selector` selects `x`; otherwise, false.
// If `selector` and `x` are the same, return true.
// If `selector` is nil, it is considered a wildcard match, returning true.
// If selector fields are empty, they are considered wildcards matching
// anything in the corresponding fields, e.g.
//
// this item:
//       <Group: "extensions", Version: "v1beta1", Kind: "Deployment">
//
// is selected by
//       <Group: "",           Version: "",        Kind: "Deployment">
//
// but rejected by
//       <Group: "apps",       Version: "",        Kind: "Deployment">
//
func (x Gvk) IsSelected(selector *Gvk) bool {
	if selector == nil {
		return true
	}
	if len(selector.Group) > 0 {
		if x.Group != selector.Group {
			return false
		}
	}
	if len(selector.Version) > 0 {
		if x.Version != selector.Version {
			return false
		}
	}
	if len(selector.Kind) > 0 {
		if x.Kind != selector.Kind {
			return false
		}
	}
	return true
}

// AsTypeMeta returns a yaml.TypeMeta from x's information.
func (x Gvk) AsTypeMeta() yaml.TypeMeta {
	return yaml.TypeMeta{
		APIVersion: x.ApiVersion(),
		Kind:       x.Kind,
	}
}

// IsClusterScoped returns true if the Gvk is certainly cluster scoped
// with respect to the available openapi data.
func (x Gvk) IsClusterScoped() bool {
	return x.isClusterScoped
}
