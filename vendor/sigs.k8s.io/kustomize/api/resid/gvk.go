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
}

// FromKind makes a Gvk with only the kind specified.
func FromKind(k string) Gvk {
	return Gvk{
		Kind: k,
	}
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
	g := values[0]
	if g == noGroup {
		g = ""
	}
	v := values[1]
	if v == noVersion {
		v = ""
	}
	k := values[2]
	if k == noKind {
		k = ""
	}
	return Gvk{
		Group:   g,
		Version: v,
		Kind:    k,
	}
}

// Values that are brief but meaningful in logs.
const (
	noGroup   = "~G"
	noVersion = "~V"
	noKind    = "~K"
	fieldSep  = "_"
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
	return strings.Join([]string{g, v, k}, fieldSep)
}

// ApiVersion returns the combination of Group and Version
func (x Gvk) ApiVersion() string {
	if x.Group == "" {
		return x.Version
	}
	return x.Group + "/" + x.Version
}

// StringWoEmptyField returns a string representation of the GVK. Non-exist
// fields will be omitted.
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
	return strings.Join(s, fieldSep)
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
	return x.String() < o.String()
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

// toKyamlTypeMeta returns a yaml.TypeMeta from x's information.
func (x Gvk) toKyamlTypeMeta() yaml.TypeMeta {
	var apiVersion strings.Builder
	if x.Group != "" {
		apiVersion.WriteString(x.Group)
		apiVersion.WriteString("/")
	}
	apiVersion.WriteString(x.Version)
	return yaml.TypeMeta{
		APIVersion: apiVersion.String(),
		Kind:       x.Kind,
	}
}

// IsNamespaceableKind returns true if x is a namespaceable Gvk
// Implements https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/#not-all-objects-are-in-a-namespace
func (x Gvk) IsNamespaceableKind() bool {
	isNamespaceScoped, found := openapi.IsNamespaceScoped(x.toKyamlTypeMeta())
	return !found || isNamespaceScoped
}
