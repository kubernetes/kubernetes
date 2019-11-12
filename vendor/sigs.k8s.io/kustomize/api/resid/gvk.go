// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resid

import (
	"strings"
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

// GvkFromString makes a Gvk with a string,
// which is constructed by String() function
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
	"MutatingWebhookConfiguration",
	"ServiceAccount",
	"PodSecurityPolicy",
	"Role",
	"ClusterRole",
	"RoleBinding",
	"ClusterRoleBinding",
	"ConfigMap",
	"Secret",
	"Service",
	"LimitRange",
	"PriorityClass",
	"Deployment",
	"StatefulSet",
	"CronJob",
	"PodDisruptionBudget",
}
var orderLast = []string{
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

var notNamespaceableKinds = []string{
	"APIService",
	"CSIDriver",
	"CSINode",
	"CertificateSigningRequest",
	"ClusterRole",
	"ClusterRoleBinding",
	"ComponentStatus",
	"CustomResourceDefinition",
	"MutatingWebhookConfiguration",
	"Namespace",
	"Node",
	"PersistentVolume",
	"PodSecurityPolicy",
	"PriorityClass",
	"RuntimeClass",
	"SelfSubjectAccessReview",
	"SelfSubjectRulesReview",
	"StorageClass",
	"SubjectAccessReview",
	"TokenReview",
	"ValidatingWebhookConfiguration",
	"VolumeAttachment",
}

// IsNamespaceableKind returns true if x is a namespaceable Gvk
// Implements https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/#not-all-objects-are-in-a-namespace
func (x Gvk) IsNamespaceableKind() bool {
	for _, k := range notNamespaceableKinds {
		if k == x.Kind {
			return false
		}
	}
	return true
}
