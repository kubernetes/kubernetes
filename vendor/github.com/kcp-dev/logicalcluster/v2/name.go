/*
Copyright 2022 The KCP Authors.

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

package logicalcluster

import (
	"encoding/json"
	"path"
	"regexp"
	"strings"
)

// ClusterHeader set to "<lcluster>" on a request is an alternative to accessing the
// cluster via /clusters/<lcluster>. With that the <lcluster> can be access via normal kube-like
// /api and /apis endpoints.
const ClusterHeader = "X-Kubernetes-Cluster"

// Name is the name of a logical cluster. A logical cluster is
// 1. a (part of) etcd prefix to store objects in that cluster
// 2. a (part of) a http path which serves a Kubernetes-cluster-like API with
//    discovery, OpenAPI and the actual API groups.
// 3. a value in metadata.clusterName in objects from cross-workspace list/watches,
//    which is used to identify the logical cluster.
//
// A logical cluster is a colon separated list of words. In other words, it is
// like a path, but with colons instead of slashes.
type Name struct {
	value string
}

const separator = ":"

var (
	// Wildcard is the name indicating cross-workspace requests.
	Wildcard = New("*")
)

// New returns a Name from a string.
func New(value string) Name {
	return Name{value}
}

// NewValidated returns a Name from a string and whether it is a valid logical cluster.
// A valid logical cluster returns true on IsValid().
func NewValidated(value string) (Name, bool) {
	n := Name{value}
	return n, n.IsValid()
}

// Empty returns true if the logical cluster value is unset.
func (n Name) Empty() bool {
	return n.value == ""
}

// Path returns a path segment for the logical cluster to access its API.
func (n Name) Path() string {
	return path.Join("/clusters", n.value)
}

// String returns the string representation of the logical cluster name.
func (n Name) String() string {
	return n.value
}

// Object is a local interface representation of the Kubernetes metav1.Object, to avoid dependencies on
// k8s.io/apimachinery.
type Object interface {
	GetAnnotations() map[string]string
}

// AnnotationKey is the name of the annotation key used to denote an object's logical cluster.
const AnnotationKey = "kcp.dev/cluster"

// From returns the logical cluster name for obj.
func From(obj Object) Name {
	return Name{obj.GetAnnotations()[AnnotationKey]}
}

// Parent returns the parent logical cluster name of the given logical cluster name.
func (n Name) Parent() (Name, bool) {
	parent, _ := n.Split()
	return parent, parent.value != ""
}

// Split splits logical cluster immediately following the final colon,
// separating it into a parent logical cluster and name component.
// If there is no colon in path, Split returns an empty logical cluster name
// and name set to path.
func (n Name) Split() (parent Name, name string) {
	i := strings.LastIndex(n.value, separator)
	if i < 0 {
		return Name{}, n.value
	}
	return Name{n.value[:i]}, n.value[i+1:]
}

// Base returns the last component of the logical cluster name.
func (n Name) Base() string {
	_, name := n.Split()
	return name
}

// Join joins a parent logical cluster name and a name component.
func (n Name) Join(name string) Name {
	if n.value == "" {
		return Name{name}
	}
	return Name{n.value + separator + name}
}

func (n Name) MarshalJSON() ([]byte, error) {
	return json.Marshal(&n.value)
}

func (n *Name) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	n.value = s
	return nil
}

func (n Name) HasPrefix(other Name) bool {
	return strings.HasPrefix(n.value, other.value)
}

var lclusterRegExp = regexp.MustCompile(`^[a-z][a-z0-9-]*[a-z0-9](:[a-z][a-z0-9-]*[a-z0-9])*$`)

// IsValid returns true if the name is a Wildcard or a colon separated list of words where each word
// starts with a lower-case letter and contains only lower-case letters, digits and hyphens.
func (n Name) IsValid() bool {
	return n == Wildcard || lclusterRegExp.MatchString(n.value)
}
