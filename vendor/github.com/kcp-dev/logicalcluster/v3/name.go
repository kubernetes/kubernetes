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
	"regexp"
	"strings"
)

var (
	clusterNameRegExp = regexp.MustCompile(clusterNameString)
)

const (
	clusterNameString string = "^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$"
)

// Name holds a value that uniquely identifies a logical cluster.
// For instance, a logical cluster may have the name 33bab531.
//
// A logical cluster is a partition on the storage layer served as autonomous kube-like generic API endpoint.
type Name string

// Path creates a new Path object from the logical cluster name.
// A convenience method for working with methods which accept a Path type.
func (n Name) Path() Path {
	return NewPath(string(n))
}

// String returns string representation of the logical cluster name.
// Satisfies the Stringer interface.
func (n Name) String() string {
	return string(n)
}

// IsValid returns true if the logical cluster name matches a defined format.
// A convenience method that could be used for enforcing a well-known structure of a logical cluster name.
//
// As of today a valid value starts with a lower-case letter or digit
// and contains only lower-case letters, digits and hyphens.
func (n Name) IsValid() bool {
	return strings.HasPrefix(string(n), "system:") || clusterNameRegExp.MatchString(string(n))
}

// Empty returns true if the logical cluster name is unset.
// It is a convenience method for checking against an empty value.
func (n Name) Empty() bool {
	return n == ""
}

// Object is a local interface representation of the Kubernetes metav1.Object, to avoid dependencies on k8s.io/apimachinery.
type Object interface {
	GetAnnotations() map[string]string
}

// AnnotationKey is the name of the annotation key used to denote an object's logical cluster.
const AnnotationKey = "kcp.io/cluster"

// From returns the logical cluster name from the given object.
func From(obj Object) Name {
	return Name(obj.GetAnnotations()[AnnotationKey])
}
