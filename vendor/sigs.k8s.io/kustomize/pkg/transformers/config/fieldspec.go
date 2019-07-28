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

package config

import (
	"fmt"
	"strings"

	"sigs.k8s.io/kustomize/pkg/gvk"
)

// FieldSpec completely specifies a kustomizable field in
// an unstructured representation of a k8s API object.
// It helps define the operands of transformations.
//
// For example, a directive to add a common label to objects
// will need to know that a 'Deployment' object (in API group
// 'apps', any version) can have labels at field path
// 'spec/template/metadata/labels', and further that it is OK
// (or not OK) to add that field path to the object if the
// field path doesn't exist already.
//
// This would look like
// {
//   group: apps
//   kind: Deployment
//   path: spec/template/metadata/labels
//   create: true
// }
type FieldSpec struct {
	gvk.Gvk            `json:",inline,omitempty" yaml:",inline,omitempty"`
	Path               string `json:"path,omitempty" yaml:"path,omitempty"`
	CreateIfNotPresent bool   `json:"create,omitempty" yaml:"create,omitempty"`
}

const (
	escapedForwardSlash  = "\\/"
	tempSlashReplacement = "???"
)

func (fs FieldSpec) String() string {
	return fmt.Sprintf(
		"%s:%v:%s", fs.Gvk.String(), fs.CreateIfNotPresent, fs.Path)
}

// If true, the primary key is the same, but other fields might not be.
func (fs FieldSpec) effectivelyEquals(other FieldSpec) bool {
	return fs.IsSelected(&other.Gvk) && fs.Path == other.Path
}

// PathSlice converts the path string to a slice of strings,
// separated by a '/'. Forward slash can be contained in a
// fieldname. such as ingress.kubernetes.io/auth-secret in
// Ingress annotations. To deal with this special case, the
// path to this field should be formatted as
//
//   metadata/annotations/ingress.kubernetes.io\/auth-secret
//
// Then PathSlice will return
//
//   []string{
//      "metadata",
//      "annotations",
//      "ingress.auth-secretkubernetes.io/auth-secret"
//   }
func (fs FieldSpec) PathSlice() []string {
	if !strings.Contains(fs.Path, escapedForwardSlash) {
		return strings.Split(fs.Path, "/")
	}
	s := strings.Replace(fs.Path, escapedForwardSlash, tempSlashReplacement, -1)
	paths := strings.Split(s, "/")
	var result []string
	for _, path := range paths {
		result = append(result, strings.Replace(path, tempSlashReplacement, "/", -1))
	}
	return result
}

type fsSlice []FieldSpec

func (s fsSlice) Len() int      { return len(s) }
func (s fsSlice) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s fsSlice) Less(i, j int) bool {
	return s[i].Gvk.IsLessThan(s[j].Gvk)
}

// mergeAll merges the argument into this, returning the result.
// Items already present are ignored.
// Items that conflict (primary key matches, but remain data differs)
// result in an error.
func (s fsSlice) mergeAll(incoming fsSlice) (result fsSlice, err error) {
	result = s
	for _, x := range incoming {
		result, err = result.mergeOne(x)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// mergeOne merges the argument into this, returning the result.
// If the item's primary key is already present, and there are no
// conflicts, it is ignored (we don't want duplicates).
// If there is a conflict, the merge fails.
func (s fsSlice) mergeOne(x FieldSpec) (fsSlice, error) {
	i := s.index(x)
	if i > -1 {
		// It's already there.
		if s[i].CreateIfNotPresent != x.CreateIfNotPresent {
			return nil, fmt.Errorf("conflicting fieldspecs")
		}
		return s, nil
	}
	return append(s, x), nil
}

func (s fsSlice) index(fs FieldSpec) int {
	for i, x := range s {
		if x.effectivelyEquals(fs) {
			return i
		}
	}
	return -1
}
