// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinconfig

import (
	"strings"

	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/types"
)

// NameBackReferences is an association between a gvk.GVK and a list
// of FieldSpec instances that could refer to it.
//
// It is used to handle name changes, and can be thought of as a
// a contact list.  If you change your own contact info (name,
// phone number, etc.), you must tell your contacts or they won't
// know about the change.
//
// For example, ConfigMaps can be used by Pods and everything that
// contains a Pod; Deployment, Job, StatefulSet, etc.  To change
// the name of a ConfigMap instance from 'alice' to 'bob', one
// must visit all objects that could refer to the ConfigMap, see if
// they mention 'alice', and if so, change the reference to 'bob'.
//
// The NameBackReferences instance to aid in this could look like
//   {
//     kind: ConfigMap
//     version: v1
//     FieldSpecs:
//     - kind: Pod
//       version: v1
//       path: spec/volumes/configMap/name
//     - kind: Deployment
//       path: spec/template/spec/volumes/configMap/name
//     - kind: Job
//       path: spec/template/spec/volumes/configMap/name
//       (etc.)
//   }
type NameBackReferences struct {
	resid.Gvk  `json:",inline,omitempty" yaml:",inline,omitempty"`
	FieldSpecs types.FsSlice `json:"FieldSpecs,omitempty" yaml:"FieldSpecs,omitempty"`
}

func (n NameBackReferences) String() string {
	var r []string
	for _, f := range n.FieldSpecs {
		r = append(r, f.String())
	}
	return n.Gvk.String() + ":  (\n" +
		strings.Join(r, "\n") + "\n)"
}

type nbrSlice []NameBackReferences

func (s nbrSlice) Len() int      { return len(s) }
func (s nbrSlice) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s nbrSlice) Less(i, j int) bool {
	return s[i].Gvk.IsLessThan(s[j].Gvk)
}

func (s nbrSlice) mergeAll(o nbrSlice) (result nbrSlice, err error) {
	result = s
	for _, r := range o {
		result, err = result.mergeOne(r)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (s nbrSlice) mergeOne(other NameBackReferences) (nbrSlice, error) {
	var result nbrSlice
	var err error
	found := false
	for _, c := range s {
		if c.Gvk.Equals(other.Gvk) {
			c.FieldSpecs, err = c.FieldSpecs.MergeAll(other.FieldSpecs)
			if err != nil {
				return nil, err
			}
			found = true
		}
		result = append(result, c)
	}

	if !found {
		result = append(result, other)
	}
	return result, nil
}
