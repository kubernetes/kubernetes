// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinconfig

import (
	"strings"

	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/resid"
)

// NameBackReferences is an association between a gvk.GVK (a ReferralTarget)
// and a list of Referrers that could refer to it.
//
// It is used to handle name changes, and can be thought of as a
// a contact list.  If you change your own contact info (name,
// phone number, etc.), you must tell your contacts or they won't
// know about the change.
//
// For example, ConfigMaps can be used by Pods and everything that
// contains a Pod; Deployment, Job, StatefulSet, etc.
// The ConfigMap is the ReferralTarget, the others are Referrers.
//
// If the name of a ConfigMap instance changed from 'alice' to 'bob',
// one must
//  - visit all objects that could refer to the ConfigMap (the Referrers)
//  - see if they mention 'alice',
//  - if so, change the Referrer's name reference to 'bob'.
//
// The NameBackReferences instance to aid in this could look like
//   {
//     kind: ConfigMap
//     version: v1
//     fieldSpecs:
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
	resid.Gvk `json:",inline,omitempty" yaml:",inline,omitempty"`
	// TODO: rename json 'fieldSpecs' to 'referrers' for clarity.
	// This will, however, break anyone using a custom config.
	Referrers types.FsSlice `json:"fieldSpecs,omitempty" yaml:"fieldSpecs,omitempty"`

	// Note: If any new pointer based members are added, DeepCopy needs to be updated
}

func (n NameBackReferences) String() string {
	var r []string
	for _, f := range n.Referrers {
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

// DeepCopy returns a new copy of nbrSlice
func (s nbrSlice) DeepCopy() nbrSlice {
	ret := make(nbrSlice, len(s))
	copy(ret, s)
	for i, slice := range ret {
		ret[i].Referrers = slice.Referrers.DeepCopy()
	}

	return ret
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
			c.Referrers, err = c.Referrers.MergeAll(other.Referrers)
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
