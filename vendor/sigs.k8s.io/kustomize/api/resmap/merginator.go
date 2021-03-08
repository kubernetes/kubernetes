// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resmap

import (
	"fmt"

	"sigs.k8s.io/kustomize/api/resource"
)

// merginator coordinates merging the resources in incoming to the result.
type merginator struct {
	incoming []*resource.Resource
	cdf      resource.ConflictDetectorFactory
	result   ResMap
}

func (m *merginator) ConflatePatches(in []*resource.Resource) (ResMap, error) {
	m.result = New()
	m.incoming = in
	for index := range m.incoming {
		alreadyInResult, err := m.appendIfNoMatch(index)
		if err != nil {
			return nil, err
		}
		if alreadyInResult != nil {
			// The resource at index has the same resId as a previously
			// considered resource.
			//
			// If they conflict with each other (e.g. they both want to change
			// the image name in a Deployment, but to different values),
			// return an error.
			//
			// If they don't conflict, then merge them into a single resource,
			// since they both target the same item, and we want cumulative
			// behavior. E.g. say both patches modify a map.  Without a merge,
			// the last patch wins, replacing the entire map.
			err = m.mergeWithExisting(index, alreadyInResult)
			if err != nil {
				return nil, err
			}
		}
	}
	return m.result, nil
}

func (m *merginator) appendIfNoMatch(index int) (*resource.Resource, error) {
	candidate := m.incoming[index]
	matchedResources := m.result.GetMatchingResourcesByAnyId(
		candidate.OrgId().Equals)
	if len(matchedResources) == 0 {
		m.result.Append(candidate)
		return nil, nil
	}
	if len(matchedResources) > 1 {
		return nil, fmt.Errorf("multiple resources targeted by patch")
	}
	return matchedResources[0], nil
}

func (m *merginator) mergeWithExisting(
	index int, alreadyInResult *resource.Resource) error {
	candidate := m.incoming[index]
	cd, err := m.cdf.New(candidate.OrgId().Gvk)
	if err != nil {
		return err
	}
	hasConflict, err := cd.HasConflict(candidate, alreadyInResult)
	if err != nil {
		return err
	}
	if hasConflict {
		return m.makeError(cd, index)
	}
	merged, err := cd.MergePatches(alreadyInResult, candidate)
	if err != nil {
		return err
	}
	_, err = m.result.Replace(merged)
	return err
}

// Make an error message describing the conflict.
func (m *merginator) makeError(cd resource.ConflictDetector, index int) error {
	conflict, err := m.findConflict(cd, index)
	if err != nil {
		return err
	}
	if conflict == nil {
		return fmt.Errorf("expected conflict for %s", m.incoming[index].OrgId())
	}
	conflictMap, _ := conflict.Map()
	incomingIndexMap, _ := m.incoming[index].Map()
	return fmt.Errorf(
		"conflict between %#v at index %d and %#v",
		incomingIndexMap,
		index,
		conflictMap,
	)
}

// findConflict looks for a conflict in a resource slice.
// It returns the first conflict between the resource at index
// and some other resource.  Two resources can only conflict if
// they have the same original ResId.
func (m *merginator) findConflict(
	cd resource.ConflictDetector, index int) (*resource.Resource, error) {
	targetId := m.incoming[index].OrgId()
	for i, p := range m.incoming {
		if i == index || !targetId.Equals(p.OrgId()) {
			continue
		}
		conflict, err := cd.HasConflict(p, m.incoming[index])
		if err != nil {
			return nil, err
		}
		if conflict {
			return p, nil
		}
	}
	return nil, nil
}
