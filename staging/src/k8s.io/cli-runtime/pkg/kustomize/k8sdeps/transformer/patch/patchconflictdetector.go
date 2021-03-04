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

package patch

import (
	"encoding/json"

	"github.com/evanphx/json-patch"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"sigs.k8s.io/kustomize/pkg/resource"
)

type conflictDetector interface {
	hasConflict(patch1, patch2 *resource.Resource) (bool, error)
	findConflict(conflictingPatchIdx int, patches []*resource.Resource) (*resource.Resource, error)
	mergePatches(patch1, patch2 *resource.Resource) (*resource.Resource, error)
}

type jsonMergePatch struct {
	rf *resource.Factory
}

var _ conflictDetector = &jsonMergePatch{}

func newJMPConflictDetector(rf *resource.Factory) conflictDetector {
	return &jsonMergePatch{rf: rf}
}

func (jmp *jsonMergePatch) hasConflict(
	patch1, patch2 *resource.Resource) (bool, error) {
	return mergepatch.HasConflicts(patch1.Map(), patch2.Map())
}

func (jmp *jsonMergePatch) findConflict(
	conflictingPatchIdx int, patches []*resource.Resource) (*resource.Resource, error) {
	for i, patch := range patches {
		if i == conflictingPatchIdx {
			continue
		}
		if !patches[conflictingPatchIdx].Id().GvknEquals(patch.Id()) {
			continue
		}
		conflict, err := mergepatch.HasConflicts(
			patch.Map(),
			patches[conflictingPatchIdx].Map())
		if err != nil {
			return nil, err
		}
		if conflict {
			return patch, nil
		}
	}
	return nil, nil
}

func (jmp *jsonMergePatch) mergePatches(
	patch1, patch2 *resource.Resource) (*resource.Resource, error) {
	baseBytes, err := json.Marshal(patch1.Map())
	if err != nil {
		return nil, err
	}
	patchBytes, err := json.Marshal(patch2.Map())
	if err != nil {
		return nil, err
	}
	mergedBytes, err := jsonpatch.MergeMergePatches(baseBytes, patchBytes)
	if err != nil {
		return nil, err
	}
	mergedMap := make(map[string]interface{})
	err = json.Unmarshal(mergedBytes, &mergedMap)
	return jmp.rf.FromMap(mergedMap), err
}

type strategicMergePatch struct {
	lookupPatchMeta strategicpatch.LookupPatchMeta
	rf              *resource.Factory
}

var _ conflictDetector = &strategicMergePatch{}

func newSMPConflictDetector(
	versionedObj runtime.Object,
	rf *resource.Factory) (conflictDetector, error) {
	lookupPatchMeta, err := strategicpatch.NewPatchMetaFromStruct(versionedObj)
	return &strategicMergePatch{lookupPatchMeta: lookupPatchMeta, rf: rf}, err
}

func (smp *strategicMergePatch) hasConflict(p1, p2 *resource.Resource) (bool, error) {
	return strategicpatch.MergingMapsHaveConflicts(
		p1.Map(), p2.Map(), smp.lookupPatchMeta)
}

func (smp *strategicMergePatch) findConflict(
	conflictingPatchIdx int, patches []*resource.Resource) (*resource.Resource, error) {
	for i, patch := range patches {
		if i == conflictingPatchIdx {
			continue
		}
		if !patches[conflictingPatchIdx].Id().GvknEquals(patch.Id()) {
			continue
		}
		conflict, err := strategicpatch.MergingMapsHaveConflicts(
			patch.Map(),
			patches[conflictingPatchIdx].Map(),
			smp.lookupPatchMeta)
		if err != nil {
			return nil, err
		}
		if conflict {
			return patch, nil
		}
	}
	return nil, nil
}

func (smp *strategicMergePatch) mergePatches(patch1, patch2 *resource.Resource) (*resource.Resource, error) {
	mergeJSONMap, err := strategicpatch.MergeStrategicMergeMapPatchUsingLookupPatchMeta(
		smp.lookupPatchMeta, patch1.Map(), patch2.Map())
	return smp.rf.FromMap(mergeJSONMap), err
}
