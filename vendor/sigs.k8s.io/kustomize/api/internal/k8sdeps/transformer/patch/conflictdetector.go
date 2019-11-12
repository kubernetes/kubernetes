// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package patch

import (
	"encoding/json"
	"fmt"

	jsonpatch "github.com/evanphx/json-patch"
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/mergepatch"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/strategicpatch"
	"sigs.k8s.io/kustomize/pseudo/k8s/client-go/kubernetes/scheme"
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
		if !patches[conflictingPatchIdx].OrgId().Equals(patch.OrgId()) {
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
		if !patches[conflictingPatchIdx].OrgId().Equals(patch.OrgId()) {
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
	if hasDeleteDirectiveMarker(patch2.Map()) {
		if hasDeleteDirectiveMarker(patch1.Map()) {
			return nil, fmt.Errorf("cannot merge patches both containing '$patch: delete' directives")
		}
		patch1, patch2 = patch2, patch1
	}
	mergeJSONMap, err := strategicpatch.MergeStrategicMergeMapPatchUsingLookupPatchMeta(
		smp.lookupPatchMeta, patch1.Map(), patch2.Map())
	return smp.rf.FromMap(mergeJSONMap), err
}

// MergePatches merge and index patches by OrgId.
// It errors out if there is conflict between patches.
func MergePatches(patches []*resource.Resource,
	rf *resource.Factory) (resmap.ResMap, error) {
	rc := resmap.New()
	for ix, patch := range patches {
		id := patch.OrgId()
		existing := rc.GetMatchingResourcesByOriginalId(id.Equals)
		if len(existing) == 0 {
			rc.Append(patch)
			continue
		}
		if len(existing) > 1 {
			return nil, fmt.Errorf("self conflict in patches")
		}

		versionedObj, err := scheme.Scheme.New(toSchemaGvk(id.Gvk))
		if err != nil && !runtime.IsNotRegisteredError(err) {
			return nil, err
		}
		var cd conflictDetector
		if err != nil {
			cd = newJMPConflictDetector(rf)
		} else {
			cd, err = newSMPConflictDetector(versionedObj, rf)
			if err != nil {
				return nil, err
			}
		}

		conflict, err := cd.hasConflict(existing[0], patch)
		if err != nil {
			return nil, err
		}
		if conflict {
			conflictingPatch, err := cd.findConflict(ix, patches)
			if err != nil {
				return nil, err
			}
			return nil, fmt.Errorf(
				"conflict between %#v and %#v",
				conflictingPatch.Map(), patch.Map())
		}
		merged, err := cd.mergePatches(existing[0], patch)
		if err != nil {
			return nil, err
		}
		rc.Replace(merged)
	}
	return rc, nil
}

// toSchemaGvk converts to a schema.GroupVersionKind.
func toSchemaGvk(x resid.Gvk) schema.GroupVersionKind {
	return schema.GroupVersionKind{
		Group:   x.Group,
		Version: x.Version,
		Kind:    x.Kind,
	}
}

func hasDeleteDirectiveMarker(patch map[string]interface{}) bool {
	if v, ok := patch["$patch"]; ok && v == "delete" {
		return true
	}
	for _, v := range patch {
		switch typedV := v.(type) {
		case map[string]interface{}:
			if hasDeleteDirectiveMarker(typedV) {
				return true
			}
		case []interface{}:
			for _, sv := range typedV {
				typedE, ok := sv.(map[string]interface{})
				if !ok {
					break
				}
				if hasDeleteDirectiveMarker(typedE) {
					return true
				}
			}
		}
	}
	return false
}
