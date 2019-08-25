// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transformers

import (
	"fmt"
	"log"

	"sigs.k8s.io/kustomize/pkg/resource"

	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type nameReferenceTransformer struct {
	backRefs []config.NameBackReferences
}

var _ Transformer = &nameReferenceTransformer{}

// NewNameReferenceTransformer constructs a nameReferenceTransformer
// with a given slice of NameBackReferences.
func NewNameReferenceTransformer(br []config.NameBackReferences) Transformer {
	if br == nil {
		log.Fatal("backrefs not expected to be nil")
	}
	return &nameReferenceTransformer{backRefs: br}
}

// Transform updates name references in resource A that
// refer to resource B, given that B's name may have
// changed.
//
// For example, a HorizontalPodAutoscaler (HPA)
// necessarily refers to a Deployment, the thing that
// the HPA scales. The Deployment name might change
// (e.g. prefix added), and the reference in the HPA
// has to be fixed.
//
// In the outer loop over the ResMap below, say we
// encounter a specific HPA. Then, in scanning backrefs,
// we encounter an entry like
//
//   - kind: Deployment
//     fieldSpecs:
//     - kind: HorizontalPodAutoscaler
//       path: spec/scaleTargetRef/name
//
// This entry says that an HPA, via its
// 'spec/scaleTargetRef/name' field, may refer to a
// Deployment.  This match to HPA means we may need to
// modify the value in its 'spec/scaleTargetRef/name'
// field, by searching for the thing it refers to,
// and getting its new name.
//
// As a filter, and search optimization, we compute a
// subset of all resources that the HPA could refer to,
// by excluding objects from other namespaces, and
// excluding objects that don't have the same prefix-
// suffix mods as the HPA.
//
// We look in this subset for all Deployment objects
// with a resId that has a Name matching the field value
// present in the HPA.  If no match do nothing; if more
// than one match, it's an error.
//
// We overwrite the HPA name field with the value found
// in the Deployment's name field (the name in the raw
// object - the modified name - not the unmodified name
// in the Deployment's resId).
//
// This process assumes that the name stored in a ResId
// (the ResMap key) isn't modified by name transformers.
// Name transformers should only modify the name in the
// body of the resource object (the value in the ResMap).
//
func (o *nameReferenceTransformer) Transform(m resmap.ResMap) error {
	// TODO: Too much looping, here and in transitive calls.
	for _, referrer := range m.Resources() {
		var candidates resmap.ResMap
		for _, target := range o.backRefs {
			for _, fSpec := range target.FieldSpecs {
				if referrer.OrgId().IsSelected(&fSpec.Gvk) {
					if candidates == nil {
						candidates = m.SubsetThatCouldBeReferencedByResource(referrer)
					}
					err := MutateField(
						referrer.Map(),
						fSpec.PathSlice(),
						fSpec.CreateIfNotPresent,
						o.getNewNameFunc(
							// referrer could be an HPA instance,
							// target could be Gvk for Deployment,
							// candidate a list of resources "reachable"
							// from the HPA.
							referrer, target.Gvk, candidates))
					if err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func (o *nameReferenceTransformer) getNewNameFunc(
	referrer *resource.Resource,
	target gvk.Gvk,
	referralCandidates resmap.ResMap) func(in interface{}) (interface{}, error) {
	return func(in interface{}) (interface{}, error) {
		switch in.(type) {
		case string:
			oldName, _ := in.(string)
			for _, res := range referralCandidates.Resources() {
				id := res.OrgId()
				if id.IsSelected(&target) && res.GetOriginalName() == oldName {
					matches := referralCandidates.GetMatchingResourcesByOriginalId(id.GvknEquals)
					// If there's more than one match, there's no way
					// to know which one to pick, so emit error.
					if len(matches) > 1 {
						return nil, fmt.Errorf(
							"string case - multiple matches for %s:\n  %v",
							id, getIds(matches))
					}
					// In the resource, note that it is referenced
					// by the referrer.
					res.AppendRefBy(referrer.CurId())
					// Return transformed name of the object,
					// complete with prefixes, hashes, etc.
					return res.GetName(), nil
				}
			}
			return in, nil
		case []interface{}:
			l, _ := in.([]interface{})
			var names []string
			for _, item := range l {
				name, ok := item.(string)
				if !ok {
					return nil, fmt.Errorf(
						"%#v is expected to be %T", item, name)
				}
				names = append(names, name)
			}
			for _, res := range referralCandidates.Resources() {
				indexes := indexOf(res.GetOriginalName(), names)
				id := res.OrgId()
				if id.IsSelected(&target) && len(indexes) > 0 {
					matches := referralCandidates.GetMatchingResourcesByOriginalId(id.GvknEquals)
					if len(matches) > 1 {
						return nil, fmt.Errorf(
							"slice case - multiple matches for %s:\n %v",
							id, getIds(matches))
					}
					for _, index := range indexes {
						l[index] = res.GetName()
					}
					res.AppendRefBy(referrer.CurId())
					return l, nil
				}
			}
			return in, nil
		default:
			return nil, fmt.Errorf(
				"%#v is expected to be either a string or a []interface{}", in)
		}
	}
}

func indexOf(s string, slice []string) []int {
	var index []int
	for i, item := range slice {
		if item == s {
			index = append(index, i)
		}
	}
	return index
}

func getIds(rs []*resource.Resource) []string {
	var result []string
	for _, r := range rs {
		result = append(result, r.CurId().String()+"\n")
	}
	return result
}
