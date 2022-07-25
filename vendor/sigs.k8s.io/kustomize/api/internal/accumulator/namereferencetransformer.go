// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package accumulator

import (
	"fmt"
	"log"

	"sigs.k8s.io/kustomize/api/filters/nameref"
	"sigs.k8s.io/kustomize/api/internal/plugins/builtinconfig"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/kyaml/resid"
)

type nameReferenceTransformer struct {
	backRefs []builtinconfig.NameBackReferences
}

const doDebug = false

var _ resmap.Transformer = &nameReferenceTransformer{}

type filterMap map[*resource.Resource][]nameref.Filter

// newNameReferenceTransformer constructs a nameReferenceTransformer
// with a given slice of NameBackReferences.
func newNameReferenceTransformer(
	br []builtinconfig.NameBackReferences) resmap.Transformer {
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
// an HPA scales. In this case:
//
//   - the HPA instance is the Referrer,
//   - the Deployment instance is the ReferralTarget.
//
// If the Deployment's name changes, e.g. a prefix is added,
// then the HPA's reference to the Deployment must be fixed.
//
func (t *nameReferenceTransformer) Transform(m resmap.ResMap) error {
	fMap := t.determineFilters(m.Resources())
	debug(fMap)
	for r, fList := range fMap {
		c, err := m.SubsetThatCouldBeReferencedByResource(r)
		if err != nil {
			return err
		}
		for _, f := range fList {
			f.Referrer = r
			f.ReferralCandidates = c
			if err := f.Referrer.ApplyFilter(f); err != nil {
				return err
			}
		}
	}
	return nil
}

func debug(fMap filterMap) {
	if !doDebug {
		return
	}
	fmt.Printf("filterMap has %d entries:\n", len(fMap))
	rCount := 0
	for r, fList := range fMap {
		yml, _ := r.AsYAML()
		rCount++
		fmt.Printf(`
---- %3d. possible referrer -------------
%s
---------`, rCount, string(yml),
		)
		for i, f := range fList {
			fmt.Printf(`
%3d/%3d update: %s
          from: %s
`, rCount, i+1, f.NameFieldToUpdate.Path, f.ReferralTarget,
			)
		}
	}
}

// Produce a map from referrer resources that might need to be fixed
// to filters that might fix them.  The keys to this map are potential
// referrers, so won't include resources like ConfigMap or Secret.
//
// In the inner loop over the resources below, say we
// encounter an HPA instance. Then, in scanning the set
// of all known backrefs, we encounter an entry like
//
//   - kind: Deployment
//     fieldSpecs:
//     - kind: HorizontalPodAutoscaler
//       path: spec/scaleTargetRef/name
//
// This entry says that an HPA, via its
// 'spec/scaleTargetRef/name' field, may refer to a
// Deployment.
//
// This means that a filter will need to hunt for the right Deployment,
// obtain it's new name, and write that name into the HPA's
// 'spec/scaleTargetRef/name' field. Return a filter that can do that.
func (t *nameReferenceTransformer) determineFilters(
	resources []*resource.Resource) (fMap filterMap) {

	// We cache the resource OrgId values because they don't change and otherwise are very visible in a memory pprof
	resourceOrgIds := make([]resid.ResId, len(resources))
	for i, resource := range resources {
		resourceOrgIds[i] = resource.OrgId()
	}

	fMap = make(filterMap)
	for _, backReference := range t.backRefs {
		for _, referrerSpec := range backReference.Referrers {
			for i, res := range resources {
				if resourceOrgIds[i].IsSelected(&referrerSpec.Gvk) {
					// If this is true, the res might be a referrer, and if
					// so, the name reference it holds might need an update.
					if resHasField(res, referrerSpec.Path) {
						// Optimization - the referrer has the field
						// that might need updating.
						fMap[res] = append(fMap[res], nameref.Filter{
							// Name field to write in the Referrer.
							// If the path specified here isn't found in
							// the Referrer, nothing happens (no error,
							// no field creation).
							NameFieldToUpdate: referrerSpec,
							// Specification of object class to read from.
							// Always read from metadata/name field.
							ReferralTarget: backReference.Gvk,
						})
					}
				}
			}
		}
	}
	return fMap
}

// TODO: check res for field existence here to avoid extra work.
// res.GetFieldValue, which uses yaml.Lookup under the hood, doesn't know
// how to parse fieldspec-style paths that make no distinction
// between maps and sequences.  This means it cannot lookup commonly
// used "indeterminate" paths like
//    spec/containers/env/valueFrom/configMapKeyRef/name
// ('containers' is a list, not a map).
// However, the fieldspec filter does know how to handle this;
// extract that code and call it here?
func resHasField(res *resource.Resource, path string) bool {
	return true
	// fld := strings.Join(utils.PathSplitter(path), ".")
	// _, e := res.GetFieldValue(fld)
	// return e == nil
}
