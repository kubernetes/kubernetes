// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package accumulator

import (
	"fmt"
	"log"

	"sigs.k8s.io/kustomize/api/internal/plugins/builtinconfig"
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/transform"
)

type nameReferenceTransformer struct {
	backRefs []builtinconfig.NameBackReferences
}

var _ resmap.Transformer = &nameReferenceTransformer{}

// newNameReferenceTransformer constructs a nameReferenceTransformer
// with a given slice of NameBackReferences.
func newNameReferenceTransformer(br []builtinconfig.NameBackReferences) resmap.Transformer {
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
					err := transform.MutateField(
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

// selectReferral picks the referral among a subset of candidates.
// It returns the current name and namespace of the selected candidate.
// Note that the content of the referricalCandidateSubset slice is most of the time
// identical to the referralCandidates resmap. Still in some cases, such
// as ClusterRoleBinding, the subset only contains the resources of a specific
// namespace.
func (o *nameReferenceTransformer) selectReferral(
	oldName string,
	referrer *resource.Resource,
	target resid.Gvk,
	referralCandidates resmap.ResMap,
	referralCandidateSubset []*resource.Resource) (interface{}, interface{}, error) {

	for _, res := range referralCandidateSubset {
		id := res.OrgId()
		if id.IsSelected(&target) && res.GetOriginalName() == oldName {
			matches := referralCandidates.GetMatchingResourcesByOriginalId(id.Equals)
			// If there's more than one match, there's no way
			// to know which one to pick, so emit error.
			if len(matches) > 1 {
				return nil, nil, fmt.Errorf(
					"multiple matches for %s:\n  %v",
					id, getIds(matches))
			}
			// In the resource, note that it is referenced
			// by the referrer.
			res.AppendRefBy(referrer.CurId())
			// Return transformed name of the object,
			// complete with prefixes, hashes, etc.
			return res.GetName(), res.GetNamespace(), nil
		}
	}

	return oldName, nil, nil
}

// utility function to replace a simple string by the new name
func (o *nameReferenceTransformer) getSimpleNameField(
	oldName string,
	referrer *resource.Resource,
	target resid.Gvk,
	referralCandidates resmap.ResMap,
	referralCandidateSubset []*resource.Resource) (interface{}, error) {

	newName, _, err := o.selectReferral(oldName, referrer, target,
		referralCandidates, referralCandidateSubset)

	return newName, err
}

// utility function to replace name field within a map[string]interface{}
// and leverage the namespace field.
func (o *nameReferenceTransformer) getNameAndNsStruct(
	inMap map[string]interface{},
	referrer *resource.Resource,
	target resid.Gvk,
	referralCandidates resmap.ResMap) (interface{}, error) {

	// Example:
	if _, ok := inMap["name"]; !ok {
		return nil, fmt.Errorf(
			"%#v is expected to contain a name field", inMap)
	}
	oldName, ok := inMap["name"].(string)
	if !ok {
		return nil, fmt.Errorf(
			"%#v is expected to contain a name field of type string", oldName)
	}

	subset := referralCandidates.Resources()
	if namespacevalue, ok := inMap["namespace"]; ok {
		namespace := namespacevalue.(string)
		bynamespace := referralCandidates.GroupedByOriginalNamespace()
		if _, ok := bynamespace[namespace]; !ok {
			return inMap, nil
		}
		subset = bynamespace[namespace]
	}

	newname, newnamespace, err := o.selectReferral(oldName, referrer, target,
		referralCandidates, subset)
	if err != nil {
		return nil, err
	}

	if (newname == oldName) && (newnamespace == nil) {
		// no candidate found.
		return inMap, nil
	}

	inMap["name"] = newname
	if newnamespace != "" {
		// We don't want value "" to replace value "default" since
		// the empty string is handled as a wild card here not default namespace
		// by kubernetes.
		inMap["namespace"] = newnamespace
	}
	return inMap, nil

}

func (o *nameReferenceTransformer) getNewNameFunc(
	referrer *resource.Resource,
	target resid.Gvk,
	referralCandidates resmap.ResMap) func(in interface{}) (interface{}, error) {
	return func(in interface{}) (interface{}, error) {
		switch in.(type) {
		case string:
			oldName, _ := in.(string)
			return o.getSimpleNameField(oldName, referrer, target,
				referralCandidates, referralCandidates.Resources())
		case map[string]interface{}:
			// Kind: ValidatingWebhookConfiguration
			// FieldSpec is webhooks/clientConfig/service
			oldMap, _ := in.(map[string]interface{})
			return o.getNameAndNsStruct(oldMap, referrer, target,
				referralCandidates)
		case []interface{}:
			l, _ := in.([]interface{})
			for idx, item := range l {
				switch item.(type) {
				case string:
					// Kind: Role/ClusterRole
					// FieldSpec is rules.resourceNames
					oldName, _ := item.(string)
					newName, err := o.getSimpleNameField(oldName, referrer, target,
						referralCandidates, referralCandidates.Resources())
					if err != nil {
						return nil, err
					}
					l[idx] = newName
				case map[string]interface{}:
					// Kind: RoleBinding/ClusterRoleBinding
					// FieldSpec is subjects
					// Note: The corresponding fieldSpec had been changed from
					// from path: subjects/name to just path: subjects. This is
					// what get mutatefield to request the mapping of the whole
					// map containing namespace and name instead of just a simple
					// string field containing the name
					oldMap, _ := item.(map[string]interface{})
					newMap, err := o.getNameAndNsStruct(oldMap, referrer, target,
						referralCandidates)
					if err != nil {
						return nil, err
					}
					l[idx] = newMap
				default:
					return nil, fmt.Errorf(
						"%#v is expected to be either a []string or a []map[string]interface{}", in)
				}
			}
			return in, nil
		default:
			return nil, fmt.Errorf(
				"%#v is expected to be either a string or a []interface{}", in)
		}
	}
}

func getIds(rs []*resource.Resource) []string {
	var result []string
	for _, r := range rs {
		result = append(result, r.CurId().String()+"\n")
	}
	return result
}
