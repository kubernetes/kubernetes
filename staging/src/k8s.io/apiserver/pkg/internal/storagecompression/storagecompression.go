/*
Copyright The Kubernetes Authors.

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

// Package storagecompression holds storage compression's validation, plus the
// pieces that validation, the config loader and the transformer must agree on.
// It's internal, so only k8s.io/apiserver/pkg/... can import it and none of it
// is public API. See KEP-6264.
package storagecompression

import (
	"fmt"
	"maps"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// MaxPlaintextBytes caps what's compressed and what a reader inflates. A value
// this size fits etcd's default 1.5 MiB request limit uncompressed, with 64 KiB
// spare. Never lower it; raise it only once every reader accepts more.
const MaxPlaintextBytes = 1536*1024 - 64*1024

// SecretsGroupResource returns core secrets, which no wildcard selects.
// Compressing a Secret tells anyone who can read etcd how compressible its
// plaintext is, so secrets are compressed only when a selector names them.
func SecretsGroupResource() schema.GroupResource {
	return schema.GroupResource{Resource: "secrets"}
}

// CoveringSelectors returns gr, '*.<group>' and '*.*' in that order, or only gr
// for secrets. Validation and the loader share it to agree on which entry wins.
func CoveringSelectors(gr schema.GroupResource) []schema.GroupResource {
	if gr == SecretsGroupResource() {
		return []schema.GroupResource{gr}
	}
	return []schema.GroupResource{gr, {Group: gr.Group, Resource: "*"}, {Group: "*", Resource: "*"}}
}

// compressionSelectorRules holds compression's error messages. Core secrets
// overlap nothing, because no wildcard selects them.
var compressionSelectorRules = selectorRules{
	bareWildcardErr:        "use '*.' to compress the core API group or '*.*' to compress every group; neither selects core secrets, which have to be named",
	nonRESTErr:             "resources which do not have REST API/s cannot be compressed",
	extensionsErr:          "'extensions' group has been removed and cannot be used for compression",
	acrossGroupsErr:        "compressing the same resource across groups is not supported",
	overlapErr:             "using overlapping resources such as 'configmaps' and '*.' in the same resource list is not allowed as they will be masked",
	notSelectedByWildcards: sets.New(SecretsGroupResource()),
}

// ValidateSelectors checks one entry's resources. It does the job of
// EncryptionConfiguration's validateResourceNames and validateResourceOverlap
// combined, in one pass: each selector must pass the name rules, and none may
// repeat or overlap an earlier one. It reports each problem once, on the later
// selector.
func ValidateSelectors(resources []string, fldPath *field.Path) field.ErrorList {
	return compressionSelectorRules.validate(resources, fldPath)
}

// ValidateMasking rejects a selector that repeats or falls under one in an
// earlier entry, since the first match wins and it could never take effect.
// Selectors that ValidateSelectors already reported aren't reported again but
// still cover the selectors after them. Selectors with a bad name are ignored.
func ValidateMasking(entries [][]string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	// decidedBy maps each selector seen so far to the earliest selector covering
	// it, which can be its own first occurrence. Masking errors name that path.
	decidedBy := map[schema.GroupResource]*field.Path{}
	for i, resources := range entries {
		entryPath := fldPath.Index(i).Child("resources")
		// flagged holds the paths ValidateSelectors reports in this entry, so
		// masking doesn't report them again.
		flagged := map[string]bool{}
		for _, err := range ValidateSelectors(resources, entryPath) {
			flagged[err.Field] = true
		}
		// inEntry joins decidedBy once this entry is done, since only an earlier
		// entry can mask a selector.
		inEntry := map[schema.GroupResource]*field.Path{}
		for j, res := range resources {
			p := entryPath.Index(j)
			// ValidateSelectors reports a bad name, so skip it here rather than report twice.
			if compressionSelectorRules.validateName(res, p) != nil {
				continue
			}
			gr := schema.ParseGroupResource(res)
			decider := p
			// Check gr against earlier entries, most specific selector first.
			for _, selector := range CoveringSelectors(gr) {
				by, ok := decidedBy[selector]
				if !ok {
					continue
				}
				decider = by
				switch {
				case flagged[p.String()]:
					// Its own entry already reported it.
				case selector == gr:
					allErrs = append(allErrs, field.Duplicate(p, res))
				default:
					allErrs = append(allErrs, field.Invalid(p, res, fmt.Sprintf("masked by %s", by)).WithOrigin("masked"))
				}
				break // Only the most specific match counts.
			}
			// A selector flagged in its own entry can sit under a wider one listed
			// earlier in that entry. Errors then name the wider one's root.
			if decider == p {
				for _, selector := range CoveringSelectors(gr) {
					if by, ok := inEntry[selector]; ok {
						decider = by
						break
					}
				}
			}
			inEntry[gr] = decider
		}
		maps.Copy(decidedBy, inEntry)
	}
	return allErrs
}
