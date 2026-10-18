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

package validation

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/storage/value/compression"
)

const (
	compressionConfigNilErr       = "StorageCompressionConfiguration can't be nil"
	compressionExtensionsGroupErr = "'extensions' group has been removed and cannot be used for compression"
	compressionStarResourceErr    = "use '*.' to compress the core API group or '*.*' to compress every group; neither selects secrets, which have to be named"
	compressionOverlapErr         = "using overlapping resources such as 'configmaps' and '*.' in the same resource list is not allowed as they will be masked"
	compressionNonRESTAPIErr      = "resources which do not have REST API/s cannot be compressed"
	compressionAcrossGroupErr     = "compressing the same resource across groups is not supported"
	compressionMinSizeFractionErr = "must be a whole number of bytes"
	compressionMaskedErrFmt       = "masked by earlier rule %q"
)

var compressionMinSizeRangeErr = fmt.Sprintf("must be at least %d and less than %s",
	compression.MinConfigurableMinSize, resource.NewQuantity(compression.MaxPlaintextBytes, resource.BinarySI))

// compressionResourceRules gives the shared resource checks their compression
// error messages. Secrets can sit beside a wildcard because no wildcard selects them.
var compressionResourceRules = resourceRules{
	starResourceErr:        compressionStarResourceErr,
	overlapErr:             compressionOverlapErr,
	nonRESTAPIResourceErr:  compressionNonRESTAPIErr,
	extensionsGroupErr:     compressionExtensionsGroupErr,
	acrossGroupErr:         compressionAcrossGroupErr,
	notSelectedByWildcards: sets.New(compression.SecretsGroupResource),
}

var supportedCompressionAlgorithms = []apiserver.CompressionAlgorithm{
	apiserver.CompressionAlgorithmDeflate,
	apiserver.CompressionAlgorithmNone,
}

// ValidateStorageCompressionConfiguration checks each entry's resources as
// ValidateEncryptionConfiguration does, plus its algorithm and minSize. It also
// rejects any selector that an earlier entry masks. The encryption loader does
// that only for wildcards; this also rejects a resource repeated in a later entry.
func ValidateStorageCompressionConfiguration(c *apiserver.StorageCompressionConfiguration) field.ErrorList {
	root := field.NewPath("resources")
	allErrs := field.ErrorList{}

	if c == nil {
		allErrs = append(allErrs, field.Required(root, compressionConfigNilErr))
		return allErrs
	}

	for i, conf := range c.Resources {
		r := root.Index(i).Child("resources")

		if len(conf.Resources) == 0 {
			allErrs = append(allErrs, field.Required(r, fmt.Sprintf(atLeastOneRequiredErrFmt, r)))
		}

		allErrs = append(allErrs, validateResourceOverlap(conf.Resources, r, compressionResourceRules)...)
		allErrs = append(allErrs, validateResourceNames(conf.Resources, r, compressionResourceRules)...)
		allErrs = append(allErrs, validateCompressionAlgorithm(conf.Algorithm, root.Index(i).Child("algorithm"))...)
		allErrs = append(allErrs, validateCompressionMinSize(conf.MinSize, root.Index(i).Child("minSize"))...)
	}
	allErrs = append(allErrs, validateCompressionMasking(c.Resources, root)...)

	return allErrs
}

// validateCompressionMasking compares each selector only with earlier entries,
// because validateResourceOverlap already covers overlap within one entry. The
// first match wins, so a masked selector could never take effect. Selectors the
// name checks reject are skipped, so one mistake isn't reported twice.
func validateCompressionMasking(entries []apiserver.CompressionResourceConfiguration, root *field.Path) field.ErrorList {
	valid := func(res string) bool {
		return len(validateResourceNames([]string{res}, nil, compressionResourceRules)) == 0
	}
	var allErrs field.ErrorList
	earlier := map[schema.GroupResource]string{}
	for i, entry := range entries {
		for j, res := range entry.Resources {
			if !valid(res) {
				continue
			}
			for _, selector := range compression.CoveringSelectors(schema.ParseGroupResource(res)) {
				if by, ok := earlier[selector]; ok {
					allErrs = append(allErrs, field.Invalid(root.Index(i).Child("resources").Index(j), res, fmt.Sprintf(compressionMaskedErrFmt, by)))
					break
				}
			}
		}
		for _, res := range entry.Resources {
			if valid(res) {
				earlier[schema.ParseGroupResource(res)] = res
			}
		}
	}
	return allErrs
}

func validateCompressionAlgorithm(algorithm apiserver.CompressionAlgorithm, fieldPath *field.Path) field.ErrorList {
	switch algorithm {
	case apiserver.CompressionAlgorithmDeflate, apiserver.CompressionAlgorithmNone:
		return nil
	case "":
		return field.ErrorList{field.Required(fieldPath, "")}
	default:
		return field.ErrorList{field.NotSupported(fieldPath, algorithm, supportedCompressionAlgorithms)}
	}
}

// validateCompressionMinSize checks the range first. Value can't represent a
// quantity beyond int64, so the whole-number test would misreport it.
func validateCompressionMinSize(minSize *resource.Quantity, fieldPath *field.Path) field.ErrorList {
	if minSize == nil {
		return nil
	}
	if minSize.Cmp(*resource.NewQuantity(compression.MinConfigurableMinSize, resource.BinarySI)) < 0 ||
		minSize.Cmp(*resource.NewQuantity(compression.MaxPlaintextBytes, resource.BinarySI)) >= 0 {
		return field.ErrorList{field.Invalid(fieldPath, minSize.String(), compressionMinSizeRangeErr)}
	}
	if minSize.Cmp(*resource.NewQuantity(minSize.Value(), resource.BinarySI)) != 0 {
		return field.ErrorList{field.Invalid(fieldPath, minSize.String(), compressionMinSizeFractionErr)}
	}
	return nil
}
