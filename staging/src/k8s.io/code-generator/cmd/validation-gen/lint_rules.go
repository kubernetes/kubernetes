/*
Copyright 2025 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"path"
	"strings"

	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

func checkAlphaBetaUsage(tag codetags.Tag, isRoot bool) (string, error) {
	if tag.Name == "k8s:alpha" || tag.Name == "k8s:beta" {
		if !isRoot {
			return fmt.Sprintf("tag %q can't be used in between", tag.Name), nil
		}
		if tag.ValueTag == nil {
			return fmt.Sprintf("tag %q requires a validation tag as its value payload", tag.Name), nil
		}
	}

	if tag.ValueTag != nil {
		return checkAlphaBetaUsage(*tag.ValueTag, false)
	}
	return "", nil
}

// alphaBetaPrefix enforces that +k8s:alpha and +k8s:beta tags are always used as prefix to
func alphaBetaPrefix() lintRule {
	return func(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
		for _, tag := range tags {
			// Only check alpha/beta tags or validation tags.
			if msg, err := checkAlphaBetaUsage(tag, true); err != nil || msg != "" {
				return msg, err
			}
		}
		return "", nil
	}
}

// checkTagStability recursively checks that a tag and its nested tags
// satisfy the stability requirements of the context.
func checkTagStability(tag codetags.Tag, contextLevel validators.TagStabilityLevel) (string, error) {
	tagStability, err := validators.GetStability(tag.Name)
	// all DV tags have stability, if a tag doesn't have stability then it is not a valid DV tag.
	if err != nil {
		return "", nil
	}
	cmpOrder, err := tagStability.Compare(contextLevel)
	if err != nil {
		return "", err
	}
	if cmpOrder < 0 {
		return fmt.Sprintf("tag %q with stability level %q cannot be used in %s validation", tag.Name, tagStability, contextLevel), nil
	}
	if tag.ValueTag != nil {
		return checkTagStability(*tag.ValueTag, contextLevel)
	}
	return "", nil
}

// validationStability enforces stability level constraints on tags.
func validationStability() lintRule {
	return func(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
		pkgPath := t.Name.Package
		if container != nil {
			pkgPath = container.Name.Package
		}

		// Unprefixed validations are normally required to be Stable.
		// In Alpha packages, we allow Alpha-level and Beta-level validations
		// without a prefix. In Beta packages, we allow Beta-level validations
		// without a prefix.
		defaultContextLevel := validators.TagStabilityLevelStable
		// APIVersion is the last element of the package path.
		apiVersion := path.Base(pkgPath)
		if strings.Contains(apiVersion, "alpha") {
			defaultContextLevel = validators.TagStabilityLevelAlpha
		} else if strings.Contains(apiVersion, "beta") {
			defaultContextLevel = validators.TagStabilityLevelBeta
		}

		for _, tag := range tags {
			contextLevel := defaultContextLevel
			tagToCheck := tag

			// For stability level tags, set the stability context for the inner validation,
			// overriding the package-level default.
			if tag.Name == "k8s:alpha" || tag.Name == "k8s:beta" {
				if tag.Name == "k8s:alpha" {
					contextLevel = validators.TagStabilityLevelAlpha
				} else {
					contextLevel = validators.TagStabilityLevelBeta
				}
				if tag.ValueTag == nil {
					continue
				}
				tagToCheck = *tag.ValueTag
			}

			// For feature gate tags, we allow developers to use nested beta validation tags
			// without forcing validation authors to write redundant handwritten code (bypassing the
			// declarative validation equivalence check), we automatically relax the stability
			// context to Beta if the current context is Stable.
			if tagToCheck.Name == "k8s:ifEnabled" || tagToCheck.Name == "k8s:ifDisabled" {
				if contextLevel == validators.TagStabilityLevelStable {
					contextLevel = validators.TagStabilityLevelBeta
				}
			}

			msg, err := checkTagStability(tagToCheck, contextLevel)
			if err != nil {
				return "", err
			}
			if msg != "" {
				return msg, nil
			}
		}
		return "", nil
	}
}

// hasTag recursively checks if a tag with given name exists in the tag tree.
func hasTag(tags []codetags.Tag, name string) bool {
	for _, tag := range tags {
		if tag.Name == name {
			return true
		}
		// Also check conditional tags value
		if tag.ValueTag != nil && hasTag([]codetags.Tag{*tag.ValueTag}, name) {
			return true
		}
	}
	return false
}

// hasRequirednessTag returns true if tags contain +k8s:optional, +k8s:required, or +k8s:forbidden.
func hasRequirednessTag(tags []codetags.Tag) bool {
	return hasTag(tags, "k8s:optional") || hasTag(tags, "k8s:required") || hasTag(tags, "k8s:forbidden")
}

// hasNonOpaqueValidationTag returns true if tags contain any registered validation tag that is not opaqueType.
func hasNonOpaqueValidationTag(tags []codetags.Tag) bool {
	for _, tag := range tags {
		switch tag.Name {
		case "k8s:optional", "k8s:opaqueType":
			continue
		case "k8s:alpha", "k8s:beta", "k8s:eachVal", "k8s:eachKey", "k8s:ifEnabled", "k8s:ifDisabled", "k8s:subfield", "k8s:ifMode":
			if tag.ValueTag != nil && hasNonOpaqueValidationTag([]codetags.Tag{*tag.ValueTag}) {
				return true
			}
			continue
		}
		// Check if it's a known validation tag.
		if _, err := validators.GetStability(tag.Name); err == nil {
			return true
		}
	}
	return false
}

// requiredAndOptional checks that fields (pointers, slices, maps, arrays) with validation
// (either direct or transitive) explicitly declare +k8s:optional or +k8s:required.
func requiredAndOptional(extractor validators.ValidationExtractor) lintRule {
	// Cache for transitive validation check.
	// Tri-state: key absent = unvisited, value nil = in-progress (cycle), value *bool = computed result.
	hasValidation := make(map[*types.Type]*bool)

	var hasTransitiveValidation func(t *types.Type, tags []codetags.Tag) (bool, bool, error)

	hasValidationAfterOpacity := func(t *types.Type, scope validators.Scope, tags []codetags.Tag) (bool, error) {
		if len(tags) == 0 {
			return true, nil
		}

		var valTags []codetags.Tag
		for _, tag := range tags {
			if _, err := validators.GetStability(tag.Name); err == nil {
				valTags = append(valTags, tag)
			}
		}

		vals, err := extractor.ExtractValidations(validators.Context{Scope: scope, Type: t}, valTags...)
		if err != nil {
			return false, err
		}

		nativeT := util.NativeType(t)
		if vals.OpaqueType {
			return false, nil
		}
		if nativeT.Kind == types.Map {
			kVal, _, err := hasTransitiveValidation(nativeT.Key, nil)
			if err != nil {
				return false, err
			}
			eVal, _, err := hasTransitiveValidation(nativeT.Elem, nil)
			if err != nil {
				return false, err
			}
			hasKeyVal := kVal && !vals.OpaqueKeyType
			hasElemVal := eVal && !vals.OpaqueValType
			return hasKeyVal || hasElemVal, nil
		}
		if nativeT.Kind == types.Slice || nativeT.Kind == types.Array {
			return !vals.OpaqueValType, nil
		}
		return true, nil
	}

	// hasTransitiveValidation returns (hasVal, cycleBroken, err).
	// If cycleBroken is true, we cannot cache a negative result.
	hasTransitiveValidation = func(t *types.Type, tags []codetags.Tag) (bool, bool, error) {
		if hasNonOpaqueValidationTag(tags) {
			return true, false, nil
		}

		var hasVal, cycleBroken bool

		if val, ok := hasValidation[t]; ok {
			if val != nil {
				hasVal = *val
			} else {
				cycleBroken = true
			}
		} else {
			hasValidation[t] = nil // Mark in-progress
			tTags, err := extractor.ExtractTags(validators.Context{Scope: validators.ScopeType, Type: t}, t.CommentLines)
			if err != nil {
				return false, false, err
			}
			hasVal = hasNonOpaqueValidationTag(tTags)

			if !hasVal {
				switch t.Kind {
				case types.Alias:
					var err error
					hasVal, cycleBroken, err = hasTransitiveValidation(t.Underlying, nil)
					if err != nil {
						return false, false, err
					}
				case types.Slice, types.Array, types.Pointer:
					var err error
					hasVal, cycleBroken, err = hasTransitiveValidation(t.Elem, nil)
					if err != nil {
						return false, false, err
					}
				case types.Map:
					hvKey, cbKey, err := hasTransitiveValidation(t.Key, nil)
					if err != nil {
						return false, false, err
					}
					if hvKey {
						hasVal, cycleBroken = true, cbKey
					} else {
						hvElem, cbElem, err := hasTransitiveValidation(t.Elem, nil)
						if err != nil {
							return false, false, err
						}
						hasVal, cycleBroken = hvElem, cbKey || cbElem
					}
				case types.Struct:
					for _, m := range t.Members {
						mTags, err := extractor.ExtractTags(validators.Context{Scope: validators.ScopeField, Type: m.Type}, m.CommentLines)
						if err != nil {
							return false, false, err
						}
						hv, cb, err := hasTransitiveValidation(m.Type, mTags)
						if err != nil {
							return false, false, err
						}
						if hv {
							hasVal = true
						}
						if cb {
							cycleBroken = true
						}
						if hasVal {
							break
						}
					}
				}
			}

			// Type-level opacity overrides structural validation.
			if hasVal {
				var err error
				hasVal, err = hasValidationAfterOpacity(t, validators.ScopeType, tTags)
				if err != nil {
					return false, false, err
				}
			}

			if hasVal || !cycleBroken {
				hasValidation[t] = &hasVal
			} else {
				// Do not cache false if a cycle was broken, as validation could exist on another path.
				delete(hasValidation, t)
			}
		}

		// Field-level opacity overrides structural validation.
		if hasVal {
			var err error
			hasVal, err = hasValidationAfterOpacity(t, validators.ScopeField, tags)
			if err != nil {
				return false, false, err
			}
		}

		return hasVal, cycleBroken, nil
	}

	return func(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
		// We only care about fields in a struct. Skip if linting the struct itself.
		if container == nil || container.Kind != types.Struct || container == t {
			return "", nil
		}

		// Skip non-pointer structs (and aliases to them) as they don't support requiredness tags.
		if util.NativeType(t).Kind == types.Struct {
			return "", nil
		}

		// Check if already has requiredness tag
		if hasRequirednessTag(tags) {
			return "", nil
		}

		// Check if it has validation (direct or active transitive)
		hasDirectVal := hasNonOpaqueValidationTag(tags)
		hasTransitiveVal, _, err := hasTransitiveValidation(t, tags)
		if err != nil {
			return fmt.Sprintf("invalid validation tags: %v", err), nil
		}

		if hasDirectVal || hasTransitiveVal {
			return "field with validation must have +k8s:optional, +k8s:required or +k8s:forbidden", nil
		}

		return "", nil
	}
}

func lintRules(extractor validators.ValidationExtractor) []lintRule {
	return []lintRule{
		alphaBetaPrefix(),
		validationStability(),
		requiredAndOptional(extractor),
	}
}
