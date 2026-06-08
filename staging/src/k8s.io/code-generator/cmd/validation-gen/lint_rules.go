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
	"path"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
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

// hasRequirednessTag returns true if tags contain +k8s:optional, +k8s:required, or +k8s:forbidden.
func hasRequirednessTag(tags []codetags.Tag) bool {
	return util.HasTag(tags, "k8s:optional") || util.HasTag(tags, "k8s:required") || util.HasTag(tags, "k8s:forbidden")
}

// hasNonOpaqueValidationTag returns true if tags contain any registered validation tag that is not opaqueType.
func hasNonOpaqueValidationTag(extractor validators.ValidationExtractor, chainTags sets.Set[string], tags []codetags.Tag) bool {
	for _, tag := range tags {
		if tag.Name == "k8s:optional" || tag.Name == "k8s:opaqueType" {
			continue
		}
		if chainTags.Has(tag.Name) {
			if tag.ValueTag != nil && hasNonOpaqueValidationTag(extractor, chainTags, []codetags.Tag{*tag.ValueTag}) {
				return true
			}
			continue
		}
		// Check if it's a known validation tag.
		if extractor.IsKnownTag(tag.Name) {
			return true
		}
	}
	return false
}

// requiredAndOptional checks that fields (pointers, slices, maps, arrays) with validation
// (either direct or transitive) explicitly declare +k8s:optional or +k8s:required.
func requiredAndOptional(extractor validators.ValidationExtractor) lintRule {
	chainTags := sets.New[string]()
	for _, doc := range extractor.Docs() {
		if doc.PayloadsType == codetags.ValueTypeTag {
			chainTags.Insert(doc.Tag)
		}
	}

	type opacity struct {
		typ, key, val bool
	}

	filterTags := func(tags []codetags.Tag) []codetags.Tag {
		var filtered []codetags.Tag
		for _, tag := range tags {
			if extractor.IsKnownTag(tag.Name) {
				filtered = append(filtered, tag)
			}
		}
		return filtered
	}

	type cacheKey struct {
		t  *types.Type
		op opacity
	}
	hasValidation := make(map[cacheKey]*bool)

	// returns hasValidation, hasCycle, error
	var hasTransitiveValidation func(t *types.Type, op opacity) (bool, bool, error)
	hasTransitiveValidation = func(t *types.Type, op opacity) (bool, bool, error) {
		if op.typ {
			return false, false, nil
		}

		ck := cacheKey{t, op}
		visitedVal, visited := hasValidation[ck]
		if visited {
			if visitedVal == nil {
				return false, true, nil // cycle detected
			}
			return *visitedVal, false, nil
		}
		hasValidation[ck] = nil

		tTags, err := extractor.ExtractTags(validators.Context{Scope: validators.ScopeType, Type: t}, t.CommentLines)
		if err != nil {
			return false, false, err
		}

		typeVals, err := extractor.ExtractValidations(
			validators.Context{Scope: validators.ScopeType, Type: t},
			filterTags(tTags)...,
		)
		if err != nil {
			return false, false, err
		}

		op.typ = op.typ || typeVals.OpaqueType
		op.key = op.key || typeVals.OpaqueKeyType
		op.val = op.val || typeVals.OpaqueValType

		if op.typ {
			hasVal := false
			hasValidation[ck] = &hasVal
			return false, false, nil
		}

		if typeVals.HasEmitable() {
			hasVal := true
			hasValidation[ck] = &hasVal
			return true, false, nil
		}

		var hasVal, cycleBroken bool
		switch t.Kind {
		case types.Alias:
			hasVal, cycleBroken, err = hasTransitiveValidation(t.Underlying, op)
		case types.Slice, types.Array:
			hasVal, cycleBroken, err = hasTransitiveValidation(t.Elem, opacity{typ: op.val})
		case types.Map:
			kVal, cbKey, err := hasTransitiveValidation(t.Key, opacity{typ: op.key})
			if err != nil {
				return false, false, err
			}
			eVal, cbVal, err := hasTransitiveValidation(t.Elem, opacity{typ: op.val})
			if err != nil {
				return false, false, err
			}
			hasVal = kVal || eVal
			cycleBroken = cbKey || cbVal
		case types.Pointer:
			hasVal, cycleBroken, err = hasTransitiveValidation(t.Elem, op)
		case types.Struct:
			for _, m := range t.Members {
				mTags, err := extractor.ExtractTags(validators.Context{Scope: validators.ScopeField, Type: m.Type}, m.CommentLines)
				if err != nil {
					return false, false, err
				}
				if hasNonOpaqueValidationTag(extractor, chainTags, mTags) {
					hasVal = true
					break
				}
				fieldVals, err := extractor.ExtractValidations(
					validators.Context{Scope: validators.ScopeField, Type: m.Type},
					filterTags(mTags)...,
				)
				if err != nil {
					return false, false, err
				}
				mOp := opacity{
					typ: op.typ || fieldVals.OpaqueType,
					key: op.key || fieldVals.OpaqueKeyType,
					val: op.val || fieldVals.OpaqueValType,
				}
				hv, cb, err := hasTransitiveValidation(m.Type, mOp)
				if err != nil {
					return false, false, err
				}
				cycleBroken = cycleBroken || cb
				if hv {
					hasVal = true
					break
				}
			}
		}

		if err != nil {
			return false, false, err
		}

		if !cycleBroken {
			hasValidation[ck] = &hasVal
		} else {
			delete(hasValidation, ck)
		}
		return hasVal, cycleBroken, nil
	}

	return func(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
		// We only care about fields in a struct. Skip if linting the struct itself.
		if container == nil || container.Kind != types.Struct || container == t {
			return "", nil
		}

		// Skip non-pointer structs (and aliases to them) as they don't support requiredness tags.
		underlying := t
		for underlying.Kind == types.Alias {
			underlying = underlying.Underlying
		}
		if underlying.Kind == types.Struct {
			return "", nil
		}

		// Check if already has requiredness tag
		if hasRequirednessTag(tags) {
			return "", nil
		}

		// Check if it has validation (direct or active transitive)
		if hasNonOpaqueValidationTag(extractor, chainTags, tags) {
			return "field with validation must have +k8s:optional, +k8s:required or +k8s:forbidden", nil
		}

		fieldVals, err := extractor.ExtractValidations(
			validators.Context{Scope: validators.ScopeField, Type: t},
			filterTags(tags)...,
		)
		if err != nil {
			return fmt.Sprintf("invalid validation tags: %v", err), nil
		}

		topOp := opacity{
			typ: fieldVals.OpaqueType,
			key: fieldVals.OpaqueKeyType,
			val: fieldVals.OpaqueValType,
		}

		hasTransitiveVal, _, err := hasTransitiveValidation(t, topOp)
		if err != nil {
			return fmt.Sprintf("invalid validation tags: %v", err), nil
		}

		if hasTransitiveVal {
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
