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
	if contextLevel == validators.TagStabilityLevelAlpha {
		return "", nil
	}

	stability, err := validators.GetStability(tag.Name)
	if err == nil {
		if contextLevel == validators.TagStabilityLevelBeta {
			if stability != validators.TagStabilityLevelStable && stability != validators.TagStabilityLevelBeta {
				return fmt.Sprintf("tag %q with stability level %q cannot be used in %s validation", tag.Name, stability, contextLevel), nil
			}
		} else if stability != validators.TagStabilityLevelStable {
			return fmt.Sprintf("tag %q with stability level %q cannot be used in %s validation", tag.Name, stability, contextLevel), nil
		}
	}

	if tag.ValueTag != nil {
		return checkTagStability(*tag.ValueTag, contextLevel)
	}
	return "", nil
}

// validationStability enforces stability level constraints on tags.
func validationStability() lintRule {
	return func(container *types.Type, t *types.Type, tags []codetags.Tag) (string, error) {
		for _, tag := range tags {
			contextLevel := validators.TagStabilityLevelStable
			switch tag.Name {
			case "k8s:alpha":
				contextLevel = validators.TagStabilityLevelAlpha
			case "k8s:beta":
				contextLevel = validators.TagStabilityLevelBeta
			default:
				// Normal tag (implicit Stable context).
				// Check the root tag itself.
				stability, err := validators.GetStability(tag.Name)
				if err == nil && stability != validators.TagStabilityLevelStable {
					return fmt.Sprintf("tag %q with stability level %q cannot be used in Stable validation", tag.Name, stability), nil
				}
			}

			// Recursively check content.
			if tag.ValueTag != nil {
				msg, err := checkTagStability(*tag.ValueTag, contextLevel)
				if err != nil {
					return "", err
				}
				if msg != "" {
					return msg, nil
				}
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

// hasAnyValidationTag returns true if tags contain any registered validation tag.
func hasAnyValidationTag(tags []codetags.Tag) bool {
	for _, tag := range tags {
		switch tag.Name {
		case "k8s:optional":
			continue
		case "k8s:alpha", "k8s:beta":
			if tag.ValueTag != nil && hasAnyValidationTag([]codetags.Tag{*tag.ValueTag}) {
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

	// checkType recursively checks if a type has any validation (transitively).
	var checkType func(t *types.Type) (bool, bool)
	checkType = func(t *types.Type) (bool, bool) {
		if val, ok := hasValidation[t]; ok {
			if val == nil {
				return false, true // Cycle detected, break conservatively
			}
			return *val, false
		}
		// Mark in-progress
		hasValidation[t] = nil
		extractedTags, err := extractor.ExtractTags(validators.Context{}, t.CommentLines)
		hasVal := err == nil && hasAnyValidationTag(extractedTags)
		cycleBroken := false

		switch t.Kind {
		case types.Alias:
			if hv, cb := checkType(t.Underlying); hv {
				hasVal = true
			} else if cb {
				cycleBroken = true
			}
		case types.Struct:
			for _, member := range t.Members {
				memberTags, err := extractor.ExtractTags(validators.Context{}, member.CommentLines)
				memberHasVal := err == nil && hasAnyValidationTag(memberTags)
				if hv, cb := checkType(member.Type); hv {
					memberHasVal = true
				} else if cb {
					cycleBroken = true
				}
				if memberHasVal {
					hasVal = true
				}
			}
		case types.Slice, types.Array, types.Pointer:
			if hv, cb := checkType(t.Elem); hv {
				hasVal = true
			} else if cb {
				cycleBroken = true
			}
		case types.Map:
			if hv, cb := checkType(t.Key); hv {
				hasVal = true
			} else if cb {
				cycleBroken = true
			}
			if hv, cb := checkType(t.Elem); hv {
				hasVal = true
			} else if cb {
				cycleBroken = true
			}
		}

		if hasVal || !cycleBroken {
			hasValidation[t] = &hasVal
		} else {
			delete(hasValidation, t)
		}
		return hasVal, cycleBroken
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

		// Check if it has validation (direct or transitive)
		hasDirectVal := hasAnyValidationTag(tags)
		hasTransitiveVal, _ := checkType(t)

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
