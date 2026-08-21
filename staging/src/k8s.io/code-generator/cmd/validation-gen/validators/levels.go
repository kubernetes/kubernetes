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

package validators

import (
	"fmt"
	"regexp"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/codetags"
)

var kubeVersionRegex = regexp.MustCompile(`^\d+\.\d+$`)

const (
	alphaTagName = "k8s:alpha"
	betaTagName  = "k8s:beta"
)

func init() {
	RegisterTagValidator(&levelTagValidator{tagName: alphaTagName, level: ValidationStabilityLevelAlpha})
	RegisterTagValidator(&levelTagValidator{tagName: betaTagName, level: ValidationStabilityLevelBeta})
	RegisterEmissionFinalizer(FinalizeStabilityOrder, stabilityFinalizer{})
}

// stabilityFinalizer applies an EmittedGroup's StabilityLevel. Stability
// marking is owned here, next to the tags which declare stability levels.
type stabilityFinalizer struct{}

func (stabilityFinalizer) Name() string { return "stability" }

func (stabilityFinalizer) Finalize(_ Context, group EmittedGroup) (EmittedGroup, error) {
	if group.StabilityLevel == "" {
		return group, nil
	}
	group.Validations = wrapWithStabilityLevel(group.Validations, group.StabilityLevel)
	group.StabilityLevel = ""
	return group, nil
}

type levelTagValidator struct {
	extractor Extractor
	tagName   string
	level     ValidationStabilityLevel
}

func (ltv *levelTagValidator) Init(cfg Config) {
	ltv.extractor = cfg.Extractor
}

func (ltv *levelTagValidator) TagName() string {
	return ltv.tagName
}

var levelTagsValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (levelTagValidator) ValidScopes() sets.Set[Scope] {
	return levelTagsValidScopes
}

func (ltv *levelTagValidator) GetValidations(context Context, metadata SchemaMetadata, tag codetags.Tag) (EmittedGroup, error) {
	if tag.ValueType != codetags.ValueTypeTag || tag.ValueTag == nil {
		return EmittedGroup{}, fmt.Errorf("requires a validation tag as its value payload")
	}

	if len(tag.Args) > 1 {
		return EmittedGroup{}, fmt.Errorf("at most one optional kubernetes version argument is supported")
	}

	var version string
	if len(tag.Args) == 1 {
		arg := tag.Args[0]
		version = arg.Value
		if !kubeVersionRegex.MatchString(version) {
			return EmittedGroup{}, fmt.Errorf("invalid kubernetes version format, expected <major>.<minor>, got %s", version)
		}
	}

	context.StabilityLevel = ltv.level
	validations, err := ltv.extractor.ExtractTagValidations(context, metadata, *tag.ValueTag)
	if err != nil {
		return EmittedGroup{}, err
	}

	return EmittedGroup{
		Validations:    validations,
		StabilityLevel: ltv.level,
	}, nil
}

// wrapWithStabilityLevel applies a stability level to all functions in validations
// that are not self-managed, ensuring similarity between how type-level and field-level
// validations process stability levels.
func wrapWithStabilityLevel(validations Validations, level ValidationStabilityLevel) Validations {
	if level == "" {
		return validations
	}
	for i, fn := range validations.Functions {
		if !fn.StabilityLevelSelfManaged {
			fn.StabilityLevel = level
			validations.Functions[i] = fn
		}
	}
	for i, fn := range validations.TypeFunctions {
		if !fn.StabilityLevelSelfManaged {
			fn.StabilityLevel = level
			validations.TypeFunctions[i] = fn
		}
	}
	return validations
}

func (ltv *levelTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	if tag.ValueTag == nil {
		return SchemaMetadata{}, nil
	}
	context.StabilityLevel = ltv.level
	return ltv.extractor.ExtractMetadata(context, *tag.ValueTag)
}

func (ltv *levelTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            ltv.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(ltv.ValidScopes()),
		Description:    fmt.Sprintf("Marks the given payload validation as a %s validation of the handwritten validation code. An optional Kubernetes version can be specified.", ltv.level),
		Args: []TagArgDoc{{
			Description: "The Kubernetes version (e.g. `1.34`) at which this validation was added.",
			Type:        codetags.ArgTypeString,
			Name:        "since",
		}},
		Payloads: []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        fmt.Sprintf("The validation tag to evaluate as a %s validation.", ltv.level),
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
	return doc
}
