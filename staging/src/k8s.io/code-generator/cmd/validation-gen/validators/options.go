/*
Copyright 2024 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	ifEnabledTag  = "k8s:ifEnabled"
	ifDisabledTag = "k8s:ifDisabled"
)

func init() {
	RegisterTagValidator(&ifTagValidator{enabled: true})
	RegisterTagValidator(&ifTagValidator{enabled: false})
	RegisterEmissionFinalizer(FinalizeConditionsOrder, conditionsFinalizer{})
}

// conditionsFinalizer applies an EmittedGroup's Conditions. Condition
// wrapping is owned here, next to the tags which declare conditions.
type conditionsFinalizer struct{}

func (conditionsFinalizer) Name() string { return "conditions" }

func (conditionsFinalizer) Finalize(context Context, group EmittedGroup) (EmittedGroup, error) {
	if group.Conditions.Empty() {
		return group, nil
	}
	wrapContext := context
	if group.TargetType != nil {
		wrapContext.Type = group.TargetType
	}
	group.Validations = wrapWithConditions(group.Validations, group.Conditions, wrapContext)
	group.Conditions = nil
	return group, nil
}

type ifTagValidator struct {
	enabled   bool
	extractor Extractor
}

func (itv *ifTagValidator) Init(cfg Config) {
	itv.extractor = cfg.Extractor
}

func (itv ifTagValidator) TagName() string {
	if itv.enabled {
		return ifEnabledTag
	}
	return ifDisabledTag
}

var ifEnabledDisabledTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal, ScopeConst)

func (ifTagValidator) ValidScopes() sets.Set[Scope] {
	return ifEnabledDisabledTagValidScopes
}

var (
	ifOption = types.Name{Package: libValidationPkg, Name: "IfOption"}
)

func (itv ifTagValidator) GetValidations(context Context, metadata SchemaMetadata, tag codetags.Tag) (Validations, error) {
	optionArg, ok := tag.PositionalArg()
	if !ok {
		return Validations{}, fmt.Errorf("missing required option name positional argument")
	}

	if itv.enabled {
		context.Conditions = context.Conditions.WithOptionEnabled(optionArg.Value)
	} else {
		context.Conditions = context.Conditions.WithOptionDisabled(optionArg.Value)
	}

	validations, err := itv.extractor.ExtractTagValidations(context, metadata, *tag.ValueTag)
	if err != nil {
		return Validations{}, err
	}

	return wrapWithConditions(validations, context.Conditions, context), nil
}

// wrapWithConditions wraps validations with condition checks,
// ensuring similarity between how type-level and field-level validations process conditions.
func wrapWithConditions(validations Validations, cond Conditions, context Context) Validations {
	if cond.Empty() {
		return validations
	}
	res := validations
	for _, c := range cond {
		res = c.Wrap(res, context)
	}
	return res
}

func (itv ifTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	if tag.ValueTag == nil {
		return SchemaMetadata{}, nil
	}
	if optionArg, ok := tag.PositionalArg(); ok {
		if itv.enabled {
			context.Conditions = context.Conditions.WithOptionEnabled(optionArg.Value)
		} else {
			context.Conditions = context.Conditions.WithOptionDisabled(optionArg.Value)
		}
	}
	return itv.extractor.ExtractMetadata(context, *tag.ValueTag)
}

func (itv ifTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            itv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Args: []TagArgDoc{{
			Description: "<option>",
			Type:        codetags.ArgTypeString,
			Required:    true,
		}},
		Scopes: sets.List(itv.ValidScopes()),
	}

	doc.PayloadsType = codetags.ValueTypeTag
	doc.PayloadsRequired = true
	if itv.enabled {
		doc.Description = "Declares a validation that only applies when an option is enabled."
		doc.Payloads = []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "This validation tag will be evaluated only if the validation option is enabled.",
		}}
	} else {
		doc.Description = "Declares a validation that only applies when an option is disabled."
		doc.Payloads = []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "This validation tag will be evaluated only if the validation option is disabled.",
		}}
	}
	return doc
}
