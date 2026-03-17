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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	featureGateTagName = "k8s:featureGate"
)

func init() {
	RegisterTagValidator(&featureGateTagValidator{})
}

// featureGateTagValidator handles +k8s:featureGate=GateName.
// It returns a GateCheck which tells the emitter to wrap the field's
// validations in a gate-enabled check and emit a forbidden fallback
// when the gate is disabled.
type featureGateTagValidator struct{}

func (*featureGateTagValidator) Init(_ Config) {}

func (*featureGateTagValidator) TagName() string {
	return featureGateTagName
}

var featureGateTagValidScopes = sets.New(ScopeField)

func (*featureGateTagValidator) ValidScopes() sets.Set[Scope] {
	return featureGateTagValidScopes
}

func (fgtv *featureGateTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	if tag.Value == "" {
		return Validations{}, fmt.Errorf("missing required feature gate name")
	}

	forbiddenFns, err := forbiddenFunctionsForType(context.Type)
	if err != nil {
		return Validations{}, fmt.Errorf("tag %q: %w", featureGateTagName, err)
	}

	return Validations{
		Conditions: &Conditions{
			OptionsEnabled: []string{tag.Value},
		},
		FallbackFunctions: forbiddenFns,
	}, nil
}

// forbiddenFunctionsForType returns the forbidden + optional short-circuit
// function pair for the given type, mirroring requirednessTagValidator.doForbidden().
func forbiddenFunctionsForType(t *types.Type) ([]FunctionGen, error) {
	var forbidden, optional types.Name
	switch util.NativeType(t).Kind {
	case types.Slice:
		forbidden = forbiddenSliceValidator
		optional = optionalSliceValidator
	case types.Map:
		forbidden = forbiddenMapValidator
		optional = optionalMapValidator
	case types.Pointer:
		forbidden = forbiddenPointerValidator
		optional = optionalPointerValidator
	case types.Struct:
		return nil, fmt.Errorf("non-pointer structs cannot use the %q tag", featureGateTagName)
	default:
		forbidden = forbiddenValueValidator
		optional = optionalValueValidator
	}

	return []FunctionGen{
		Function(featureGateTagName, ShortCircuit, forbidden),
		Function(featureGateTagName, ShortCircuit|NonError, optional),
	}, nil
}

func (fgtv *featureGateTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            fgtv.TagName(),
		StabilityLevel: TagStabilityLevelAlpha,
		Scopes:         sets.List(fgtv.ValidScopes()),
		Description:    "Declares that a field is gated behind a feature gate. When the gate is disabled, the field is forbidden. When enabled, normal validation applies.",
		Payloads: []TagPayloadDoc{{
			Description: "<gate-name>",
			Docs:        "The name of the feature gate. Use multiple +k8s:featureGate tags for multiple gates; all must be enabled for the field to be allowed.",
		}},
		PayloadsType:     codetags.ValueTypeString,
		PayloadsRequired: true,
	}
}
