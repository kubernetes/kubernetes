/*
Copyright 2023 The Kubernetes Authors.

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

package schemavalidation

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/warning"
)

// Resources that have opted into Declarative Validation will implement this
// interface for their strategies.
type DeclarativeValidationStrategy interface {
	// SetValidator sets the validator to be used by the strategy. The Validator
	// must be able to validate any gvk given to the strategy.
	//
	// During API installation the APIServer is expected to create the validator
	// from its OpenAPI configuration and provide it to any applicable strategies.
	SetValidator(Validator)
}

// Opts the given RESTCreateStrategy into Declarative Validation if the feature
// is enabled. Otherwise, returns the strategy unmodified.
func WrapCreateStrategyIfEnabled(strategy rest.RESTCreateStrategy) rest.RESTCreateStrategy {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DeclarativeValidation) {
		return strategy
	}
	return &schemaCreateStrategy{strategy, nil}
}

// Opts the given RESTUpdateStrategy into Declarative Validation if the feature
// is enabled. Otherwise, returns the strategy unmodified.
func WrapUpdateStrategyIfEnabled(strategy rest.RESTUpdateStrategy) rest.RESTUpdateStrategy {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DeclarativeValidation) {
		return strategy
	}
	return &schemaUpdateStrategy{strategy, nil}
}

type schemaCreateStrategy struct {
	rest.RESTCreateStrategy

	// Schema validator instantiated from the OpenAPI spec.
	validator Validator
}

func (s *schemaCreateStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	res := s.RESTCreateStrategy.Validate(ctx, obj)
	if s.validator != nil {
		declarativeResult := s.validator.Validate(ctx, obj)
		additions, deletions := diffNativeToDeclarativeErrors(res, declarativeResult)
		for _, v := range additions {
			warning.AddWarning(ctx, "DeclarativeValidation", fmt.Sprintf("Added Error: %v", v.Error()))
		}

		for _, v := range deletions {
			warning.AddWarning(ctx, "DeclarativeValidation", fmt.Sprintf("Deleted Error: %v", v.Error()))
		}
	}
	return res
}

func (s *schemaCreateStrategy) SetValidator(validator Validator) {
	s.validator = validator
}

type schemaUpdateStrategy struct {
	rest.RESTUpdateStrategy
	validator Validator
}

func (s *schemaUpdateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	res := s.RESTUpdateStrategy.ValidateUpdate(ctx, obj, old)
	if s.validator != nil {
		declarativeResult := s.validator.ValidateUpdate(ctx, obj, old)
		additions, deletions := diffNativeToDeclarativeErrors(res, declarativeResult)
		for _, v := range additions {
			warning.AddWarning(ctx, "DeclarativeValidation", fmt.Sprintf("Added Error: %v", v.Error()))
		}

		for _, v := range deletions {
			warning.AddWarning(ctx, "DeclarativeValidation", fmt.Sprintf("Deleted Error: %v", v.Error()))
		}
	}
	return res
}

func (s *schemaUpdateStrategy) SetValidator(validator Validator) {
	s.validator = validator
}
