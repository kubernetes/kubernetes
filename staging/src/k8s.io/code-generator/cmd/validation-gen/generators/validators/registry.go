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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

var registry = &Registry{}

type DeclarativeValidatorInit func(c *generator.Context) DeclarativeValidator

// AddToRegistry adds a DeclarativeValidator to the registry by providing the
// registry with an initializer it can use to construct a DeclarativeValidator for each
// generator context.
func AddToRegistry(validator DeclarativeValidatorInit) {
	registry.Add(validator)
}

type Registry struct {
	inits []DeclarativeValidatorInit
}

func (r *Registry) Add(validator DeclarativeValidatorInit) {
	r.inits = append(r.inits, validator)
}

func NewValidator(c *generator.Context, enabledTags, disabledTags []string) DeclarativeValidator {
	validators := make([]DeclarativeValidator, len(registry.inits))
	for i, init := range registry.inits {
		validators[i] = init(c)
	}
	return &compositeValidator{validators: validators, enabledTags: sets.New(enabledTags...), disabledTags: sets.New(disabledTags...)}
}

type compositeValidator struct {
	validators                []DeclarativeValidator
	enabledTags, disabledTags sets.Set[string]
}

func (c *compositeValidator) ExtractValidations(t *types.Type, comments []string) ([]FunctionGen, error) {
	var result []FunctionGen
	for _, v := range c.validators {
		validations, err := v.ExtractValidations(t, comments)
		if err != nil {
			return nil, err
		}
		for _, f := range validations {
			if c.allow(f.TagName()) {
				result = append(result, f)
			}
		}
	}
	return result, nil
}

func (c *compositeValidator) allow(tagName string) bool {
	if c.disabledTags.Has(tagName) {
		return false
	}

	return len(c.enabledTags) == 0 || c.enabledTags.Has(tagName)
}
