/*
Copyright 2021 The Kubernetes Authors.

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

package cel

import (
	"fmt"
	"strings"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/third_party/forked/celopenapi/model"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Validator parallels the structure of schema.Structural and includes the compiled CEL programs
// for the x-kubernetes-validations of each schema node.
type Validator struct {
	Items      *Validator
	Properties map[string]Validator

	AdditionalProperties *Validator

	compiledRules []CompilationResult

	// Program compilation is pre-checked at CRD creation/update time, so we don't expect compilation to fail
	// they are recompiled and added to this type, and it does, it is an internal bug.
	// But if somehow we get any compilation errors, we track them and then surface them as validation errors.
	compilationErr error

	// isResourceRoot is true if this validator node is for the root of a resource. Either the root of the
	// custom resource being validated, or the root of an XEmbeddedResource object.
	isResourceRoot bool
}

// NewValidator returns compiles all the CEL programs defined in x-kubernetes-validations extensions
// of the Structural schema and returns a custom resource validator that contains nested
// validators for all items, properties and additionalProperties that transitively contain validator rules.
// Returns nil only if there no validator rules in the Structural schema. May return a validator containing
// only errors.
func NewValidator(s *schema.Structural) *Validator {
	return validator(s, true)
}

func validator(s *schema.Structural, isResourceRoot bool) *Validator {
	compiledRules, err := Compile(s, isResourceRoot)
	var itemsValidator, additionalPropertiesValidator *Validator
	var propertiesValidators map[string]Validator
	if s.Items != nil {
		itemsValidator = validator(s.Items, s.Items.XEmbeddedResource)
	}
	if len(s.Properties) > 0 {
		propertiesValidators = make(map[string]Validator, len(s.Properties))
		for k, prop := range s.Properties {
			if p := validator(&prop, prop.XEmbeddedResource); p != nil {
				propertiesValidators[k] = *p
			}
		}
	}
	if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
		additionalPropertiesValidator = validator(s.AdditionalProperties.Structural, s.AdditionalProperties.Structural.XEmbeddedResource)
	}
	if len(compiledRules) > 0 || err != nil || itemsValidator != nil || additionalPropertiesValidator != nil || len(propertiesValidators) > 0 {
		return &Validator{
			compiledRules:        compiledRules,
			compilationErr:       err,
			isResourceRoot:       isResourceRoot,
			Items:                itemsValidator,
			AdditionalProperties: additionalPropertiesValidator,
			Properties:           propertiesValidators,
		}
	}

	return nil
}

// Validate validates all x-kubernetes-validations rules in Validator against obj and returns any errors.
func (s *Validator) Validate(fldPath *field.Path, sts *schema.Structural, obj interface{}) field.ErrorList {
	if s == nil || obj == nil {
		return nil
	}

	errs := s.validateExpressions(fldPath, sts, obj)
	switch obj := obj.(type) {
	case []interface{}:
		return append(errs, s.validateArray(fldPath, sts, obj)...)
	case map[string]interface{}:
		return append(errs, s.validateMap(fldPath, sts, obj)...)
	}
	return errs
}

func (s *Validator) validateExpressions(fldPath *field.Path, sts *schema.Structural, obj interface{}) (errs field.ErrorList) {
	if obj == nil {
		// We only validate non-null values. Rules that need to check for the state of a nullable value or the presence of an optional
		// field must do so from the surrounding schema. E.g. if an array has nullable string items, a rule on the array
		// schema can check if items are null, but a rule on the nullable string schema only validates the non-null strings.
		return nil
	}
	if s.compilationErr != nil {
		errs = append(errs, field.Invalid(fldPath, obj, fmt.Sprintf("rule compiler initialization error: %v", s.compilationErr)))
		return errs
	}
	if len(s.compiledRules) == 0 {
		return nil // nothing to do
	}
	if s.isResourceRoot {
		sts = model.WithTypeAndObjectMeta(sts)
	}
	activation := NewValidationActivation(obj, sts)
	for i, compiled := range s.compiledRules {
		rule := sts.XValidations[i]
		if compiled.Error != nil {
			errs = append(errs, field.Invalid(fldPath, obj, fmt.Sprintf("rule compile error: %v", compiled.Error)))
			continue
		}
		if compiled.Program == nil {
			// rule is empty
			continue
		}
		evalResult, _, err := compiled.Program.Eval(activation)
		if err != nil {
			// see types.Err for list of well defined error types
			if strings.HasPrefix(err.Error(), "no such overload") {
				// Most overload errors are caught by the compiler, which provides details on where exactly in the rule
				// error was found. Here, an overload error has occurred at runtime no details are provided, so we
				// append a more descriptive error message. This error can only occur when static type checking has
				// been bypassed. int-or-string is typed as dynamic and so bypasses compiler type checking.
				errs = append(errs, field.Invalid(fldPath, obj, fmt.Sprintf("'%v': call arguments did not match a supported operator, function or macro signature for rule: %v", err, ruleErrorString(rule))))
			} else {
				// no such key: {key}, index out of bounds: {index}, integer overflow, division by zero, ...
				errs = append(errs, field.Invalid(fldPath, obj, fmt.Sprintf("%v evaluating rule: %v", err, ruleErrorString(rule))))
			}
			continue
		}
		if evalResult != types.True {
			if len(rule.Message) != 0 {
				errs = append(errs, field.Invalid(fldPath, obj, rule.Message))
			} else {
				errs = append(errs, field.Invalid(fldPath, obj, fmt.Sprintf("failed rule: %s", ruleErrorString(rule))))
			}
		}
	}
	return errs
}

func ruleErrorString(rule apiextensions.ValidationRule) string {
	if len(rule.Message) > 0 {
		return strings.TrimSpace(rule.Message)
	}
	return strings.TrimSpace(rule.Rule)
}

type validationActivation struct {
	self ref.Val
}

func NewValidationActivation(obj interface{}, structural *schema.Structural) *validationActivation {
	return &validationActivation{self: UnstructuredToVal(obj, structural)}
}

func (a *validationActivation) ResolveName(name string) (interface{}, bool) {
	if name == ScopedVarName {
		return a.self, true
	}
	return nil, false
}

func (a *validationActivation) Parent() interpreter.Activation {
	return nil
}

func (s *Validator) validateMap(fldPath *field.Path, sts *schema.Structural, obj map[string]interface{}) (errs field.ErrorList) {
	if s == nil || obj == nil {
		return nil
	}

	if s.AdditionalProperties != nil && sts.AdditionalProperties != nil && sts.AdditionalProperties.Structural != nil {
		for k, v := range obj {
			errs = append(errs, s.AdditionalProperties.Validate(fldPath.Key(k), sts.AdditionalProperties.Structural, v)...)
		}
	}
	if s.Properties != nil && sts.Properties != nil {
		for k, v := range obj {
			stsProp, stsOk := sts.Properties[k]
			sub, ok := s.Properties[k]
			if ok && stsOk {
				errs = append(errs, sub.Validate(fldPath.Child(k), &stsProp, v)...)
			}
		}
	}

	return errs
}

func (s *Validator) validateArray(fldPath *field.Path, sts *schema.Structural, obj []interface{}) field.ErrorList {
	var errs field.ErrorList

	if s.Items != nil && sts.Items != nil {
		for i := range obj {
			errs = append(errs, s.Items.Validate(fldPath.Index(i), sts.Items, obj[i])...)
		}
	}

	return errs
}
