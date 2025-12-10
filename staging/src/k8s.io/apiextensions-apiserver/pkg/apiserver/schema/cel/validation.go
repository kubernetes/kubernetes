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
	"bufio"
	"context"
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strings"
	"time"

	celgo "github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/model"
	"k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/metrics"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"

	celconfig "k8s.io/apiserver/pkg/apis/cel"
)

// Validator parallels the structure of schema.Structural and includes the compiled CEL programs
// for the x-kubernetes-validations of each schema node.
type Validator struct {
	Items                *Validator
	Properties           map[string]Validator
	AllOfValidators      []*Validator
	AdditionalProperties *Validator

	Schema *schema.Structural

	uncompiledRules []apiextensions.ValidationRule
	compiledRules   []CompilationResult

	// Program compilation is pre-checked at CRD creation/update time, so we don't expect compilation to fail
	// they are recompiled and added to this type, and it does, it is an internal bug.
	// But if somehow we get any compilation errors, we track them and then surface them as validation errors.
	compilationErr error

	// isResourceRoot is true if this validator node is for the root of a resource. Either the root of the
	// custom resource being validated, or the root of an XEmbeddedResource object.
	isResourceRoot bool

	// celActivationFactory produces a Activations, which resolve identifiers
	// (e.g. self and oldSelf) to CEL values. One activation must be produced
	// for each of the cases when oldSelf is optional and non-optional.
	celActivationFactory func(sts *schema.Structural, obj, oldObj interface{}) (activation interpreter.Activation, optionalOldSelfActivation interpreter.Activation)
}

// NewValidator returns compiles all the CEL programs defined in x-kubernetes-validations extensions
// of the Structural schema and returns a custom resource validator that contains nested
// validators for all items, properties and additionalProperties that transitively contain validator rules.
// Returns nil if there are no validator rules in the Structural schema. May return a validator containing only errors.
// Adding perCallLimit as input arg for testing purpose only. Callers should always use const PerCallLimit from k8s.io/apiserver/pkg/apis/cel/config.go as input
func NewValidator(s *schema.Structural, isResourceRoot bool, perCallLimit uint64) *Validator {
	if !hasXValidations(s) {
		return nil
	}
	return validator(s, s, isResourceRoot, model.SchemaDeclType(s, isResourceRoot), perCallLimit)
}

// validator creates a Validator for all x-kubernetes-validations at the level of the provided schema and lower and
// returns the Validator if any x-kubernetes-validations exist in the schema, or nil if no x-kubernetes-validations
// exist. declType is expected to be a CEL DeclType corresponding to the structural schema.
// perCallLimit was added for testing purpose only. Callers should always use const PerCallLimit from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func validator(validationSchema, nodeSchema *schema.Structural, isResourceRoot bool, declType *cel.DeclType, perCallLimit uint64) *Validator {
	compilationSchema := *nodeSchema
	compilationSchema.XValidations = validationSchema.XValidations
	compiledRules, err := Compile(&compilationSchema, declType, perCallLimit, environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()), StoredExpressionsEnvLoader())

	var itemsValidator, additionalPropertiesValidator *Validator
	var propertiesValidators map[string]Validator
	var allOfValidators []*Validator
	var elemType *cel.DeclType
	if declType != nil {
		elemType = declType.ElemType
	} else {
		elemType = declType
	}

	if validationSchema.Items != nil && nodeSchema.Items != nil {
		itemsValidator = validator(validationSchema.Items, nodeSchema.Items, nodeSchema.Items.XEmbeddedResource, elemType, perCallLimit)
	}

	if len(validationSchema.Properties) > 0 {
		propertiesValidators = make(map[string]Validator, len(validationSchema.Properties))
		for k, validationProperty := range validationSchema.Properties {
			nodeProperty, ok := nodeSchema.Properties[k]
			if !ok {
				// Can only add value validations for fields that are on the
				// structural spine of the schema.
				continue
			}

			var fieldType *cel.DeclType
			if escapedPropName, ok := cel.Escape(k); ok {
				if declType == nil {
					continue
				}
				if f, ok := declType.Fields[escapedPropName]; ok {
					fieldType = f.Type
				} else {
					// fields with unknown types are omitted from CEL validation entirely
					continue
				}
			} else {
				// field may be absent from declType if the property name is unescapable, in which case we should convert
				// the field value type to a DeclType.
				fieldType = model.SchemaDeclType(&nodeProperty, nodeProperty.XEmbeddedResource)
				if fieldType == nil {
					continue
				}
			}
			if p := validator(&validationProperty, &nodeProperty, nodeProperty.XEmbeddedResource, fieldType, perCallLimit); p != nil {
				propertiesValidators[k] = *p
			}
		}
	}
	if validationSchema.AdditionalProperties != nil && validationSchema.AdditionalProperties.Structural != nil &&
		nodeSchema.AdditionalProperties != nil && nodeSchema.AdditionalProperties.Structural != nil {
		additionalPropertiesValidator = validator(validationSchema.AdditionalProperties.Structural, nodeSchema.AdditionalProperties.Structural, nodeSchema.AdditionalProperties.Structural.XEmbeddedResource, elemType, perCallLimit)
	}

	if validationSchema.ValueValidation != nil && len(validationSchema.ValueValidation.AllOf) > 0 {
		allOfValidators = make([]*Validator, 0, len(validationSchema.ValueValidation.AllOf))
		for _, allOf := range validationSchema.ValueValidation.AllOf {
			allOfValidator := validator(nestedToStructural(&allOf), nodeSchema, isResourceRoot, declType, perCallLimit)
			if allOfValidator != nil {
				allOfValidators = append(allOfValidators, allOfValidator)
			}
		}
	}

	if len(compiledRules) > 0 || err != nil || itemsValidator != nil || additionalPropertiesValidator != nil || len(propertiesValidators) > 0 || len(allOfValidators) > 0 {
		activationFactory := validationActivationWithoutOldSelf
		for _, rule := range compiledRules {
			if rule.UsesOldSelf {
				activationFactory = validationActivationWithOldSelf
				break
			}
		}

		return &Validator{
			compiledRules:        compiledRules,
			uncompiledRules:      validationSchema.XValidations,
			compilationErr:       err,
			isResourceRoot:       isResourceRoot,
			Items:                itemsValidator,
			AdditionalProperties: additionalPropertiesValidator,
			Properties:           propertiesValidators,
			AllOfValidators:      allOfValidators,
			celActivationFactory: activationFactory,
			Schema:               nodeSchema,
		}
	}

	return nil
}

type options struct {
	ratchetingOptions
}

type Option func(*options)

func WithRatcheting(correlation *common.CorrelatedObject) Option {
	return func(o *options) {
		o.currentCorrelation = correlation
	}
}

// Validate validates all x-kubernetes-validations rules in Validator against obj and returns any errors.
// If the validation rules exceed the costBudget, subsequent evaluations will be skipped, the list of errs returned will not be empty, and a negative remainingBudget will be returned.
// Most callers can ignore the returned remainingBudget value unless another validate call is going to be made
// context is passed for supporting context cancellation during cel validation
func (s *Validator) Validate(ctx context.Context, fldPath *field.Path, _ *schema.Structural, obj, oldObj interface{}, costBudget int64, opts ...Option) (errs field.ErrorList, remainingBudget int64) {
	opt := options{}
	for _, o := range opts {
		o(&opt)
	}

	return s.validate(ctx, fldPath, obj, oldObj, opt.ratchetingOptions, costBudget)
}

// ratchetingOptions stores the current correlation object and the nearest
// parent which was correlatable. The parent is stored so that we can check at
// the point an error is thrown whether it should be ratcheted using simple
// logic
// Key and Index should be used as normally to traverse to the next node.
type ratchetingOptions struct {
	// Current correlation object. If nil, then this node is from an uncorrelatable
	// part of the schema
	currentCorrelation *common.CorrelatedObject

	// If currentCorrelation is nil, this is the nearest parent to this node
	// which was correlatable. If the parent is deepequal to its old value,
	// then errors thrown by this node are ratcheted
	nearestParentCorrelation *common.CorrelatedObject
}

// shouldRatchetError returns true if the errors raised by the current node
// should be ratcheted.
//
// Errors for the current node should be ratcheted if one of the following is true:
//  1. The current node is correlatable, and it is equal to its old value
//  2. The current node has a correlatable ancestor, and the ancestor is equal
//     to its old value.
func (r ratchetingOptions) shouldRatchetError() bool {
	if r.currentCorrelation != nil {
		return r.currentCorrelation.CachedDeepEqual()
	}

	return r.nearestParentCorrelation.CachedDeepEqual()
}

// Finds the next node following the field in the tree and returns options using
// that node. If none could be found, then retains a reference to the last
// correlatable ancestor for ratcheting purposes
func (r ratchetingOptions) key(field string) ratchetingOptions {
	if r.currentCorrelation == nil {
		return r
	} else if r.nearestParentCorrelation == nil && (field == "apiVersion" || field == "kind") {
		// We cannot ratchet changes to the APIVersion and kind fields field since
		// they aren't visible. (both old and new are converted to the same type)
		//
		return ratchetingOptions{}
	}

	// nearestParentCorrelation is always non-nil except for the root node.
	// The below line ensures that the next nearestParentCorrelation is set
	// to a non-nil r.currentCorrelation
	return ratchetingOptions{currentCorrelation: r.currentCorrelation.Key(field), nearestParentCorrelation: r.currentCorrelation}
}

// Finds the next node following the index in the tree and returns options using
// that node. If none could be found, then retains a reference to the last
// correlatable ancestor for ratcheting purposes
func (r ratchetingOptions) index(idx int) ratchetingOptions {
	if r.currentCorrelation == nil {
		return r
	}

	return ratchetingOptions{currentCorrelation: r.currentCorrelation.Index(idx), nearestParentCorrelation: r.currentCorrelation}
}

func nestedToStructural(nested *schema.NestedValueValidation) *schema.Structural {
	if nested == nil {
		return nil
	}

	structuralConversion := &schema.Structural{
		ValueValidation:      &nested.ValueValidation,
		ValidationExtensions: nested.ValidationExtensions,
		Generic:              nested.ForbiddenGenerics,
		Extensions:           nested.ForbiddenExtensions,
		Items:                nestedToStructural(nested.Items),
	}

	if len(nested.Properties) > 0 {
		structuralConversion.Properties = make(map[string]schema.Structural, len(nested.Properties))
		for k, v := range nested.Properties {
			structuralConversion.Properties[k] = *nestedToStructural(&v)
		}
	}

	if nested.AdditionalProperties != nil {
		structuralConversion.AdditionalProperties = &schema.StructuralOrBool{
			Structural: nestedToStructural(nested.AdditionalProperties),
		}
	}

	return structuralConversion
}

func (s *Validator) validate(ctx context.Context, fldPath *field.Path, obj, oldObj interface{}, correlation ratchetingOptions, costBudget int64) (errs field.ErrorList, remainingBudget int64) {
	t := time.Now()
	defer func() {
		metrics.Metrics.ObserveEvaluation(time.Since(t))
	}()
	remainingBudget = costBudget
	if s == nil || obj == nil {
		return nil, remainingBudget
	}

	errs, remainingBudget = s.validateExpressions(ctx, fldPath, obj, oldObj, correlation, remainingBudget)

	if remainingBudget < 0 {
		return errs, remainingBudget
	}

	// If the schema has allOf, recurse through those elements to see if there
	// are any validation rules that need to be evaluated.
	for _, allOfValidator := range s.AllOfValidators {
		var allOfErrs field.ErrorList
		// Pass options with nil currentCorrelation to mirror schema ratcheting
		// behavior which does not ratchet allOf errors. This may change in the
		// future for allOf.
		allOfErrs, remainingBudget = allOfValidator.validate(ctx, fldPath, obj, oldObj, ratchetingOptions{nearestParentCorrelation: correlation.nearestParentCorrelation}, remainingBudget)
		errs = append(errs, allOfErrs...)
		if remainingBudget < 0 {
			return errs, remainingBudget
		}
	}

	switch obj := obj.(type) {
	case []interface{}:
		oldArray, _ := oldObj.([]interface{})
		var arrayErrs field.ErrorList
		arrayErrs, remainingBudget = s.validateArray(ctx, fldPath, obj, oldArray, correlation, remainingBudget)
		errs = append(errs, arrayErrs...)
		return errs, remainingBudget
	case map[string]interface{}:
		oldMap, _ := oldObj.(map[string]interface{})
		var mapErrs field.ErrorList
		mapErrs, remainingBudget = s.validateMap(ctx, fldPath, obj, oldMap, correlation, remainingBudget)
		errs = append(errs, mapErrs...)
		return errs, remainingBudget
	}

	return errs, remainingBudget
}

func (s *Validator) validateExpressions(ctx context.Context, fldPath *field.Path, obj, oldObj interface{}, correlation ratchetingOptions, costBudget int64) (errs field.ErrorList, remainingBudget int64) {
	sts := s.Schema

	// guard against oldObj being a non-nil interface with a nil value
	if oldObj != nil {
		v := reflect.ValueOf(oldObj)
		switch v.Kind() {
		case reflect.Map, reflect.Pointer, reflect.Interface, reflect.Slice:
			if v.IsNil() {
				oldObj = nil // +k8s:verify-mutation:reason=clone
			}
		}
	}

	remainingBudget = costBudget
	if obj == nil {
		// We only validate non-null values. Rules that need to check for the state of a nullable value or the presence of an optional
		// field must do so from the surrounding schema. E.g. if an array has nullable string items, a rule on the array
		// schema can check if items are null, but a rule on the nullable string schema only validates the non-null strings.
		return nil, remainingBudget
	}
	if s.compilationErr != nil {
		errs = append(errs, field.Invalid(fldPath, sts.Type, fmt.Sprintf("rule compiler initialization error: %v", s.compilationErr)))
		return errs, remainingBudget
	}
	if len(s.compiledRules) == 0 {
		return nil, remainingBudget // nothing to do
	}
	if remainingBudget <= 0 {
		errs = append(errs, field.Invalid(fldPath, sts.Type, fmt.Sprintf("validation failed due to running out of cost budget, no further validation rules will be run")))
		return errs, -1
	}
	if s.isResourceRoot {
		sts = model.WithTypeAndObjectMeta(sts)
	}
	activation, optionalOldSelfActivation := s.celActivationFactory(sts, obj, oldObj)
	for i, compiled := range s.compiledRules {
		rule := s.uncompiledRules[i]
		if compiled.Error != nil {
			errs = append(errs, field.Invalid(fldPath, sts.Type, fmt.Sprintf("rule compile error: %v", compiled.Error)))
			continue
		}
		if compiled.Program == nil {
			// rule is empty
			continue
		}

		// If ratcheting is enabled, allow rule with oldSelf to evaluate
		// when `optionalOldSelf` is set to true
		optionalOldSelfRule := ptr.Deref(rule.OptionalOldSelf, false)
		if compiled.UsesOldSelf && oldObj == nil {
			// transition rules are evaluated only if there is a comparable existing value
			// But if the rule uses optional oldSelf and gate is enabled we allow
			// the rule to be evaluated
			if !utilfeature.DefaultFeatureGate.Enabled(features.CRDValidationRatcheting) {
				continue
			}

			if !optionalOldSelfRule {
				continue
			}
		}

		ruleActivation := activation
		if optionalOldSelfRule {
			ruleActivation = optionalOldSelfActivation
		}

		evalResult, evalDetails, err := compiled.Program.ContextEval(ctx, ruleActivation)
		if evalDetails == nil {
			errs = append(errs, field.InternalError(fldPath, fmt.Errorf("runtime cost could not be calculated for validation rule: %v, no further validation rules will be run", ruleErrorString(rule))))
			return errs, -1
		} else {
			rtCost := evalDetails.ActualCost()
			if rtCost == nil {
				errs = append(errs, field.Invalid(fldPath, sts.Type, fmt.Sprintf("runtime cost could not be calculated for validation rule: %v, no further validation rules will be run", ruleErrorString(rule))))
				return errs, -1
			} else {
				if *rtCost > math.MaxInt64 || int64(*rtCost) > remainingBudget {
					errs = append(errs, field.Invalid(fldPath, sts.Type, fmt.Sprintf("validation failed due to running out of cost budget, no further validation rules will be run")))
					return errs, -1
				}
				remainingBudget -= int64(*rtCost)
			}
		}
		if err != nil {
			// see types.Err for list of well defined error types
			if strings.HasPrefix(err.Error(), "no such overload") {
				// Most overload errors are caught by the compiler, which provides details on where exactly in the rule
				// error was found. Here, an overload error has occurred at runtime no details are provided, so we
				// append a more descriptive error message. This error can only occur when static type checking has
				// been bypassed. int-or-string is typed as dynamic and so bypasses compiler type checking.
				errs = append(errs, field.Invalid(fldPath, sts.Type, fmt.Sprintf("'%v': call arguments did not match a supported operator, function or macro signature for rule: %v", err, ruleErrorString(rule))))
			} else if strings.HasPrefix(err.Error(), "operation cancelled: actual cost limit exceeded") {
				errs = append(errs, field.Invalid(fldPath, sts.Type, fmt.Sprintf("'%v': no further validation rules will be run due to call cost exceeds limit for rule: %v", err, ruleErrorString(rule))))
				return errs, -1
			} else {
				// no such key: {key}, index out of bounds: {index}, integer overflow, division by zero, ...
				errs = append(errs, field.Invalid(fldPath, sts.Type, fmt.Sprintf("%v evaluating rule: %v", err, ruleErrorString(rule))))
			}
			continue
		}

		if evalResult == types.True {
			continue
		}

		// Prepare a field error describing why the expression evaluated to False.
		// Its detail may come from another expression that might fail to evaluate or exceed the budget.

		currentFldPath := fldPath
		if len(compiled.NormalizedRuleFieldPath) > 0 {
			currentFldPath = currentFldPath.Child(compiled.NormalizedRuleFieldPath)
		}

		addErr := func(e *field.Error) {
			if !compiled.UsesOldSelf && correlation.shouldRatchetError() {
				warning.AddWarning(ctx, "", e.Error())
			} else {
				errs = append(errs, e)
			}
		}

		detail, ok := "", false
		if compiled.MessageExpression != nil {
			messageExpression, newRemainingBudget, msgErr := evalMessageExpression(ctx, compiled.MessageExpression, rule.MessageExpression, activation, remainingBudget)
			if msgErr == nil {
				detail, ok = messageExpression, true
				remainingBudget = newRemainingBudget
			} else if msgErr.Type == cel.ErrorTypeInternal {
				addErr(field.InternalError(currentFldPath, msgErr))
				return errs, -1
			} else if msgErr.Type == cel.ErrorTypeInvalid {
				addErr(field.Invalid(currentFldPath, sts.Type, msgErr.Error()))
				return errs, -1
			} else {
				klog.V(2).ErrorS(msgErr, "messageExpression evaluation failed")
				remainingBudget = newRemainingBudget
			}
		}
		if !ok {
			detail = ruleMessageOrDefault(rule)
		}

		value := obj
		if sts.Type == "object" || sts.Type == "array" {
			value = field.OmitValueType{}
		}

		addErr(fieldErrorForReason(currentFldPath, value, detail, rule.Reason))
	}
	return errs, remainingBudget
}

var unescapeMatcher = regexp.MustCompile(`\\.`)

func unescapeSingleQuote(s string) (string, error) {
	var err error
	unescaped := unescapeMatcher.ReplaceAllStringFunc(s, func(matchStr string) string {
		directive := matchStr[1]
		switch directive {
		case 'a':
			return "\a"
		case 'b':
			return "\b"
		case 'f':
			return "\f"
		case 'n':
			return "\n"
		case 'r':
			return "\r"
		case 't':
			return "\t"
		case 'v':
			return "\v"
		case '\'':
			return "'"
		case '\\':
			return "\\"
		default:
			err = fmt.Errorf("invalid escape char %s", matchStr)
			return ""
		}
	})
	return unescaped, err
}

type validFieldPathOptions struct {
	allowArrayNotation bool
}

// ValidFieldPathOption provides vararg options for ValidFieldPath.
type ValidFieldPathOption func(*validFieldPathOptions)

// WithFieldPathAllowArrayNotation sets of array annotation ('[<index or map key>]') is allowed
// in field paths.
// Defaults to true
func WithFieldPathAllowArrayNotation(allow bool) ValidFieldPathOption {
	return func(options *validFieldPathOptions) {
		options.allowArrayNotation = allow
	}
}

// ValidFieldPath validates that jsonPath is a valid JSON Path containing only field and map accessors
// that are valid for the given schema, and returns a field.Path representation of the validated jsonPath or an error.
func ValidFieldPath(jsonPath string, schema *schema.Structural, options ...ValidFieldPathOption) (validFieldPath *field.Path, foundSchema *schema.Structural, err error) {
	opts := &validFieldPathOptions{allowArrayNotation: true}
	for _, opt := range options {
		opt(opts)
	}

	appendToPath := func(name string, isNamed bool) error {
		if !isNamed {
			validFieldPath = validFieldPath.Key(name)
			schema = schema.AdditionalProperties.Structural
		} else {
			validFieldPath = validFieldPath.Child(name)
			val, ok := schema.Properties[name]
			if !ok {
				return fmt.Errorf("does not refer to a valid field")
			}
			schema = &val
		}
		return nil
	}

	validFieldPath = nil

	scanner := bufio.NewScanner(strings.NewReader(jsonPath))

	// configure the scanner to split the string into tokens.
	// The three delimiters ('.', '[', ']') will be returned as single char tokens.
	// All other text between delimiters is returned as string tokens.
	scanner.Split(func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		if len(data) > 0 {
			for i := 0; i < len(data); i++ {
				// If in a single quoted string, look for the end of string
				// ignoring delimiters.
				if data[0] == '\'' {
					if i > 0 && data[i] == '\'' && data[i-1] != '\\' {
						// Return quoted string
						return i + 1, data[:i+1], nil
					}
					continue
				}
				switch data[i] {
				case '.', '[', ']': // delimiters
					if i == 0 {
						// Return the delimiter.
						return 1, data[:1], nil
					} else {
						// Return identifier leading up to the delimiter.
						// The next call to split will return the delimiter.
						return i, data[:i], nil
					}
				}
			}
			if atEOF {
				// Return the string.
				return len(data), data, nil
			}
		}
		return 0, nil, nil
	})

	var tok string
	var isNamed bool
	for scanner.Scan() {
		tok = scanner.Text()
		switch tok {
		case "[":
			if !opts.allowArrayNotation {
				return nil, nil, fmt.Errorf("array notation is not allowed")
			}
			if !scanner.Scan() {
				return nil, nil, fmt.Errorf("unexpected end of JSON path")
			}
			tok = scanner.Text()
			if len(tok) < 2 || tok[0] != '\'' || tok[len(tok)-1] != '\'' {
				return nil, nil, fmt.Errorf("expected single quoted string but got %s", tok)
			}
			unescaped, err := unescapeSingleQuote(tok[1 : len(tok)-1])
			if err != nil {
				return nil, nil, fmt.Errorf("invalid string literal: %w", err)
			}

			if schema.Properties != nil {
				isNamed = true
			} else if schema.AdditionalProperties != nil {
				isNamed = false
			} else {
				return nil, nil, fmt.Errorf("does not refer to a valid field")
			}
			if err := appendToPath(unescaped, isNamed); err != nil {
				return nil, nil, err
			}
			if !scanner.Scan() {
				return nil, nil, fmt.Errorf("unexpected end of JSON path")
			}
			tok = scanner.Text()
			if tok != "]" {
				return nil, nil, fmt.Errorf("expected ] but got %s", tok)
			}
		case ".":
			if !scanner.Scan() {
				return nil, nil, fmt.Errorf("unexpected end of JSON path")
			}
			tok = scanner.Text()
			if schema.Properties != nil {
				isNamed = true
			} else if schema.AdditionalProperties != nil {
				isNamed = false
			} else {
				return nil, nil, fmt.Errorf("does not refer to a valid field")
			}
			if err := appendToPath(tok, isNamed); err != nil {
				return nil, nil, err
			}
		default:
			return nil, nil, fmt.Errorf("expected [ or . but got: %s", tok)
		}
	}

	return validFieldPath, schema, nil
}

func fieldErrorForReason(fldPath *field.Path, value interface{}, detail string, reason *apiextensions.FieldValueErrorReason) *field.Error {
	if reason == nil {
		return field.Invalid(fldPath, value, detail)
	}
	switch *reason {
	case apiextensions.FieldValueForbidden:
		return field.Forbidden(fldPath, detail)
	case apiextensions.FieldValueRequired:
		return field.Required(fldPath, detail)
	case apiextensions.FieldValueDuplicate:
		return field.Duplicate(fldPath, value)
	default:
		return field.Invalid(fldPath, value, detail)
	}
}

// evalMessageExpression evaluates the given message expression and returns the evaluated string form and the remaining budget, or an error if one
// occurred during evaluation.
func evalMessageExpression(ctx context.Context, expr celgo.Program, exprSrc string, activation interpreter.Activation, remainingBudget int64) (string, int64, *cel.Error) {
	evalResult, evalDetails, err := expr.ContextEval(ctx, activation)
	if evalDetails == nil {
		return "", -1, &cel.Error{
			Type:   cel.ErrorTypeInternal,
			Detail: fmt.Sprintf("runtime cost could not be calculated for messageExpression: %q", exprSrc),
		}
	}
	rtCost := evalDetails.ActualCost()
	if rtCost == nil {
		return "", -1, &cel.Error{
			Type:   cel.ErrorTypeInternal,
			Detail: fmt.Sprintf("runtime cost could not be calculated for messageExpression: %q", exprSrc),
		}
	} else if *rtCost > math.MaxInt64 || int64(*rtCost) > remainingBudget {
		return "", -1, &cel.Error{
			Type:   cel.ErrorTypeInvalid,
			Detail: "messageExpression evaluation failed due to running out of cost budget, no further validation rules will be run",
		}
	}
	if err != nil {
		if strings.HasPrefix(err.Error(), "operation cancelled: actual cost limit exceeded") {
			return "", -1, &cel.Error{
				Type:   cel.ErrorTypeInvalid,
				Detail: fmt.Sprintf("no further validation rules will be run due to call cost exceeds limit for messageExpression: %q", exprSrc),
			}
		}
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: fmt.Sprintf("messageExpression evaluation failed due to: %v", err.Error()),
		}
	}
	messageStr, ok := evalResult.Value().(string)
	if !ok {
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: "messageExpression failed to convert to string",
		}
	}
	trimmedMsgStr := strings.TrimSpace(messageStr)
	if len(trimmedMsgStr) > celconfig.MaxEvaluatedMessageExpressionSizeBytes {
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: fmt.Sprintf("messageExpression beyond allowable length of %d", celconfig.MaxEvaluatedMessageExpressionSizeBytes),
		}
	} else if hasNewlines(trimmedMsgStr) {
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: "messageExpression should not contain line breaks",
		}
	} else if len(trimmedMsgStr) == 0 {
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: "messageExpression should evaluate to a non-empty string",
		}
	}
	return trimmedMsgStr, remainingBudget - int64(*rtCost), nil
}

var newlineMatcher = regexp.MustCompile(`[\n]+`)

func hasNewlines(s string) bool {
	return newlineMatcher.MatchString(s)
}

func ruleMessageOrDefault(rule apiextensions.ValidationRule) string {
	if len(rule.Message) == 0 {
		return fmt.Sprintf("failed rule: %s", ruleErrorString(rule))
	} else {
		return strings.TrimSpace(rule.Message)
	}
}

func ruleErrorString(rule apiextensions.ValidationRule) string {
	if len(rule.Message) > 0 {
		return strings.TrimSpace(rule.Message)
	}
	return strings.TrimSpace(rule.Rule)
}

type validationActivation struct {
	self, oldSelf ref.Val
	hasOldSelf    bool
}

func validationActivationWithOldSelf(sts *schema.Structural, obj, oldObj interface{}) (activation interpreter.Activation, optionalOldSelfActivation interpreter.Activation) {
	va := &validationActivation{
		self: UnstructuredToVal(obj, sts),
	}
	optionalVA := &validationActivation{
		self:       va.self,
		hasOldSelf: true, // this means the oldSelf variable is defined for CEL to reference, not that it has a value
		oldSelf:    types.OptionalNone,
	}

	if oldObj != nil {
		va.oldSelf = UnstructuredToVal(oldObj, sts) // +k8s:verify-mutation:reason=clone
		va.hasOldSelf = true

		optionalVA.oldSelf = types.OptionalOf(va.oldSelf) // +k8s:verify-mutation:reason=clone
	}

	return va, optionalVA
}

func validationActivationWithoutOldSelf(sts *schema.Structural, obj, _ interface{}) (interpreter.Activation, interpreter.Activation) {
	res := &validationActivation{
		self: UnstructuredToVal(obj, sts),
	}
	return res, res
}

func (a *validationActivation) ResolveName(name string) (interface{}, bool) {
	switch name {
	case ScopedVarName:
		return a.self, true
	case OldScopedVarName:
		return a.oldSelf, a.hasOldSelf
	default:
		return nil, false
	}
}

func (a *validationActivation) Parent() interpreter.Activation {
	return nil
}

func (s *Validator) validateMap(ctx context.Context, fldPath *field.Path, obj, oldObj map[string]interface{}, correlation ratchetingOptions, costBudget int64) (errs field.ErrorList, remainingBudget int64) {
	remainingBudget = costBudget
	if remainingBudget < 0 {
		return errs, remainingBudget
	}
	if s == nil || obj == nil {
		return nil, remainingBudget
	}

	correlatable := MapIsCorrelatable(s.Schema.XMapType)

	if s.AdditionalProperties != nil {
		for k, v := range obj {
			var oldV interface{}
			if correlatable {
				oldV = oldObj[k] // +k8s:verify-mutation:reason=clone
			}

			var err field.ErrorList
			err, remainingBudget = s.AdditionalProperties.validate(ctx, fldPath.Key(k), v, oldV, correlation.key(k), remainingBudget)
			errs = append(errs, err...)
			if remainingBudget < 0 {
				return errs, remainingBudget
			}
		}
	}
	if s.Properties != nil {
		for k, v := range obj {
			sub, ok := s.Properties[k]
			if ok {
				var oldV interface{}
				if correlatable {
					oldV = oldObj[k] // +k8s:verify-mutation:reason=clone
				}

				var err field.ErrorList
				err, remainingBudget = sub.validate(ctx, fldPath.Child(k), v, oldV, correlation.key(k), remainingBudget)
				errs = append(errs, err...)
				if remainingBudget < 0 {
					return errs, remainingBudget
				}
			}
		}
	}

	return errs, remainingBudget
}

func (s *Validator) validateArray(ctx context.Context, fldPath *field.Path, obj, oldObj []interface{}, correlation ratchetingOptions, costBudget int64) (errs field.ErrorList, remainingBudget int64) {
	remainingBudget = costBudget
	if remainingBudget < 0 {
		return errs, remainingBudget
	}

	if s.Items != nil {
		// only map-type lists support self-oldSelf correlation for cel rules. if this isn't a
		// map-type list, then makeMapList returns an implementation that always returns nil
		correlatableOldItems := makeMapList(s.Schema, oldObj)
		for i := range obj {
			var err field.ErrorList
			err, remainingBudget = s.Items.validate(ctx, fldPath.Index(i), obj[i], correlatableOldItems.Get(obj[i]), correlation.index(i), remainingBudget)
			errs = append(errs, err...)
			if remainingBudget < 0 {
				return errs, remainingBudget
			}
		}
	}

	return errs, remainingBudget
}

// MapIsCorrelatable returns true if the mapType can be used to correlate the data elements of a map after an update
// with the data elements of the map from before the updated.
func MapIsCorrelatable(mapType *string) bool {
	// if a third map type is introduced, assume it's not correlatable. granular is the default if unspecified.
	return mapType == nil || *mapType == "granular" || *mapType == "atomic"
}

func hasXValidations(s *schema.Structural) bool {
	if s == nil {
		return false
	}
	if len(s.XValidations) > 0 {
		return true
	}
	if hasXValidations(s.Items) {
		return true
	}
	if s.AdditionalProperties != nil && hasXValidations(s.AdditionalProperties.Structural) {
		return true
	}
	if s.Properties != nil {
		for _, prop := range s.Properties {
			if hasXValidations(&prop) {
				return true
			}
		}
	}
	return false
}
