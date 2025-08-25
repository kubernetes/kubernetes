// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cel

import (
	"fmt"
	"reflect"
	"regexp"

	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/env"
	"github.com/google/cel-go/common/overloads"
)

const (
	durationValidatorName     = "cel.validator.duration"
	regexValidatorName        = "cel.validator.matches"
	timestampValidatorName    = "cel.validator.timestamp"
	homogeneousValidatorName  = "cel.validator.homogeneous_literals"
	nestingLimitValidatorName = "cel.validator.comprehension_nesting_limit"

	// HomogeneousAggregateLiteralExemptFunctions is the ValidatorConfig key used to configure
	// the set of function names which are exempt from homogeneous type checks. The expected type
	// is a string list of function names.
	//
	// As an example, the `<string>.format([args])` call expects the input arguments list to be
	// comprised of a variety of types which correspond to the types expected by the format control
	// clauses; however, all other uses of a mixed element type list, would be unexpected.
	HomogeneousAggregateLiteralExemptFunctions = homogeneousValidatorName + ".exempt"
)

var (
	astValidatorFactories = map[string]ASTValidatorFactory{
		nestingLimitValidatorName: func(val *env.Validator) (ASTValidator, error) {
			if limit, found := val.ConfigValue("limit"); found {
				if val, isInt := limit.(int); isInt {
					return ValidateComprehensionNestingLimit(val), nil
				}
				return nil, fmt.Errorf("invalid validator: %s unsupported limit type: %v", nestingLimitValidatorName, limit)
			}
			return nil, fmt.Errorf("invalid validator: %s missing limit", nestingLimitValidatorName)
		},
		durationValidatorName: func(*env.Validator) (ASTValidator, error) {
			return ValidateDurationLiterals(), nil
		},
		regexValidatorName: func(*env.Validator) (ASTValidator, error) {
			return ValidateRegexLiterals(), nil
		},
		timestampValidatorName: func(*env.Validator) (ASTValidator, error) {
			return ValidateTimestampLiterals(), nil
		},
		homogeneousValidatorName: func(*env.Validator) (ASTValidator, error) {
			return ValidateHomogeneousAggregateLiterals(), nil
		},
	}
)

// ASTValidatorFactory creates an ASTValidator as configured by the input map
type ASTValidatorFactory func(*env.Validator) (ASTValidator, error)

// ASTValidators configures a set of ASTValidator instances into the target environment.
//
// Validators are applied in the order in which the are specified and are treated as singletons.
// The same ASTValidator with a given name will not be applied more than once.
func ASTValidators(validators ...ASTValidator) EnvOption {
	return func(e *Env) (*Env, error) {
		for _, v := range validators {
			if !e.HasValidator(v.Name()) {
				e.validators = append(e.validators, v)
			}
		}
		return e, nil
	}
}

// ASTValidator defines a singleton interface for validating a type-checked Ast against an environment.
//
// Note: the Issues argument is mutable in the sense that it is intended to collect errors which will be
// reported to the caller.
type ASTValidator interface {
	// Name returns the name of the validator. Names must be unique.
	Name() string

	// Validate validates a given Ast within an Environment and collects a set of potential issues.
	//
	// The ValidatorConfig is generated from the set of ASTValidatorConfigurer instances prior to
	// the invocation of the Validate call. The expectation is that the validator configuration
	// is created in sequence and immutable once provided to the Validate call.
	//
	// See individual validators for more information on their configuration keys and configuration
	// properties.
	Validate(*Env, ValidatorConfig, *ast.AST, *Issues)
}

// ConfigurableASTValidator supports conversion of an object to an `env.Validator` instance used for
// YAML serialization.
type ConfigurableASTValidator interface {
	// ToConfig converts the internal configuration of an ASTValidator into an env.Validator instance
	// which minimally must include the validator name, but may also include a map[string]any config
	// object to be serialized to YAML. The string keys represent the configuration parameter name,
	// and the any value must mirror the internally supported type associated with the config key.
	//
	// Note: only primitive CEL types are supported by CEL validators at this time.
	ToConfig() *env.Validator
}

// ValidatorConfig provides an accessor method for querying validator configuration state.
type ValidatorConfig interface {
	GetOrDefault(name string, value any) any
}

// MutableValidatorConfig provides mutation methods for querying and updating validator configuration
// settings.
type MutableValidatorConfig interface {
	ValidatorConfig
	Set(name string, value any) error
}

// ASTValidatorConfigurer indicates that this object, currently expected to be an ASTValidator,
// participates in validator configuration settings.
//
// This interface may be split from the expectation of being an ASTValidator instance in the future.
type ASTValidatorConfigurer interface {
	Configure(MutableValidatorConfig) error
}

// validatorConfig implements the ValidatorConfig and MutableValidatorConfig interfaces.
type validatorConfig struct {
	data map[string]any
}

// newValidatorConfig initializes the validator config with default values for core CEL validators.
func newValidatorConfig() *validatorConfig {
	return &validatorConfig{
		data: map[string]any{
			HomogeneousAggregateLiteralExemptFunctions: []string{},
		},
	}
}

// GetOrDefault returns the configured value for the name, if present, else the input default value.
//
// Note, the type-agreement between the input default and configured value is not checked on read.
func (config *validatorConfig) GetOrDefault(name string, value any) any {
	v, found := config.data[name]
	if !found {
		return value
	}
	return v
}

// Set configures a validator option with the given name and value.
//
// If the value had previously been set, the new value must have the same reflection type as the old one,
// or the call will error.
func (config *validatorConfig) Set(name string, value any) error {
	v, found := config.data[name]
	if found && reflect.TypeOf(v) != reflect.TypeOf(value) {
		return fmt.Errorf("incompatible configuration type for %s, got %T, wanted %T", name, value, v)
	}
	config.data[name] = value
	return nil
}

// ExtendedValidations collects a set of common AST validations which reduce the likelihood of runtime errors.
//
// - Validate duration and timestamp literals
// - Ensure regex strings are valid
// - Disable mixed type list and map literals
func ExtendedValidations() EnvOption {
	return ASTValidators(
		ValidateDurationLiterals(),
		ValidateTimestampLiterals(),
		ValidateRegexLiterals(),
		ValidateHomogeneousAggregateLiterals(),
	)
}

// ValidateDurationLiterals ensures that duration literal arguments are valid immediately after type-check.
func ValidateDurationLiterals() ASTValidator {
	return newFormatValidator(overloads.TypeConvertDuration, 0, evalCall)
}

// ValidateTimestampLiterals ensures that timestamp literal arguments are valid immediately after type-check.
func ValidateTimestampLiterals() ASTValidator {
	return newFormatValidator(overloads.TypeConvertTimestamp, 0, evalCall)
}

// ValidateRegexLiterals ensures that regex patterns are validated after type-check.
func ValidateRegexLiterals() ASTValidator {
	return newFormatValidator(overloads.Matches, 0, compileRegex)
}

// ValidateHomogeneousAggregateLiterals checks that all list and map literals entries have the same types, i.e.
// no mixed list element types or mixed map key or map value types.
//
// Note: the string format call relies on a mixed element type list for ease of use, so this check skips all
// literals which occur within string format calls.
func ValidateHomogeneousAggregateLiterals() ASTValidator {
	return homogeneousAggregateLiteralValidator{}
}

// ValidateComprehensionNestingLimit ensures that comprehension nesting does not exceed the specified limit.
//
// This validator can be useful for preventing arbitrarily nested comprehensions which can take high polynomial
// time to complete.
//
// Note, this limit does not apply to comprehensions with an empty iteration range, as these comprehensions have
// no actual looping cost. The cel.bind() utilizes the comprehension structure to perform local variable
// assignments and supplies an empty iteration range, so they won't count against the nesting limit either.
func ValidateComprehensionNestingLimit(limit int) ASTValidator {
	return nestingLimitValidator{limit: limit}
}

type argChecker func(env *Env, call, arg ast.Expr) error

func newFormatValidator(funcName string, argNum int, check argChecker) formatValidator {
	return formatValidator{
		funcName: funcName,
		check:    check,
		argNum:   argNum,
	}
}

type formatValidator struct {
	funcName string
	argNum   int
	check    argChecker
}

// Name returns the unique name of this function format validator.
func (v formatValidator) Name() string {
	return fmt.Sprintf("cel.validator.%s", v.funcName)
}

// ToConfig converts the ASTValidator to an env.Validator specifying the validator name.
func (v formatValidator) ToConfig() *env.Validator {
	return env.NewValidator(v.Name())
}

// Validate searches the AST for uses of a given function name with a constant argument and performs a check
// on whether the argument is a valid literal value.
func (v formatValidator) Validate(e *Env, _ ValidatorConfig, a *ast.AST, iss *Issues) {
	root := ast.NavigateAST(a)
	funcCalls := ast.MatchDescendants(root, ast.FunctionMatcher(v.funcName))
	for _, call := range funcCalls {
		callArgs := call.AsCall().Args()
		if len(callArgs) <= v.argNum {
			continue
		}
		litArg := callArgs[v.argNum]
		if litArg.Kind() != ast.LiteralKind {
			continue
		}
		if err := v.check(e, call, litArg); err != nil {
			iss.ReportErrorAtID(litArg.ID(), "invalid %s argument", v.funcName)
		}
	}
}

func evalCall(env *Env, call, arg ast.Expr) error {
	ast := &Ast{impl: ast.NewAST(call, ast.NewSourceInfo(nil))}
	prg, err := env.Program(ast)
	if err != nil {
		return err
	}
	_, _, err = prg.Eval(NoVars())
	return err
}

func compileRegex(_ *Env, _, arg ast.Expr) error {
	pattern := arg.AsLiteral().Value().(string)
	_, err := regexp.Compile(pattern)
	return err
}

type homogeneousAggregateLiteralValidator struct{}

// Name returns the unique name of the homogeneous type validator.
func (homogeneousAggregateLiteralValidator) Name() string {
	return homogeneousValidatorName
}

// ToConfig converts the ASTValidator to an env.Validator specifying the validator name.
func (v homogeneousAggregateLiteralValidator) ToConfig() *env.Validator {
	return env.NewValidator(v.Name())
}

// Validate validates that all lists and map literals have homogeneous types, i.e. don't contain dyn types.
//
// This validator makes an exception for list and map literals which occur at any level of nesting within
// string format calls.
func (v homogeneousAggregateLiteralValidator) Validate(_ *Env, c ValidatorConfig, a *ast.AST, iss *Issues) {
	var exemptedFunctions []string
	exemptedFunctions = c.GetOrDefault(HomogeneousAggregateLiteralExemptFunctions, exemptedFunctions).([]string)
	root := ast.NavigateAST(a)
	listExprs := ast.MatchDescendants(root, ast.KindMatcher(ast.ListKind))
	for _, listExpr := range listExprs {
		if inExemptFunction(listExpr, exemptedFunctions) {
			continue
		}
		l := listExpr.AsList()
		elements := l.Elements()
		optIndices := l.OptionalIndices()
		var elemType *Type
		for i, e := range elements {
			et := a.GetType(e.ID())
			if isOptionalIndex(i, optIndices) {
				et = et.Parameters()[0]
			}
			if elemType == nil {
				elemType = et
				continue
			}
			if !elemType.IsEquivalentType(et) {
				v.typeMismatch(iss, e.ID(), elemType, et)
				break
			}
		}
	}
	mapExprs := ast.MatchDescendants(root, ast.KindMatcher(ast.MapKind))
	for _, mapExpr := range mapExprs {
		if inExemptFunction(mapExpr, exemptedFunctions) {
			continue
		}
		m := mapExpr.AsMap()
		entries := m.Entries()
		var keyType, valType *Type
		for _, e := range entries {
			mapEntry := e.AsMapEntry()
			key, val := mapEntry.Key(), mapEntry.Value()
			kt, vt := a.GetType(key.ID()), a.GetType(val.ID())
			if mapEntry.IsOptional() {
				vt = vt.Parameters()[0]
			}
			if keyType == nil && valType == nil {
				keyType, valType = kt, vt
				continue
			}
			if !keyType.IsEquivalentType(kt) {
				v.typeMismatch(iss, key.ID(), keyType, kt)
			}
			if !valType.IsEquivalentType(vt) {
				v.typeMismatch(iss, val.ID(), valType, vt)
			}
		}
	}
}

func inExemptFunction(e ast.NavigableExpr, exemptFunctions []string) bool {
	parent, found := e.Parent()
	for found {
		if parent.Kind() == ast.CallKind {
			fnName := parent.AsCall().FunctionName()
			for _, exempt := range exemptFunctions {
				if exempt == fnName {
					return true
				}
			}
		}
		parent, found = parent.Parent()
	}
	return false
}

func isOptionalIndex(i int, optIndices []int32) bool {
	for _, optInd := range optIndices {
		if i == int(optInd) {
			return true
		}
	}
	return false
}

func (homogeneousAggregateLiteralValidator) typeMismatch(iss *Issues, id int64, expected, actual *Type) {
	iss.ReportErrorAtID(id, "expected type '%s' but found '%s'", FormatCELType(expected), FormatCELType(actual))
}

type nestingLimitValidator struct {
	limit int
}

// Name returns the name of the nesting limit validator.
func (v nestingLimitValidator) Name() string {
	return nestingLimitValidatorName
}

// ToConfig converts the ASTValidator to an env.Validator specifying the validator name and the nesting limit
// as an integer value: {"limit": int}
func (v nestingLimitValidator) ToConfig() *env.Validator {
	return env.NewValidator(v.Name()).SetConfig(map[string]any{"limit": v.limit})
}

// Validate implements the ASTValidator interface method.
func (v nestingLimitValidator) Validate(e *Env, _ ValidatorConfig, a *ast.AST, iss *Issues) {
	root := ast.NavigateAST(a)
	comprehensions := ast.MatchDescendants(root, ast.KindMatcher(ast.ComprehensionKind))
	if len(comprehensions) <= v.limit {
		return
	}
	for _, comp := range comprehensions {
		count := 0
		e := comp
		hasParent := true
		for hasParent {
			// When the expression is not a comprehension, continue to the next ancestor.
			if e.Kind() != ast.ComprehensionKind {
				e, hasParent = e.Parent()
				continue
			}
			// When the comprehension has an empty range, continue to the next ancestor
			// as this comprehension does not have any associated cost.
			iterRange := e.AsComprehension().IterRange()
			if iterRange.Kind() == ast.ListKind && iterRange.AsList().Size() == 0 {
				e, hasParent = e.Parent()
				continue
			}
			// Otherwise check the nesting limit.
			count++
			if count > v.limit {
				iss.ReportErrorAtID(comp.ID(), "comprehension exceeds nesting limit")
				break
			}
			e, hasParent = e.Parent()
		}
	}
}
