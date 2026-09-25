/*
Copyright 2022 The Kubernetes Authors.

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
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/ext"

	"k8s.io/utils/ptr"

	"github.com/blang/semver/v4"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

const (
	deviceVar     = "device"
	driverVar     = "driver"
	multiAllocVar = "allowMultipleAllocations"
	attributesVar = "attributes"
	capacityVar   = "capacity"
)

var (
	layzCompilerMutex sync.Mutex
	lazyCompiler      *compiler
	lazyFeatures      Features

	// Other strings also have a known maximum size.
	domainType = withMaxElements(apiservercel.StringType, resourceapi.DeviceMaxDomainLength)
	idType     = withMaxElements(apiservercel.StringType, resourceapi.DeviceMaxIDLength)
	driverType = withMaxElements(apiservercel.StringType, resourceapi.DriverNameMaxLength)

	// A variant of BoolType with a known cost. Usage of apiservercel.BoolType
	// is underestimated without this (found when comparing estimated against
	// actual cost in compile_test.go).
	multiAllocType = withMaxElements(apiservercel.BoolType, 1)

	// Same for capacity.
	innerCapacityMapType = apiservercel.NewMapType(idType, apiservercel.QuantityDeclType, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice)
	outerCapacityMapType = apiservercel.NewMapType(domainType, innerCapacityMapType, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice)
)

// Features contains feature gates supported by the package.
type Features struct {
	EnableConsumableCapacity bool
	EnableListTypeAttributes bool
}

func GetCompiler(features Features) *compiler {
	layzCompilerMutex.Lock()
	defer layzCompilerMutex.Unlock()

	// In practice, features should not change back and forth between calls,
	// so only one compiler gets cached.
	if lazyCompiler == nil || lazyFeatures != features {
		lazyCompiler = newCompiler(features)
		lazyFeatures = features
	}
	return lazyCompiler
}

// CompilationResult represents a compiled expression.
type CompilationResult struct {
	Program     cel.Program
	Error       *apiservercel.Error
	Expression  string
	OutputType  *cel.Type
	Environment *cel.Env

	// MaxCost represents the worst-case cost of the compiled MessageExpression in terms of CEL's cost units,
	// as used by cel.EstimateCost.
	MaxCost uint64

	typesFromEnv
	emptyMapVal ref.Val
}

// Device defines the input values for a CEL selector expression.
type Device struct {
	// Driver gets used as domain for any attribute which does not already
	// have a domain prefix. If set, then it is also made available as a
	// string attribute.
	Driver                   string
	AllowMultipleAllocations *bool
	LookupUniqueString       func(string) draapi.UniqueString
	Attributes               draapi.DeviceAttributes
	Capacity                 draapi.DeviceCapacities
}

type compiler struct {
	envset        *environment.EnvSet
	declaredTypes []*apiservercel.DeclType
	features      Features
}

// Options contains several additional parameters
// for [CompileCELExpression]. All of them have reasonable
// defaults.
type Options struct {
	// EnvType allows to override the default environment type [environment.StoredExpressions].
	EnvType *environment.Type

	// CostLimit allows overriding the default runtime cost limit [resourceapi.CELSelectorExpressionMaxCost].
	CostLimit *uint64

	// DisableCostEstimation can be set to skip estimating the worst-case CEL cost.
	// If disabled or after an error, [CompilationResult.MaxCost] will be set to [math.Uint64].
	DisableCostEstimation bool

	// DerivedAttribute indicates if the expression is for a derived attribute.
	DerivedAttribute bool
}

// CompileCELExpression returns a compiled CEL expression. It evaluates to bool.
//
// TODO (https://github.com/kubernetes/kubernetes/issues/125826): validate AST to detect invalid attribute names.
func (c *compiler) CompileCELExpression(expression string, options Options) CompilationResult {
	resultError := func(errorString string, errType apiservercel.ErrorType) CompilationResult {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   errType,
				Detail: errorString,
			},
			Expression: expression,
			MaxCost:    math.MaxUint64,
		}
	}

	envType := ptr.Deref(options.EnvType, environment.StoredExpressions)
	env, err := c.envset.Env(envType)
	if err != nil {
		return resultError(fmt.Sprintf("unexpected error loading CEL environment: %v", err), apiservercel.ErrorTypeInternal)
	}

	ast, issues := env.Compile(expression)
	if issues != nil {
		return resultError("compilation failed: "+issues.String(), apiservercel.ErrorTypeInvalid)
	}

	typesFromEnv, err := c.getTypesFromEnv(env)
	if err != nil {
		return resultError("unexpected error loading CEL environment: "+err.Error(), apiservercel.ErrorTypeInternal)
	}

	if outputTypeErr := c.validateOutputType(typesFromEnv, ast.OutputType(), envType, options.DerivedAttribute); outputTypeErr != nil {
		return resultError(outputTypeErr.Detail, outputTypeErr.Type)
	}

	_, err = cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	prog, err := env.Program(ast,
		// The Kubernetes CEL base environment sets the VAP limit as runtime cost limit.
		// DRA has its own default cost limit and also allows the caller to change that
		// limit.
		cel.CostLimit(ptr.Deref(options.CostLimit, resourceapi.CELSelectorExpressionMaxCost)),
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
	if err != nil {
		return resultError("program instantiation failed: "+err.Error(), apiservercel.ErrorTypeInternal)
	}

	compilationResult := CompilationResult{
		Program:      prog,
		Expression:   expression,
		OutputType:   ast.OutputType(),
		Environment:  env,
		emptyMapVal:  env.CELTypeAdapter().NativeToValue(map[string]any{}),
		MaxCost:      math.MaxUint64,
		typesFromEnv: typesFromEnv,
	}

	if !options.DisableCostEstimation {
		// We don't have a SizeEstimator. The potential size of the input (= a
		// device) is already declared in the definition of the environment.
		estimator := c.newCostEstimator(typesFromEnv)
		costEst, err := env.EstimateCost(ast, estimator)
		if err != nil {
			compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "cost estimation failed: " + err.Error()}
			return compilationResult
		}
		compilationResult.MaxCost = costEst.Max
	}

	return compilationResult
}

func (c *compiler) newCostEstimator(typesFromEnv typesFromEnv) checker.CostEstimator {
	return &library.CostEstimator{SizeEstimator: &sizeEstimator{typesFromEnv}}
}

type typesFromEnv struct {
	deviceType          *apiservercel.DeclType // "device"
	outerAttributesType *apiservercel.DeclType // "device.attributes" = map to map to value
	innerAttributesType *apiservercel.DeclType // "device.attributes[foo]" = map to value
	attributeType       *apiservercel.DeclType // "device.attributes[foo].bar" = value
}

// getTypesFromEnv determines the actual type definitions based on
// the environment, which reflects features and stored vs. new expression.
func (c *compiler) getTypesFromEnv(env *cel.Env) (typesFromEnv, error) {
	var result typesFromEnv

	// Use the *latest* variable with the right name because that
	// instance overwrites the ones before it.
	for _, variable := range slices.Backward(env.Variables()) {
		if variable.Name() == deviceVar {
			result.deviceType = c.getDeclType(variable.Type())
			break
		}
	}
	if result.deviceType == nil {
		return result, fmt.Errorf("%q variable is not declared", deviceVar)
	}

	fieldType, ok := env.CELTypeProvider().FindStructFieldType(result.deviceType.TypeName(), attributesVar)
	if !ok {
		return result, fmt.Errorf("%q variable does not declare %q field", deviceVar, attributesVar)
	}
	result.outerAttributesType = c.getDeclType(fieldType.Type)
	if result.outerAttributesType == nil {
		return result, fmt.Errorf("%q field has no type", attributesVar)
	}

	outerMapParams := result.outerAttributesType.CelType().Parameters()
	if len(outerMapParams) != 2 {
		return result, fmt.Errorf("%q field should be a map, got %s", attributesVar, result.outerAttributesType.String())
	}
	result.innerAttributesType = c.getDeclType(outerMapParams[1])
	if result.innerAttributesType == nil {
		return result, fmt.Errorf("%q field does not map to type", attributesVar)
	}

	innerMapParams := result.innerAttributesType.CelType().Parameters()
	if len(innerMapParams) != 2 {
		return result, fmt.Errorf("%q field values should be maps, got %s", attributesVar, result.innerAttributesType.String())
	}
	result.attributeType = c.getDeclType(innerMapParams[1])
	if result.attributeType == nil {
		return result, fmt.Errorf("%q field mapped to no type at leafs", attributesVar)
	}

	return result, nil
}

// getDeclType maps the cel.Type to the apiservercel.DeclType. Only works for types defined by newCompiler.
func (c *compiler) getDeclType(t *cel.Type) *apiservercel.DeclType {
	for _, dt := range c.declaredTypes {
		if dt.CelType() == t {
			return dt
		}
	}
	return nil
}

var boolType = reflect.TypeOf(true)

func (c *CompilationResult) DeviceMatches(ctx context.Context, input Device) (bool, *cel.EvalDetails, error) {
	attributes := &domainToAttributes{
		typesFromEnv:        c.typesFromEnv,
		adapter:             c.Environment.CELTypeAdapter(),
		lookupUniqueStrings: input.LookupUniqueString,
		emptyMapValue:       c.emptyMapVal,
		attrs:               input.Attributes.Nested,
	}

	capacity := &domainToCapacity{
		lookupUniqueStrings: input.LookupUniqueString,
		emptyMapValue:       c.emptyMapVal,
		caps:                input.Capacity.Nested,
	}

	variables := map[string]any{
		deviceVar: map[string]any{
			driverVar:     input.Driver,
			multiAllocVar: ptr.Deref(input.AllowMultipleAllocations, false),
			attributesVar: attributes,
			capacityVar:   capacity,
		},
	}

	result, details, err := c.Program.ContextEval(ctx, variables)
	if err != nil {
		// CEL does not wrap the context error. We have to deduce why it failed.
		// See https://github.com/google/cel-go/issues/1195.
		if strings.Contains(err.Error(), "operation interrupted") && ctx.Err() != nil {
			return false, details, fmt.Errorf("%w: %w", err, context.Cause(ctx))
		}
		return false, details, err
	}
	resultAny, err := result.ConvertToNative(boolType)
	if err != nil {
		return false, details, fmt.Errorf("CEL result of type %s could not be converted to bool: %w", result.Type().TypeName(), err)
	}
	resultBool, ok := resultAny.(bool)
	if !ok {
		return false, details, fmt.Errorf("CEL native result value should have been a bool, got instead: %T", resultAny)
	}
	return resultBool, details, nil
}

// EvaluateDerivedAttribute evaluates the compiled CEL expression as a derived attribute against a device,
// returning the result using the same types as the attributes in draapi.Device or an error.
func (c CompilationResult) EvaluateDerivedAttribute(ctx context.Context, input Device) (any, *cel.EvalDetails, error) {
	attributes := &domainToAttributes{
		typesFromEnv:        c.typesFromEnv,
		adapter:             c.Environment.CELTypeAdapter(),
		lookupUniqueStrings: input.LookupUniqueString,
		emptyMapValue:       c.emptyMapVal,
		attrs:               input.Attributes.Nested,
	}

	capacity := &domainToCapacity{
		lookupUniqueStrings: input.LookupUniqueString,
		emptyMapValue:       c.emptyMapVal,
		caps:                input.Capacity.Nested,
	}
	variables := map[string]any{
		deviceVar: map[string]any{
			driverVar:     input.Driver,
			multiAllocVar: ptr.Deref(input.AllowMultipleAllocations, false),
			attributesVar: attributes,
			capacityVar:   capacity,
		},
	}

	result, details, err := c.Program.ContextEval(ctx, variables)
	if err != nil {
		if strings.Contains(err.Error(), "operation interrupted") && ctx.Err() != nil {
			return nil, details, fmt.Errorf("%w: %w", err, context.Cause(ctx))
		}
		return nil, details, err
	}

	attr, err := valToDeviceAttribute(result)
	if err != nil {
		return nil, details, err
	}
	return attr, details, nil
}

func valToDeviceAttribute(val ref.Val) (any, error) {
	switch v := val.Value().(type) {
	case bool:
		return v, nil
	case int64:
		return v, nil
	case string:
		return v, nil
	case semver.Version:
		return apiservercel.Semver{Version: v}, nil
	}

	if lister, ok := val.(traits.Lister); ok {
		sz, ok := lister.Size().Value().(int64)
		if !ok {
			return nil, fmt.Errorf("expected list size to be int64, got %T", lister.Size().Value())
		}

		// CEL list sizes are int64, but Go slice lengths and indices must be int.
		// Casting to int is safe (CEL lists won't practically exceed 32-bit int max)
		// and simplifies slice allocation and index-based iteration (e.g. vals[i]).
		// The CEL types.Int(i) cast would be required for lister.Get() regardless.
		length := int(sz)
		if length == 0 {
			return &resourceapi.DeviceAttribute{StringValues: []string{}}, nil
		}

		first := lister.Get(types.Int(0))
		switch first.Value().(type) {
		case bool:
			vals := make([]bool, length)
			for i := range length {
				elem := lister.Get(types.Int(i))
				b, ok := elem.Value().(bool)
				if !ok {
					return nil, fmt.Errorf("expected list element at index %d to be bool, got %T", i, elem.Value())
				}
				vals[i] = b
			}
			return vals, nil

		case int64:
			vals := make([]int64, length)
			for i := range length {
				elem := lister.Get(types.Int(i))
				v, ok := elem.Value().(int64)
				if !ok {
					return nil, fmt.Errorf("expected list element at index %d to be int, got %T", i, elem.Value())
				}
				vals[i] = v
			}
			return vals, nil

		case string:
			vals := make([]string, length)
			for i := range length {
				elem := lister.Get(types.Int(i))
				s, ok := elem.Value().(string)
				if !ok {
					return nil, fmt.Errorf("expected list element at index %d to be string, got %T", i, elem.Value())
				}
				vals[i] = s
			}
			return vals, nil

		case semver.Version:
			vals := make([]apiservercel.Semver, length)
			for i := range length {
				elem := lister.Get(types.Int(i))
				v, ok := elem.Value().(semver.Version)
				if !ok {
					return nil, fmt.Errorf("expected list element at index %d to be semver, got %T", i, elem.Value())
				}
				vals[i] = apiservercel.Semver{Version: v}
			}
			return vals, nil

		default:
			return nil, fmt.Errorf("unsupported list element type: %T", first.Value())
		}
	}

	return nil, fmt.Errorf("unsupported CEL return type: %T", val.Value())
}

func newCompiler(features Features) *compiler {
	envset := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())
	field := func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField {
		return apiservercel.NewDeclField(name, declType, required, nil, nil)
	}
	fields := func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField {
		result := make(map[string]*apiservercel.DeclField, len(fields))
		for _, f := range fields {
			result[f.Name] = f
		}
		return result
	}

	// declaredTypes collects all types that we declare for the environment.
	var declaredTypes []*apiservercel.DeclType
	declareType := func(t *apiservercel.DeclType) *apiservercel.DeclType {
		declaredTypes = append(declaredTypes, t)
		return t
	}

	attributeTypeV131 := declareType(withMaxElements(
		apiservercel.AnyType,
		resourceapi.DeviceAttributeMaxValueLength,
	))
	attributeTypeV136ListTypeAttributes := declareType(withMaxElements(
		// Use DynType so that iterate functions can work (e.g. exists, all)
		// for list type attributes.
		apiservercel.DynType,
		// At compile time we don't know whether an attribute will be a scalar or list.
		uint64(max(resourceapi.DeviceAttributeMaxValueLength, resourceapi.ResourceSliceMaxAttributeValuesPerDevice)),
	))
	// Each map is bound by the maximum number of different attributes.
	innerAttributesMapTypeV131 := declareType(apiservercel.NewMapType(idType, attributeTypeV131, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice))
	outerAttributesMapTypeV131 := declareType(apiservercel.NewMapType(domainType, innerAttributesMapTypeV131, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice))
	innerAttributesMapTypeV136ListTypeAttributes := declareType(apiservercel.NewMapType(
		idType, attributeTypeV136ListTypeAttributes, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice,
	))
	outerAttributesMapTypeV136ListTypeAttributes := declareType(apiservercel.NewMapType(
		domainType, innerAttributesMapTypeV136ListTypeAttributes, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice,
	))

	fieldsV131 := []*apiservercel.DeclField{
		field(driverVar, driverType, true),
		field(attributesVar, outerAttributesMapTypeV131, true),
		field(capacityVar, outerCapacityMapType, true),
	}
	deviceTypeV131 := declareType(apiservercel.NewObjectType("kubernetes.DRADevice", fields(fieldsV131...)))

	// One additional field, feature-gated below.
	fieldsV134ConsumableCapacity := []*apiservercel.DeclField{field(multiAllocVar, multiAllocType, true)}
	fieldsV134ConsumableCapacity = append(fieldsV134ConsumableCapacity, fieldsV131...)
	deviceTypeV134ConsumableCapacity := declareType(apiservercel.NewObjectType("kubernetes.DRADevice", fields(fieldsV134ConsumableCapacity...)))

	fieldsV136ListTypeAttributes := []*apiservercel.DeclField{
		field(driverVar, driverType, true),
		field(attributesVar, outerAttributesMapTypeV136ListTypeAttributes, true),
		field(capacityVar, outerCapacityMapType, true),
	}
	deviceTypeV136ListTypeAttributes := declareType(apiservercel.NewObjectType("kubernetes.DRADevice", fields(fieldsV136ListTypeAttributes...)))

	fieldsV136ConsumableCapacityListTypeAttributes := []*apiservercel.DeclField{field(multiAllocVar, multiAllocType, true)}
	fieldsV136ConsumableCapacityListTypeAttributes = append(fieldsV136ConsumableCapacityListTypeAttributes, fieldsV136ListTypeAttributes...)
	deviceTypeV136ConsumableCapacityListTypeAttributes := declareType(apiservercel.NewObjectType("kubernetes.DRADevice", fields(fieldsV136ConsumableCapacityListTypeAttributes...)))

	versioned := []environment.VersionedOptions{
		{
			IntroducedVersion: version.MajorMinor(1, 31),
			EnvOptions: []cel.EnvOption{
				// https://pkg.go.dev/github.com/google/cel-go/ext#Bindings
				//
				// This is useful to simplify attribute lookups because the
				// domain only needs to be given once:
				//
				//    cel.bind(dra, device.attributes["dra.example.com"], dra.oneBool && dra.anotherBool)
				ext.Bindings(ext.BindingsVersion(0)),
			},
		},
		// NewExpressions selects one of these device declarations with FeatureEnabled.
		// StoredExpressions ignores FeatureEnabled; the DeclTypeProvider keeps the
		// last declaration for a shared type name, so keep these ordered oldest to newest.
		{
			IntroducedVersion: version.MajorMinor(1, 31),
			FeatureEnabled: func() bool {
				return !features.EnableConsumableCapacity && !features.EnableListTypeAttributes
			},
			EnvOptions: []cel.EnvOption{
				cel.Variable(deviceVar, deviceTypeV131.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				deviceTypeV131,
			},
		},
		{
			IntroducedVersion: version.MajorMinor(1, 34),
			FeatureEnabled: func() bool {
				return features.EnableConsumableCapacity && !features.EnableListTypeAttributes
			},
			EnvOptions: []cel.EnvOption{
				cel.Variable(deviceVar, deviceTypeV134ConsumableCapacity.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				deviceTypeV134ConsumableCapacity,
			},
		},
		{
			// This type was added in 1.37, but 1.36 already behaved like this type
			// when ListTypeAttributes was enabled and ConsumableCapacity was disabled.
			IntroducedVersion: version.MajorMinor(1, 36),
			FeatureEnabled: func() bool {
				return !features.EnableConsumableCapacity && features.EnableListTypeAttributes
			},
			EnvOptions: []cel.EnvOption{
				cel.Variable(deviceVar, deviceTypeV136ListTypeAttributes.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				deviceTypeV136ListTypeAttributes,
			},
		},
		{
			// This type was added in 1.37, but 1.36 already behaved like this type
			// when both ListTypeAttributes and ConsumableCapacity was enabled.
			IntroducedVersion: version.MajorMinor(1, 36),
			FeatureEnabled: func() bool {
				return features.EnableConsumableCapacity && features.EnableListTypeAttributes
			},
			EnvOptions: []cel.EnvOption{
				cel.Variable(deviceVar, deviceTypeV136ConsumableCapacityListTypeAttributes.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				deviceTypeV136ConsumableCapacityListTypeAttributes,
			},
		},
	}
	envset, err := envset.Extend(versioned...)
	if err != nil {
		panic(fmt.Errorf("internal error building CEL environment: %w", err))
	}

	// Features can be still be changed before using the compiler.
	// The FeatureEnabled checks above ensure that the environment
	// is set up as necessary.
	//
	// Also, the compiler supports both stored and new expressions.
	//
	// Code which needs to know the actual type of items in the current
	// expression needs to look up the type from the active environment
	// (see deviceFieldTypeFromEnv above).
	return &compiler{
		envset:        envset,
		declaredTypes: declaredTypes,
		features:      features,
	}
}

func withMaxElements(in *apiservercel.DeclType, maxElements uint64) *apiservercel.DeclType {
	out := *in
	out.MaxElements = int64(maxElements)
	return &out
}

type domainToAttributes struct {
	typesFromEnv
	adapter             types.Adapter
	emptyMapValue       ref.Val
	lookupUniqueStrings func(string) draapi.UniqueString
	attrs               map[draapi.UniqueString]map[draapi.UniqueString]any
}

func (m *domainToAttributes) Find(key ref.Val) (ref.Val, bool) {
	strKey := key.ConvertToType(cel.StringType)
	if strKey.Type() != cel.StringType {
		return strKey, false
	}
	// This either returns the unique key for lookup in the device
	// attributes or NullUniqueKey, which isn't going to be found.
	// In that case we continue with returning the default value below,
	// without actually converting the key (would need locking).
	uniqueKey := m.lookupUniqueStrings(strKey.Value().(string))
	value, found := m.attrs[uniqueKey]
	if found {
		return &nameToAttributes{
			typesFromEnv:         m.typesFromEnv,
			adapter:              m.adapter,
			makeUniqueAttrString: m.lookupUniqueStrings,
			attrs:                value,
		}, true
	}

	return m.emptyMapValue, true
}

func (m *domainToAttributes) ConvertToNative(typeDesc reflect.Type) (any, error) {
	// This shouldn't be needed for evaluating CEL expressions.
	return nil, errors.New("not implemented")

}

func (m *domainToAttributes) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case types.MapType:
		return m
	case types.TypeType:
		return m.outerAttributesType.CelType()
	}
	return types.NewErr("type conversion error from '%s' to '%s'", m.outerAttributesType, typeVal)
}

func (m *domainToAttributes) Equal(other ref.Val) ref.Val {
	otherMap, ok := other.(traits.Mapper)
	if !ok {
		return types.False
	}
	return equalMaps(m, otherMap)
}

func (m *domainToAttributes) Type() ref.Type { return m.outerAttributesType }

func (m *domainToAttributes) Value() any {
	return m.attrs
}

func (m *domainToAttributes) Contains(key ref.Val) ref.Val {
	val, found := m.Find(key)
	if val == m.emptyMapValue {
		// Not really, it was the default.
		found = false
	}
	return types.Bool(found)
}

func (m *domainToAttributes) Get(key ref.Val) ref.Val {
	val, found := m.Find(key)
	if !found {
		return types.ValOrErr(val, "no such key: %v", key)
	}
	return val
}

func (m *domainToAttributes) Iterator() traits.Iterator {
	return &mapIterator{
		mapKeys: reflect.ValueOf(m.attrs).MapRange(),
		len:     len(m.attrs),
	}
}

func (m *domainToAttributes) Size() ref.Val {
	return types.Int(len(m.attrs))
}

type nameToAttributes struct {
	typesFromEnv
	adapter              types.Adapter
	makeUniqueAttrString func(string) draapi.UniqueString
	attrs                map[draapi.UniqueString]any
}

func (m *nameToAttributes) Find(key ref.Val) (ref.Val, bool) {
	strKey := key.ConvertToType(cel.StringType)
	if strKey.Type() != cel.StringType {
		return strKey, false
	}
	uniqueKey := m.makeUniqueAttrString(strKey.Value().(string))
	value, found := m.attrs[uniqueKey]
	if !found {
		return nil, false
	}

	switch value := value.(type) {
	case int64:
		return types.Int(value), true
	case bool:
		return types.Bool(value), true
	case string:
		return types.String(value), true
	case apiservercel.Semver:
		return value, true
	case []int64:
		return types.NewDynamicList(m.adapter, value), true
	case []bool:
		return types.NewDynamicList(m.adapter, value), true
	case []string:
		return types.NewStringList(m.adapter, value), true
	case []apiservercel.Semver:
		return types.NewDynamicList(m.adapter, value), true
	default:
		return types.NewErr("internal error: missing support for value type %T", value), false
	}
}

func (m *nameToAttributes) ConvertToNative(typeDesc reflect.Type) (any, error) {
	// This shouldn't be needed for evaluating CEL expressions.
	return nil, errors.New("not implemented")

}

func (m *nameToAttributes) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case types.MapType:
		return m
	case types.TypeType:
		return m.innerAttributesType.CelType()
	}
	return types.NewErr("type conversion error from '%s' to '%s'", m.innerAttributesType, typeVal)
}

func (m *nameToAttributes) Equal(other ref.Val) ref.Val {
	otherMap, ok := other.(traits.Mapper)
	if !ok {
		return types.False
	}
	return equalMaps(m, otherMap)
}

func (m *nameToAttributes) Type() ref.Type { return m.innerAttributesType }

func (m *nameToAttributes) Value() any {
	return m.attrs
}

func (m *nameToAttributes) Contains(key ref.Val) ref.Val {
	_, found := m.Find(key)
	return types.Bool(found)
}

func (m *nameToAttributes) Get(key ref.Val) ref.Val {
	val, found := m.Find(key)
	if !found {
		return types.ValOrErr(val, "no such key: %v", key)
	}
	return val
}

func (m *nameToAttributes) Iterator() traits.Iterator {
	return &mapIterator{
		mapKeys: reflect.ValueOf(m.attrs).MapRange(),
		len:     len(m.attrs),
	}
}

func (m *nameToAttributes) Size() ref.Val {
	return types.Int(len(m.attrs))
}

// domainToCapacity is a CEL map proxy for device.capacity[<domain>].
// Lookup by domain returns a nameToCapacity for the inner name→Quantity map.
// Lookup of an unknown domain returns emptyMapValue (an empty map).
type domainToCapacity struct {
	lookupUniqueStrings func(string) draapi.UniqueString
	emptyMapValue       ref.Val
	caps                map[draapi.UniqueString]map[draapi.UniqueString]draapi.DeviceCapacity
}

func (m *domainToCapacity) Find(key ref.Val) (ref.Val, bool) {
	strKey := key.ConvertToType(cel.StringType)
	if strKey.Type() != cel.StringType {
		return strKey, false
	}
	uniqueKey := m.lookupUniqueStrings(strKey.Value().(string))
	inner, found := m.caps[uniqueKey]
	if found {
		return &nameToCapacity{
			lookupUniqueStrings: m.lookupUniqueStrings,
			caps:                inner,
		}, true
	}
	return m.emptyMapValue, true
}

func (m *domainToCapacity) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, errors.New("not implemented")
}

func (m *domainToCapacity) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case types.MapType:
		return m
	case types.TypeType:
		return outerCapacityMapType.CelType()
	}
	return types.NewErr("type conversion error from '%s' to '%s'", outerCapacityMapType, typeVal)
}

func (m *domainToCapacity) Equal(other ref.Val) ref.Val {
	otherMap, ok := other.(traits.Mapper)
	if !ok {
		return types.False
	}
	return equalMaps(m, otherMap)
}

func (m *domainToCapacity) Type() ref.Type { return outerCapacityMapType }

func (m *domainToCapacity) Value() any { return m.caps }

func (m *domainToCapacity) Contains(key ref.Val) ref.Val {
	val, found := m.Find(key)
	if val == m.emptyMapValue {
		found = false
	}
	return types.Bool(found)
}

func (m *domainToCapacity) Get(key ref.Val) ref.Val {
	val, found := m.Find(key)
	if !found {
		return types.ValOrErr(val, "no such key: %v", key)
	}
	return val
}

func (m *domainToCapacity) Iterator() traits.Iterator {
	return &mapIterator{
		mapKeys: reflect.ValueOf(m.caps).MapRange(),
		len:     len(m.caps),
	}
}

func (m *domainToCapacity) Size() ref.Val { return types.Int(len(m.caps)) }

// nameToCapacity is a CEL map proxy for device.capacity[<domain>][<name>].
type nameToCapacity struct {
	lookupUniqueStrings func(string) draapi.UniqueString
	caps                map[draapi.UniqueString]draapi.DeviceCapacity
}

func (m *nameToCapacity) Find(key ref.Val) (ref.Val, bool) {
	strKey := key.ConvertToType(cel.StringType)
	if strKey.Type() != cel.StringType {
		return strKey, false
	}
	uniqueKey := m.lookupUniqueStrings(strKey.Value().(string))
	cap, found := m.caps[uniqueKey]
	if !found {
		return nil, false
	}
	return cap.Value, true
}

func (m *nameToCapacity) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, errors.New("not implemented")
}

func (m *nameToCapacity) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case types.MapType:
		return m
	case types.TypeType:
		return innerCapacityMapType.CelType()
	}
	return types.NewErr("type conversion error from '%s' to '%s'", innerCapacityMapType, typeVal)
}

func (m *nameToCapacity) Equal(other ref.Val) ref.Val {
	otherMap, ok := other.(traits.Mapper)
	if !ok {
		return types.False
	}
	return equalMaps(m, otherMap)
}

func (m *nameToCapacity) Type() ref.Type { return innerCapacityMapType }

func (m *nameToCapacity) Value() any { return m.caps }

func (m *nameToCapacity) Contains(key ref.Val) ref.Val {
	_, found := m.Find(key)
	return types.Bool(found)
}

func (m *nameToCapacity) Get(key ref.Val) ref.Val {
	val, found := m.Find(key)
	if !found {
		return types.ValOrErr(val, "no such key: %v", key)
	}
	return val
}

func (m *nameToCapacity) Iterator() traits.Iterator {
	return &mapIterator{
		mapKeys: reflect.ValueOf(m.caps).MapRange(),
		len:     len(m.caps),
	}
}

func (m *nameToCapacity) Size() ref.Val { return types.Int(len(m.caps)) }

// sizeEstimator tells the cost estimator the maximum size of maps, strings, or lists accessible through the `device` variable.
// Without this, the maximum string size of e.g. `device.attributes["dra.example.com"].services` would be unknown.
//
// sizeEstimator is derived from the sizeEstimator in k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel.
type sizeEstimator struct {
	typesFromEnv
}

func (s *sizeEstimator) EstimateSize(element checker.AstNode) (res *checker.SizeEstimate) {
	path := element.Path()
	if len(path) == 0 {
		// Path() can return an empty list, early exit if it does since we can't
		// provide size estimates when that happens
		return nil
	}

	// The estimator provides information about the environment's variable(s).
	var currentNode *apiservercel.DeclType
	switch path[0] {
	case deviceVar:
		currentNode = s.deviceType
	default:
		// Unknown root, shouldn't happen.
		return nil
	}

	// Cut off initial variable from path, it was checked above.
	for _, name := range path[1:] {
		switch name {
		case "@items", "@values":
			if currentNode.ElemType == nil {
				return nil
			}
			currentNode = currentNode.ElemType
		case "@keys":
			if currentNode.KeyType == nil {
				return nil
			}
			currentNode = currentNode.KeyType
		default:
			field, ok := currentNode.Fields[name]
			if !ok {
				// If this is an attribute map, then we know that all elements
				// have the same maximum size as set in attributeType, regardless
				// of their name.
				if currentNode.ElemType == s.attributeType {
					currentNode = s.attributeType
					continue
				}
				return nil
			}
			if field.Type == nil {
				return nil
			}
			currentNode = field.Type
		}
	}
	return &checker.SizeEstimate{Min: 0, Max: uint64(currentNode.MaxElements)}
}

func (s *sizeEstimator) EstimateCallCost(function, overloadID string, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	return nil
}

// validateOutputType checks that the CEL expression output type t is valid.
// For derived attributes, it must evaluate to a primitive scalar or a list
// of those scalars. For device selector expressions, it must evaluate to a
// boolean or match the attribute type. The types parameter should contain
// the pre-calculated types from the CEL environment.
func (c compiler) validateOutputType(types typesFromEnv, t *cel.Type, envType environment.Type, isDerivedAttribute bool) *apiservercel.Error {
	// Scenario: Derived attributes.
	// They must evaluate to primitive scalars or lists of those scalars to
	// ensure they can be matched by the scheduler using basic equality or
	// disjointness comparisons.
	if isDerivedAttribute {
		// For stored expressions, we always allow 'lists' to prevent breaking
		// existing objects if the feature gate is disabled after previously
		// being enabled.
		allowListType := c.features.EnableListTypeAttributes || envType == environment.StoredExpressions
		if c.isAllowedDerivedAttributeType(t, allowListType) {
			return nil
		}
		detail := "must evaluate to a primitive scalar (string, integer, boolean, semver), not %v"
		if allowListType {
			detail = "must evaluate to a primitive scalar (string, integer, boolean, semver) or a list of these scalars, not %v"
		}
		return &apiservercel.Error{
			Type:   apiservercel.ErrorTypeInvalid,
			Detail: fmt.Sprintf(detail, t.String()),
		}
	}

	// Scenario: Device selector expressions.

	expectedReturnType := cel.BoolType
	if t.IsExactType(expectedReturnType) || t.IsExactType(types.attributeType.CelType()) {
		return nil
	}
	return &apiservercel.Error{
		Type:   apiservercel.ErrorTypeInvalid,
		Detail: fmt.Sprintf("must evaluate to %v or the unknown type, not %v", expectedReturnType.String(), t.String()),
	}
}

func (c compiler) isAllowedDerivedAttributeType(t *cel.Type, allowListType bool) bool {
	if c.isAllowedDerivedAttributeElementType(t) {
		return true
	}
	// A list in CEL is a unary generic type parameterized by exactly 1
	// type parameter (the element type). We check that the length is 1
	// to ensure it is a valid list and to safely access the element type.
	if allowListType && t.TypeName() == "list" && len(t.Parameters()) == 1 {
		return c.isAllowedDerivedAttributeElementType(t.Parameters()[0])
	}
	return false
}

func (c compiler) isAllowedDerivedAttributeElementType(t *cel.Type) bool {
	return t.IsExactType(cel.BoolType) ||
		t.IsExactType(cel.IntType) ||
		t.IsExactType(cel.StringType) ||
		t.IsExactType(cel.AnyType) ||
		t.IsExactType(cel.DynType) ||
		t.IsExactType(apiservercel.SemverType)
}

func equalMaps(a, b traits.Mapper) ref.Val {
	if a.Size() != b.Size() {
		return types.False
	}
	it := a.Iterator()
	for it.HasNext() == types.True {
		key := it.Next()
		thisVal, _ := a.Find(key)
		otherVal, found := b.Find(key)
		if !found {
			return types.False
		}
		valEq := types.Equal(thisVal, otherVal)
		if valEq == types.False {
			return types.False
		}
	}
	return types.True
}

// mapIterator iterates over a Go map with draapi.UniqueString as keys via reflection.
// It's based on baseIterator and mapIterator:
// - https://github.com/cel-expr/cel-go/blob/a4d0d643deeea6408654de2ec9944895d23f36a5/common/types/iterator.go#L30-L55
// - https://github.com/cel-expr/cel-go/blob/a82c68b770ac0cb67f7b4f76166827c14b145eb8/common/types/map.go#L922-L943
type mapIterator struct {
	mapKeys *reflect.MapIter
	cursor  int
	len     int
}

func (*mapIterator) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, fmt.Errorf("type conversion on iterators not supported")
}

func (*mapIterator) ConvertToType(typeVal ref.Type) ref.Val {
	return types.NewErr("no such overload")
}

func (*mapIterator) Equal(other ref.Val) ref.Val {
	return types.NewErr("no such overload")
}

func (*mapIterator) Type() ref.Type {
	return types.IteratorType
}

func (*mapIterator) Value() any {
	return nil
}

func (it *mapIterator) HasNext() ref.Val {
	return types.Bool(it.cursor < it.len)
}

func (it *mapIterator) Next() ref.Val {
	if it.HasNext() == types.True && it.mapKeys.Next() {
		it.cursor++
		refKey := it.mapKeys.Key()
		return types.String(refKey.Interface().(draapi.UniqueString).String())
	}
	return nil
}
