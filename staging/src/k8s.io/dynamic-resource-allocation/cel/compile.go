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
	"strings"
	"sync"

	"github.com/blang/semver/v4"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/ext"

	"k8s.io/utils/ptr"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
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

	emptyMapVal ref.Val

	features Features
}

// Device defines the input values for a CEL selector expression.
type Device struct {
	// Driver gets used as domain for any attribute which does not already
	// have a domain prefix. If set, then it is also made available as a
	// string attribute.
	Driver                   string
	AllowMultipleAllocations *bool
	Attributes               map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
	Capacity                 map[resourceapi.QualifiedName]resourceapi.DeviceCapacity
}

type compiler struct {
	// deviceType is a definition for the latest type of the `device` variable.
	// This is needed for the cost estimator.
	// If that ever changes such as involving type-checking expressions,
	// some additional logic might be needed to make
	// cost estimates version-dependent.
	deviceType *apiservercel.DeclType
	envset     *environment.EnvSet

	// A variant of AnyType = https://github.com/kubernetes/kubernetes/blob/ec2e0de35a298363872897e5904501b029817af3/staging/src/k8s.io/apiserver/pkg/cel/types.go#L550:
	// unknown actual type (could be bool, int, string, etc.) but with a known maximum size.
	// If DRAListTypeAttributes is enabled, this also applies to list attributes, so that the cost estimator can take into account the size of the list.
	attributeType *apiservercel.DeclType

	features Features
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
}

// CompileCELExpression returns a compiled CEL expression. It evaluates to bool.
//
// TODO (https://github.com/kubernetes/kubernetes/issues/125826): validate AST to detect invalid attribute names.
func (c compiler) CompileCELExpression(expression string, options Options) CompilationResult {
	resultError := func(errorString string, errType apiservercel.ErrorType) CompilationResult {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   errType,
				Detail: errorString,
			},
			Expression: expression,
			MaxCost:    math.MaxUint64,
			features:   c.features,
		}
	}

	env, err := c.envset.Env(ptr.Deref(options.EnvType, environment.StoredExpressions))
	if err != nil {
		return resultError(fmt.Sprintf("unexpected error loading CEL environment: %v", err), apiservercel.ErrorTypeInternal)
	}

	ast, issues := env.Compile(expression)
	if issues != nil {
		return resultError("compilation failed: "+issues.String(), apiservercel.ErrorTypeInvalid)
	}

	// When the DRAListTypeAttributes feature is enabled,
	// we use DynType instead of AnyType for the attributes
	// so that standard iterate functions(e.g., exists, all etc.)
	// and overridden includes function can work (See newCompiler() for details).
	// Thus, the unknown return type can also be DynType, not just AnyType as before.
	//
	// This has to be valid because the end result of a CEL expression might be
	// a boolean attribute, which then has AnyType/DynType.
	expectedReturnType := cel.BoolType
	if ast.OutputType() == expectedReturnType ||
		ast.OutputType() == cel.AnyType ||
		c.features.EnableListTypeAttributes && ast.OutputType() == cel.DynType {
		// Okay, is one of the acceptable types.
	} else {
		return resultError(fmt.Sprintf("must evaluate to %v or the unknown type, not %v", expectedReturnType.String(), ast.OutputType().String()), apiservercel.ErrorTypeInvalid)
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
		Program:     prog,
		Expression:  expression,
		OutputType:  ast.OutputType(),
		Environment: env,
		emptyMapVal: env.CELTypeAdapter().NativeToValue(map[string]any{}),
		MaxCost:     math.MaxUint64,
		features:    c.features,
	}

	if !options.DisableCostEstimation {
		// We don't have a SizeEstimator. The potential size of the input (= a
		// device) is already declared in the definition of the environment.
		estimator := c.newCostEstimator()
		costEst, err := env.EstimateCost(ast, estimator)
		if err != nil {
			compilationResult.Error = &apiservercel.Error{Type: apiservercel.ErrorTypeInternal, Detail: "cost estimation failed: " + err.Error()}
			return compilationResult
		}
		compilationResult.MaxCost = costEst.Max
	}

	return compilationResult
}

func (c *compiler) newCostEstimator() checker.CostEstimator {
	base := &library.CostEstimator{SizeEstimator: &sizeEstimator{compiler: c}}
	return &draCostEstimator{base: base}
}

// getAttributeValue returns the native representation of the one value that
// should be stored in the attribute, otherwise an error. An error is
// also returned when there is no supported value.
//
// If the DRAListTypeAttributes feature  is disabled and an attribute
// contains a list value, then we fall through to returning the
// "unsupported attribute value" error below.
// This failure then aborts scheduling until ResourceSlices get updated.
// This is better than incorrectly scheduling a pod
// because that is harder to correct.
func (c CompilationResult) getAttributeValue(attr resourceapi.DeviceAttribute) (any, error) {
	if c.features.EnableListTypeAttributes {
		switch {
		case attr.IntValues != nil:
			return attr.IntValues, nil
		case attr.BoolValues != nil:
			return attr.BoolValues, nil
		case attr.StringValues != nil:
			return attr.StringValues, nil
		case attr.VersionValues != nil:
			semVers := make([]apiservercel.Semver, len(attr.VersionValues))
			for i, versionStr := range attr.VersionValues {
				v, err := semver.Parse(versionStr)
				if err != nil {
					return nil, fmt.Errorf("parse semantic version: %w", err)
				}
				semVers[i] = apiservercel.Semver{Version: v}
			}
			return semVers, nil
		}
	}
	switch {
	case attr.IntValue != nil:
		return *attr.IntValue, nil
	case attr.BoolValue != nil:
		return *attr.BoolValue, nil
	case attr.StringValue != nil:
		return *attr.StringValue, nil
	case attr.VersionValue != nil:
		v, err := semver.Parse(*attr.VersionValue)
		if err != nil {
			return nil, fmt.Errorf("parse semantic version: %w", err)
		}
		return apiservercel.Semver{Version: v}, nil
	default:
		return nil, errors.New("unsupported attribute value")
	}
}

var boolType = reflect.TypeOf(true)

func (c CompilationResult) DeviceMatches(ctx context.Context, input Device) (bool, *cel.EvalDetails, error) {
	// TODO (future): avoid building these maps and instead use a proxy
	// which wraps the underlying maps and directly looks up values.
	attributes := make(map[string]any)
	for name, attr := range input.Attributes {
		value, err := c.getAttributeValue(attr)
		if err != nil {
			return false, nil, fmt.Errorf("attribute %s: %w", name, err)
		}
		domain, id := parseQualifiedName(name, input.Driver)
		if attributes[domain] == nil {
			attributes[domain] = make(map[string]any)
		}
		attributes[domain].(map[string]any)[id] = value
	}

	capacity := make(map[string]any)
	for name, cap := range input.Capacity {
		domain, id := parseQualifiedName(name, input.Driver)
		if capacity[domain] == nil {
			capacity[domain] = make(map[string]apiservercel.Quantity)
		}
		capacity[domain].(map[string]apiservercel.Quantity)[id] = apiservercel.Quantity{Quantity: &cap.Value}
	}

	variables := map[string]any{
		deviceVar: map[string]any{
			driverVar:     input.Driver,
			multiAllocVar: ptr.Deref(input.AllowMultipleAllocations, false),
			attributesVar: newStringInterfaceMapWithDefault(c.Environment.CELTypeAdapter(), attributes, c.emptyMapVal),
			capacityVar:   newStringInterfaceMapWithDefault(c.Environment.CELTypeAdapter(), capacity, c.emptyMapVal),
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

	attributeType := withMaxElements(apiservercel.AnyType, resourceapi.DeviceAttributeMaxValueLength)
	if features.EnableListTypeAttributes {
		attributeType = withMaxElements(
			// use DynType instead of AnyType so that iterate functions can work(e.g., exists, all, etc.)
			apiservercel.DynType,
			// When DRAListTypeAttributes feature gate is enabled, we cannot
			// determine statically whether an attribute will be a scalar or a list
			// at compile time. MaxElements should describe
			// - the max number of items when it's a list.
			// - the max length of the string representation when it's a scalar.
			// Thus, we set it to the larger one of the two cases.
			uint64(max(resourceapi.DeviceAttributeMaxValueLength, resourceapi.ResourceSliceMaxAttributeValuesPerDevice)),
		)
	}
	// Each map is bound by the maximum number of different attributes.
	innerAttributesMapType := apiservercel.NewMapType(idType, attributeType, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice)
	outerAttributesMapType := apiservercel.NewMapType(domainType, innerAttributesMapType, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice)

	fieldsV131 := []*apiservercel.DeclField{
		field(driverVar, driverType, true),
		field(attributesVar, outerAttributesMapType, true),
		field(capacityVar, outerCapacityMapType, true),
	}
	deviceTypeV131 := apiservercel.NewObjectType("kubernetes.DRADevice", fields(fieldsV131...))

	// One additional field, feature-gated below.
	fieldsV134ConsumableCapacity := []*apiservercel.DeclField{field(multiAllocVar, multiAllocType, true)}
	fieldsV134ConsumableCapacity = append(fieldsV134ConsumableCapacity, fieldsV131...)
	deviceTypeV134ConsumableCapacity := apiservercel.NewObjectType("kubernetes.DRADevice", fields(fieldsV134ConsumableCapacity...))

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
		// deviceTypeV131 and deviceTypeV134ConsumableCapacity are complimentary and picked
		// based on the feature gate.
		{
			IntroducedVersion: version.MajorMinor(1, 31),
			FeatureEnabled: func() bool {
				return !features.EnableConsumableCapacity
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
				return features.EnableConsumableCapacity
			},
			EnvOptions: []cel.EnvOption{
				cel.Variable(deviceVar, deviceTypeV134ConsumableCapacity.CelType()),
			},
			DeclTypes: []*apiservercel.DeclType{
				deviceTypeV134ConsumableCapacity,
			},
		},
		{
			IntroducedVersion: version.MajorMinor(1, 36),
			FeatureEnabled: func() bool {
				return features.EnableListTypeAttributes
			},
			EnvOptions: []cel.EnvOption{
				cel.Function("includes",
					cel.MemberOverload("dra_includes_dyn_dyn",
						[]*cel.Type{cel.DynType, cel.DynType},
						cel.BoolType,
						cel.BinaryBinding(includesFunc),
					),
				),
			},
		},
	}
	envset, err := envset.Extend(versioned...)
	if err != nil {
		panic(fmt.Errorf("internal error building CEL environment: %w", err))
	}
	// return with newest deviceType
	return &compiler{envset: envset, deviceType: deviceTypeV134ConsumableCapacity, features: features, attributeType: attributeType}
}

// includesFunc implements the "includes" function for CEL (<target>.includes(<arg>)),
// which checks whether the target includes the argument.
// It supports both singular values and lists.
//
// WARNING: includes is not applicable to lists longer than
// resourceapi.ResourceSliceMaxAttributeValuesPerDevice. A runtime error gets
// returned in that case to prevent unbounded execution cost.
// The target is expected to be a scalar or list attribute, but users can
// technically call "includes" on any value.
func includesFunc(target, arg ref.Val) ref.Val {
	if list, ok := target.(traits.Lister); ok {
		i := 0
		it := list.Iterator()
		for it.HasNext() == types.True {
			if i >= resourceapi.ResourceSliceMaxAttributeValuesPerDevice {
				return types.NewErr("'includes' function cannot be applied to lists longer than %d values", resourceapi.ResourceSliceMaxAttributeValuesPerDevice)
			}
			item := it.Next()
			if item.Equal(arg) == types.True {
				return types.True
			}
			i++
		}
		return types.False
	}

	if target.Equal(arg) == types.True {
		return types.True
	}
	return types.False
}

func withMaxElements(in *apiservercel.DeclType, maxElements uint64) *apiservercel.DeclType {
	out := *in
	out.MaxElements = int64(maxElements)
	return &out
}

// parseQualifiedName splits into domain and identified, using the default domain
// if the name does not contain one.
func parseQualifiedName(name resourceapi.QualifiedName, defaultDomain string) (string, string) {
	sep := strings.Index(string(name), "/")
	if sep == -1 {
		return defaultDomain, string(name)
	}
	return string(name[0:sep]), string(name[sep+1:])
}

// newStringInterfaceMapWithDefault is like
// https://pkg.go.dev/github.com/google/cel-go@v0.20.1/common/types#NewStringInterfaceMap,
// except that looking up an unknown key returns a default value.
func newStringInterfaceMapWithDefault(adapter types.Adapter, value map[string]any, defaultValue ref.Val) traits.Mapper {
	return mapper{
		Mapper:       types.NewStringInterfaceMap(adapter, value),
		defaultValue: defaultValue,
	}
}

type mapper struct {
	traits.Mapper
	defaultValue ref.Val
}

// Find wraps the mapper's Find so that a default empty map is returned when
// the lookup did not find the entry.
func (m mapper) Find(key ref.Val) (ref.Val, bool) {
	value, found := m.Mapper.Find(key)
	if found {
		return value, true
	}

	return m.defaultValue, true
}

// draCostEstimator is a wrapper around the base CEL CostEstimator to provide custom cost estimates for DRA-specific functions and types.
type draCostEstimator struct {
	base *library.CostEstimator
}

func (e *draCostEstimator) EstimateSize(element checker.AstNode) *checker.SizeEstimate {
	return e.base.EstimateSize(element)
}

func (e *draCostEstimator) EstimateCallCost(function, overloadID string, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	if function == "includes" && overloadID == "dra_includes_dyn_dyn" {
		// "<target>.includes(<arg>)" is equivalent with "<arg> in <target>"
		// whose complexity is linear with the size of the target.
		if target != nil {
			targetSizeEstimate := checker.SizeEstimate{Min: 0, Max: resourceapi.ResourceSliceMaxAttributeValuesPerDevice}
			return &checker.CallEstimate{CostEstimate: targetSizeEstimate.MultiplyByCost(checker.CostEstimate{Min: 1, Max: 1})}
		}
	}

	return e.base.EstimateCallCost(function, overloadID, target, args)
}

// sizeEstimator tells the cost estimator the maximum size of maps, strings, or lists accessible through the `device` variable.
// Without this, the maximum string size of e.g. `device.attributes["dra.example.com"].services` would be unknown.
//
// sizeEstimator is derived from the sizeEstimator in k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel.
type sizeEstimator struct {
	compiler *compiler
}

func (s *sizeEstimator) EstimateSize(element checker.AstNode) *checker.SizeEstimate {
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
		currentNode = s.compiler.deviceType
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
				if currentNode.ElemType == s.compiler.attributeType {
					currentNode = s.compiler.attributeType
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
