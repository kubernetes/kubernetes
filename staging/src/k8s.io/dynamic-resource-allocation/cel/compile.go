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
	"reflect"
	"strings"
	"sync"

	"github.com/blang/semver/v4"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/ext"

	resourceapi "k8s.io/api/resource/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
)

const (
	deviceVar     = "device"
	driverVar     = "driver"
	attributesVar = "attributes"
	capacityVar   = "capacity"
)

var (
	lazyCompilerInit sync.Once
	lazyCompiler     *compiler
)

func GetCompiler() *compiler {
	lazyCompilerInit.Do(func() {
		lazyCompiler = newCompiler()
	})
	return lazyCompiler
}

// CompilationResult represents a compiled expression.
type CompilationResult struct {
	Program     cel.Program
	Error       *apiservercel.Error
	Expression  string
	OutputType  *cel.Type
	Environment *cel.Env

	emptyMapVal ref.Val
}

// Device defines the input values for a CEL selector expression.
type Device struct {
	// Driver gets used as domain for any attribute which does not already
	// have a domain prefix. If set, then it is also made available as a
	// string attribute.
	Driver     string
	Attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
	Capacity   map[resourceapi.QualifiedName]resource.Quantity
}

type compiler struct {
	envset *environment.EnvSet
}

func newCompiler() *compiler {
	return &compiler{envset: mustBuildEnv()}
}

// CompileCELExpression returns a compiled CEL expression. It evaluates to bool.
//
// TODO (https://github.com/kubernetes/kubernetes/issues/125826): validate AST to detect invalid attribute names.
func (c compiler) CompileCELExpression(expression string, envType environment.Type) CompilationResult {
	resultError := func(errorString string, errType apiservercel.ErrorType) CompilationResult {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   errType,
				Detail: errorString,
			},
			Expression: expression,
		}
	}

	env, err := c.envset.Env(envType)
	if err != nil {
		return resultError(fmt.Sprintf("unexpected error loading CEL environment: %v", err), apiservercel.ErrorTypeInternal)
	}

	ast, issues := env.Compile(expression)
	if issues != nil {
		return resultError("compilation failed: "+issues.String(), apiservercel.ErrorTypeInvalid)
	}
	expectedReturnType := cel.BoolType
	if ast.OutputType() != expectedReturnType &&
		ast.OutputType() != cel.AnyType {
		return resultError(fmt.Sprintf("must evaluate to %v or the unknown type, not %v", expectedReturnType.String(), ast.OutputType().String()), apiservercel.ErrorTypeInvalid)
	}
	_, err = cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	prog, err := env.Program(ast,
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
	if err != nil {
		return resultError("program instantiation failed: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	return CompilationResult{
		Program:     prog,
		Expression:  expression,
		OutputType:  ast.OutputType(),
		Environment: env,
		emptyMapVal: env.CELTypeAdapter().NativeToValue(map[string]any{}),
	}
}

// getAttributeValue returns the native representation of the one value that
// should be stored in the attribute, otherwise an error. An error is
// also returned when there is no supported value.
func getAttributeValue(attr resourceapi.DeviceAttribute) (any, error) {
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

func (c CompilationResult) DeviceMatches(ctx context.Context, input Device) (bool, error) {
	// TODO (future): avoid building these maps and instead use a proxy
	// which wraps the underlying maps and directly looks up values.
	attributes := make(map[string]any)
	for name, attr := range input.Attributes {
		value, err := getAttributeValue(attr)
		if err != nil {
			return false, fmt.Errorf("attribute %s: %w", name, err)
		}
		domain, id := parseQualifiedName(name, input.Driver)
		if attributes[domain] == nil {
			attributes[domain] = make(map[string]any)
		}
		attributes[domain].(map[string]any)[id] = value
	}

	capacity := make(map[string]any)
	for name, quantity := range input.Capacity {
		domain, id := parseQualifiedName(name, input.Driver)
		if capacity[domain] == nil {
			capacity[domain] = make(map[string]apiservercel.Quantity)
		}
		capacity[domain].(map[string]apiservercel.Quantity)[id] = apiservercel.Quantity{Quantity: &quantity}
	}

	variables := map[string]any{
		deviceVar: map[string]any{
			driverVar:     input.Driver,
			attributesVar: newStringInterfaceMapWithDefault(c.Environment.CELTypeAdapter(), attributes, c.emptyMapVal),
			capacityVar:   newStringInterfaceMapWithDefault(c.Environment.CELTypeAdapter(), capacity, c.emptyMapVal),
		},
	}

	result, _, err := c.Program.ContextEval(ctx, variables)
	if err != nil {
		return false, err
	}
	resultAny, err := result.ConvertToNative(boolType)
	if err != nil {
		return false, fmt.Errorf("CEL result of type %s could not be converted to bool: %w", result.Type().TypeName(), err)
	}
	resultBool, ok := resultAny.(bool)
	if !ok {
		return false, fmt.Errorf("CEL native result value should have been a bool, got instead: %T", resultAny)
	}
	return resultBool, nil
}

func mustBuildEnv() *environment.EnvSet {
	envset := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), false /* strictCost */)
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
	deviceType := apiservercel.NewObjectType("kubernetes.DRADevice", fields(
		field(driverVar, apiservercel.StringType, true),
		field(attributesVar, apiservercel.NewMapType(apiservercel.StringType, apiservercel.NewMapType(apiservercel.StringType, apiservercel.AnyType, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice), true),
		field(capacityVar, apiservercel.NewMapType(apiservercel.StringType, apiservercel.NewMapType(apiservercel.StringType, apiservercel.QuantityDeclType, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice), resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice), true),
	))

	versioned := []environment.VersionedOptions{
		{
			// Feature epoch was actually 1.31, but we artificially set it to 1.0 because these
			// options should always be present.
			//
			// TODO (https://github.com/kubernetes/kubernetes/issues/123687): set this
			// version properly before going to beta.
			IntroducedVersion: version.MajorMinor(1, 0),
			EnvOptions: []cel.EnvOption{
				cel.Variable(deviceVar, deviceType.CelType()),

				library.SemverLib(),

				// https://pkg.go.dev/github.com/google/cel-go/ext#Bindings
				//
				// This is useful to simplify attribute lookups because the
				// domain only needs to be given once:
				//
				//    cel.bind(dra, device.attributes["dra.example.com"], dra.oneBool && dra.anotherBool)
				ext.Bindings(),
			},
			DeclTypes: []*apiservercel.DeclType{
				deviceType,
			},
		},
	}
	envset, err := envset.Extend(versioned...)
	if err != nil {
		panic(fmt.Errorf("internal error building CEL environment: %w", err))
	}
	return envset
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
