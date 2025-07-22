// Copyright 2019 Google LLC
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
	"errors"
	"fmt"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/dynamicpb"

	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/env"
	"github.com/google/cel-go/common/functions"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/pb"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	descpb "google.golang.org/protobuf/types/descriptorpb"
)

// These constants beginning with "Feature" enable optional behavior in
// the library.  See the documentation for each constant to see its
// effects, compatibility restrictions, and standard conformance.
const (
	_ = iota

	// Enable the tracking of function call expressions replaced by macros.
	featureEnableMacroCallTracking

	// Enable the use of cross-type numeric comparisons at the type-checker.
	featureCrossTypeNumericComparisons

	// Enable eager validation of declarations to ensure that Env values created
	// with `Extend` inherit a validated list of declarations from the parent Env.
	featureEagerlyValidateDeclarations

	// Enable the use of the default UTC timezone when a timezone is not specified
	// on a CEL timestamp operation. This fixes the scenario where the input time
	// is not already in UTC.
	featureDefaultUTCTimeZone

	// Enable the serialization of logical operator ASTs as variadic calls, thus
	// compressing the logic graph to a single call when multiple like-operator
	// expressions occur: e.g. a && b && c && d -> call(_&&_, [a, b, c, d])
	featureVariadicLogicalASTs

	// Enable error generation when a presence test or optional field selection is
	// performed on a primitive type.
	featureEnableErrorOnBadPresenceTest

	// Enable escape syntax for field identifiers (`).
	featureIdentEscapeSyntax
)

var featureIDsToNames = map[int]string{
	featureEnableMacroCallTracking:     "cel.feature.macro_call_tracking",
	featureCrossTypeNumericComparisons: "cel.feature.cross_type_numeric_comparisons",
	featureIdentEscapeSyntax:           "cel.feature.backtick_escape_syntax",
}

func featureNameByID(id int) (string, bool) {
	name, found := featureIDsToNames[id]
	return name, found
}

func featureIDByName(name string) (int, bool) {
	for id, n := range featureIDsToNames {
		if n == name {
			return id, true
		}
	}
	return 0, false
}

// EnvOption is a functional interface for configuring the environment.
type EnvOption func(e *Env) (*Env, error)

// ClearMacros options clears all parser macros.
//
// Clearing macros will ensure CEL expressions can only contain linear evaluation paths, as
// comprehensions such as `all` and `exists` are enabled only via macros.
func ClearMacros() EnvOption {
	return func(e *Env) (*Env, error) {
		e.macros = NoMacros
		return e, nil
	}
}

// CustomTypeAdapter swaps the default types.Adapter implementation with a custom one.
//
// Note: This option must be specified before the Types and TypeDescs options when used together.
func CustomTypeAdapter(adapter types.Adapter) EnvOption {
	return func(e *Env) (*Env, error) {
		e.adapter = adapter
		return e, nil
	}
}

// CustomTypeProvider replaces the types.Provider implementation with a custom one.
//
// The `provider` variable type may either be types.Provider or ref.TypeProvider (deprecated)
//
// Note: This option must be specified before the Types and TypeDescs options when used together.
func CustomTypeProvider(provider any) EnvOption {
	return func(e *Env) (*Env, error) {
		var err error
		e.provider, err = maybeInteropProvider(provider)
		return e, err
	}
}

// Declarations option extends the declaration set configured in the environment.
//
// Note: Declarations will by default be appended to the pre-existing declaration set configured
// for the environment. The NewEnv call builds on top of the standard CEL declarations. For a
// purely custom set of declarations use NewCustomEnv.
//
// Deprecated: use FunctionDecls and VariableDecls or FromConfig instead.
func Declarations(decls ...*exprpb.Decl) EnvOption {
	declOpts := []EnvOption{}
	var err error
	var opt EnvOption
	// Convert the declarations to `EnvOption` values ahead of time.
	// Surface any errors in conversion when the options are applied.
	for _, d := range decls {
		opt, err = ExprDeclToDeclaration(d)
		if err != nil {
			break
		}
		declOpts = append(declOpts, opt)
	}
	return func(e *Env) (*Env, error) {
		if err != nil {
			return nil, err
		}
		for _, o := range declOpts {
			e, err = o(e)
			if err != nil {
				return nil, err
			}
		}
		return e, nil
	}
}

// EagerlyValidateDeclarations ensures that any collisions between configured declarations are caught
// at the time of the `NewEnv` call.
//
// Eagerly validating declarations is also useful for bootstrapping a base `cel.Env` value.
// Calls to base `Env.Extend()` will be significantly faster when declarations are eagerly validated
// as declarations will be collision-checked at most once and only incrementally by way of `Extend`
//
// Disabled by default as not all environments are used for type-checking.
func EagerlyValidateDeclarations(enabled bool) EnvOption {
	return features(featureEagerlyValidateDeclarations, enabled)
}

// HomogeneousAggregateLiterals disables mixed type list and map literal values.
//
// Note, it is still possible to have heterogeneous aggregates when provided as variables to the
// expression, as well as via conversion of well-known dynamic types, or with unchecked
// expressions.
func HomogeneousAggregateLiterals() EnvOption {
	return ASTValidators(ValidateHomogeneousAggregateLiterals())
}

// variadicLogicalOperatorASTs flatten like-operator chained logical expressions into a single
// variadic call with N-terms. This behavior is useful when serializing to a protocol buffer as
// it will reduce the number of recursive calls needed to deserialize the AST later.
//
// For example, given the following expression the call graph will be rendered accordingly:
//
//	expression: a && b && c && (d || e)
//	ast: call(_&&_, [a, b, c, call(_||_, [d, e])])
func variadicLogicalOperatorASTs() EnvOption {
	return features(featureVariadicLogicalASTs, true)
}

// Macros option extends the macro set configured in the environment.
//
// Note: This option must be specified after ClearMacros if used together.
func Macros(macros ...Macro) EnvOption {
	return func(e *Env) (*Env, error) {
		e.macros = append(e.macros, macros...)
		return e, nil
	}
}

// Container sets the container for resolving variable names. Defaults to an empty container.
//
// If all references within an expression are relative to a protocol buffer package, then
// specifying a container of `google.type` would make it possible to write expressions such as
// `Expr{expression: 'a < b'}` instead of having to write `google.type.Expr{...}`.
func Container(name string) EnvOption {
	return func(e *Env) (*Env, error) {
		cont, err := e.Container.Extend(containers.Name(name))
		if err != nil {
			return nil, err
		}
		e.Container = cont
		return e, nil
	}
}

// Abbrevs configures a set of simple names as abbreviations for fully-qualified names.
//
// An abbreviation (abbrev for short) is a simple name that expands to a fully-qualified name.
// Abbreviations can be useful when working with variables, functions, and especially types from
// multiple namespaces:
//
//	// CEL object construction
//	qual.pkg.version.ObjTypeName{
//	   field: alt.container.ver.FieldTypeName{value: ...}
//	}
//
// Only one the qualified names above may be used as the CEL container, so at least one of these
// references must be a long qualified name within an otherwise short CEL program. Using the
// following abbreviations, the program becomes much simpler:
//
//	// CEL Go option
//	Abbrevs("qual.pkg.version.ObjTypeName", "alt.container.ver.FieldTypeName")
//	// Simplified Object construction
//	ObjTypeName{field: FieldTypeName{value: ...}}
//
// There are a few rules for the qualified names and the simple abbreviations generated from them:
// - Qualified names must be dot-delimited, e.g. `package.subpkg.name`.
// - The last element in the qualified name is the abbreviation.
// - Abbreviations must not collide with each other.
// - The abbreviation must not collide with unqualified names in use.
//
// Abbreviations are distinct from container-based references in the following important ways:
// - Abbreviations must expand to a fully-qualified name.
// - Expanded abbreviations do not participate in namespace resolution.
// - Abbreviation expansion is done instead of the container search for a matching identifier.
// - Containers follow C++ namespace resolution rules with searches from the most qualified name
//
//	to the least qualified name.
//
// - Container references within the CEL program may be relative, and are resolved to fully
//
//	qualified names at either type-check time or program plan time, whichever comes first.
//
// If there is ever a case where an identifier could be in both the container and as an
// abbreviation, the abbreviation wins as this will ensure that the meaning of a program is
// preserved between compilations even as the container evolves.
func Abbrevs(qualifiedNames ...string) EnvOption {
	return func(e *Env) (*Env, error) {
		cont, err := e.Container.Extend(containers.Abbrevs(qualifiedNames...))
		if err != nil {
			return nil, err
		}
		e.Container = cont
		return e, nil
	}
}

// customTypeRegistry is an internal-only interface containing the minimum methods required to support
// custom types. It is a subset of methods from ref.TypeRegistry.
type customTypeRegistry interface {
	RegisterDescriptor(protoreflect.FileDescriptor) error
	RegisterType(...ref.Type) error
}

// Types adds one or more type declarations to the environment, allowing for construction of
// type-literals whose definitions are included in the common expression built-in set.
//
// The input types may either be instances of `proto.Message` or `ref.Type`. Any other type
// provided to this option will result in an error.
//
// Well-known protobuf types within the `google.protobuf.*` package are included in the standard
// environment by default.
//
// Note: This option must be specified after the CustomTypeProvider option when used together.
func Types(addTypes ...any) EnvOption {
	return func(e *Env) (*Env, error) {
		reg, isReg := e.provider.(customTypeRegistry)
		if !isReg {
			return nil, fmt.Errorf("custom types not supported by provider: %T", e.provider)
		}
		for _, t := range addTypes {
			switch v := t.(type) {
			case proto.Message:
				fdMap := pb.CollectFileDescriptorSet(v)
				for _, fd := range fdMap {
					err := reg.RegisterDescriptor(fd)
					if err != nil {
						return nil, err
					}
				}
			case ref.Type:
				err := reg.RegisterType(v)
				if err != nil {
					return nil, err
				}
			default:
				return nil, fmt.Errorf("unsupported type: %T", t)
			}
		}
		return e, nil
	}
}

// TypeDescs adds type declarations from any protoreflect.FileDescriptor, protoregistry.Files,
// google.protobuf.FileDescriptorProto or google.protobuf.FileDescriptorSet provided.
//
// Note that messages instantiated from these descriptors will be *dynamicpb.Message values
// rather than the concrete message type.
//
// TypeDescs are hermetic to a single Env object, but may be copied to other Env values via
// extension or by re-using the same EnvOption with another NewEnv() call.
func TypeDescs(descs ...any) EnvOption {
	return func(e *Env) (*Env, error) {
		reg, isReg := e.provider.(customTypeRegistry)
		if !isReg {
			return nil, fmt.Errorf("custom types not supported by provider: %T", e.provider)
		}
		// Scan the input descriptors for FileDescriptorProto messages and accumulate them into a
		// synthetic FileDescriptorSet as the FileDescriptorProto messages may refer to each other
		// and will not resolve properly unless they are part of the same set.
		var fds *descpb.FileDescriptorSet
		for _, d := range descs {
			switch f := d.(type) {
			case *descpb.FileDescriptorProto:
				if fds == nil {
					fds = &descpb.FileDescriptorSet{
						File: []*descpb.FileDescriptorProto{},
					}
				}
				fds.File = append(fds.File, f)
			}
		}
		if fds != nil {
			if err := registerFileSet(reg, fds); err != nil {
				return nil, err
			}
		}
		for _, d := range descs {
			switch f := d.(type) {
			case *protoregistry.Files:
				if err := registerFiles(reg, f); err != nil {
					return nil, err
				}
			case protoreflect.FileDescriptor:
				if err := reg.RegisterDescriptor(f); err != nil {
					return nil, err
				}
			case *descpb.FileDescriptorSet:
				if err := registerFileSet(reg, f); err != nil {
					return nil, err
				}
			case *descpb.FileDescriptorProto:
				// skip, handled as a synthetic file descriptor set.
			default:
				return nil, fmt.Errorf("unsupported type descriptor: %T", d)
			}
		}
		return e, nil
	}
}

func registerFileSet(reg customTypeRegistry, fileSet *descpb.FileDescriptorSet) error {
	files, err := protodesc.NewFiles(fileSet)
	if err != nil {
		return fmt.Errorf("protodesc.NewFiles(%v) failed: %v", fileSet, err)
	}
	return registerFiles(reg, files)
}

func registerFiles(reg customTypeRegistry, files *protoregistry.Files) error {
	var err error
	files.RangeFiles(func(fd protoreflect.FileDescriptor) bool {
		err = reg.RegisterDescriptor(fd)
		return err == nil
	})
	return err
}

// ProgramOption is a functional interface for configuring evaluation bindings and behaviors.
type ProgramOption func(p *prog) (*prog, error)

// CustomDecorator appends an InterpreterDecorator to the program.
//
// InterpretableDecorators can be used to inspect, alter, or replace the Program plan.
func CustomDecorator(dec interpreter.InterpretableDecorator) ProgramOption {
	return func(p *prog) (*prog, error) {
		p.plannerOptions = append(p.plannerOptions, interpreter.CustomDecorator(dec))
		return p, nil
	}
}

// Functions adds function overloads that extend or override the set of CEL built-ins.
//
// Deprecated: use Function() instead to declare the function, its overload signatures,
// and the overload implementations.
func Functions(funcs ...*functions.Overload) ProgramOption {
	return func(p *prog) (*prog, error) {
		if err := p.dispatcher.Add(funcs...); err != nil {
			return nil, err
		}
		return p, nil
	}
}

// Globals sets the global variable values for a given program. These values may be shadowed by
// variables with the same name provided to the Eval() call. If Globals is used in a Library with
// a Lib EnvOption, vars may shadow variables provided by previously added libraries.
//
// The vars value may either be an `cel.Activation` instance or a `map[string]any`.
func Globals(vars any) ProgramOption {
	return func(p *prog) (*prog, error) {
		defaultVars, err := NewActivation(vars)
		if err != nil {
			return nil, err
		}
		if p.defaultVars != nil {
			defaultVars = interpreter.NewHierarchicalActivation(p.defaultVars, defaultVars)
		}
		p.defaultVars = defaultVars
		return p, nil
	}
}

// OptimizeRegex provides a way to replace the InterpretableCall for regex functions. This can be used
// to compile regex string constants at program creation time and report any errors and then use the
// compiled regex for all regex function invocations.
func OptimizeRegex(regexOptimizations ...*interpreter.RegexOptimization) ProgramOption {
	return func(p *prog) (*prog, error) {
		p.regexOptimizations = append(p.regexOptimizations, regexOptimizations...)
		return p, nil
	}
}

// ConfigOptionFactory declares a signature which accepts a configuration element, e.g. env.Extension
// and optionally produces an EnvOption in response.
//
// If there are multiple ConfigOptionFactory values which could apply to the same configuration node
// the first one that returns an EnvOption and a `true` response will be used, and the config node
// will not be passed along to any other option factory.
//
// Only the *env.Extension type is provided at this time, but validators, optimizers, and other tuning
// parameters may be supported in the future.
type ConfigOptionFactory func(any) (EnvOption, bool)

// FromConfig produces and applies a set of EnvOption values derived from an env.Config object.
//
// For configuration elements which refer to features outside of the `cel` package, an optional set of
// ConfigOptionFactory values may be passed in to support the conversion from static configuration to
// configured cel.Env value.
//
// Note: disabling the standard library will clear the EnvOptions values previously set for the
// environment with the exception of propagating types and adapters over to the new environment.
//
// Note: to support custom types referenced in the configuration file, you must ensure that one of
// the following options appears before the FromConfig option: Types, TypeDescs, or CustomTypeProvider
// as the type provider configured at the time when the config is processed is the one used to derive
// type references from the configuration.
func FromConfig(config *env.Config, optFactories ...ConfigOptionFactory) EnvOption {
	return func(e *Env) (*Env, error) {
		if err := config.Validate(); err != nil {
			return nil, err
		}
		opts, err := configToEnvOptions(config, e.CELTypeProvider(), optFactories)
		if err != nil {
			return nil, err
		}
		for _, o := range opts {
			e, err = o(e)
			if err != nil {
				return nil, err
			}
		}
		return e, nil
	}
}

// configToEnvOptions generates a set of EnvOption values (or error) based on a config, a type provider,
// and an optional set of environment options.
func configToEnvOptions(config *env.Config, provider types.Provider, optFactories []ConfigOptionFactory) ([]EnvOption, error) {
	envOpts := []EnvOption{}
	// Configure the standard lib subset.
	if config.StdLib != nil {
		envOpts = append(envOpts, func(e *Env) (*Env, error) {
			if e.HasLibrary("cel.lib.std") {
				return nil, errors.New("invalid subset of stdlib: create a custom env")
			}
			return e, nil
		})
		if !config.StdLib.Disabled {
			envOpts = append(envOpts, StdLib(StdLibSubset(config.StdLib)))
		}
	} else {
		envOpts = append(envOpts, StdLib())
	}

	// Configure the container
	if config.Container != "" {
		envOpts = append(envOpts, Container(config.Container))
	}

	// Configure abbreviations
	for _, imp := range config.Imports {
		envOpts = append(envOpts, Abbrevs(imp.Name))
	}

	// Configure the context variable declaration
	if config.ContextVariable != nil {
		typeName := config.ContextVariable.TypeName
		if _, found := provider.FindStructType(typeName); !found {
			return nil, fmt.Errorf("invalid context proto type: %q", typeName)
		}
		// Attempt to instantiate the proto in order to reflect to its descriptor
		msg := provider.NewValue(typeName, map[string]ref.Val{})
		pbMsg, ok := msg.Value().(proto.Message)
		if !ok {
			return nil, fmt.Errorf("unsupported context type: %T", msg.Value())
		}
		envOpts = append(envOpts, DeclareContextProto(pbMsg.ProtoReflect().Descriptor()))
	}

	// Configure variables
	if len(config.Variables) != 0 {
		vars := make([]*decls.VariableDecl, 0, len(config.Variables))
		for _, v := range config.Variables {
			vDef, err := v.AsCELVariable(provider)
			if err != nil {
				return nil, err
			}
			vars = append(vars, vDef)
		}
		envOpts = append(envOpts, VariableDecls(vars...))
	}

	// Configure functions
	if len(config.Functions) != 0 {
		funcs := make([]*decls.FunctionDecl, 0, len(config.Functions))
		for _, f := range config.Functions {
			fnDef, err := f.AsCELFunction(provider)
			if err != nil {
				return nil, err
			}
			funcs = append(funcs, fnDef)
		}
		envOpts = append(envOpts, FunctionDecls(funcs...))
	}

	// Configure features
	for _, feat := range config.Features {
		// Note, if a feature is not found, it is skipped as it is possible the feature
		// is not intended to be supported publicly. In the future, a refinement of
		// to this strategy to report unrecognized features and validators should probably
		// be covered as a standard ConfigOptionFactory
		if id, found := featureIDByName(feat.Name); found {
			envOpts = append(envOpts, features(id, feat.Enabled))
		}
	}

	// Configure validators
	for _, val := range config.Validators {
		if fac, found := astValidatorFactories[val.Name]; found {
			envOpts = append(envOpts, func(e *Env) (*Env, error) {
				validator, err := fac(val)
				if err != nil {
					return nil, fmt.Errorf("%w", err)
				}
				return ASTValidators(validator)(e)
			})
		} else if opt, handled := handleExtendedConfigOption(val, optFactories); handled {
			envOpts = append(envOpts, opt)
		}
		// we don't error when the validator isn't found as it may be part
		// of an extension library and enabled implicitly.
	}

	// Configure extensions
	for _, ext := range config.Extensions {
		// version number has been validated by the call to `Validate`
		ver, _ := ext.VersionNumber()
		if ext.Name == "optional" {
			envOpts = append(envOpts, OptionalTypes(OptionalTypesVersion(ver)))
		} else {
			opt, handled := handleExtendedConfigOption(ext, optFactories)
			if !handled {
				return nil, fmt.Errorf("unrecognized extension: %s", ext.Name)
			}
			envOpts = append(envOpts, opt)
		}
	}

	return envOpts, nil
}

func handleExtendedConfigOption(conf any, optFactories []ConfigOptionFactory) (EnvOption, bool) {
	for _, optFac := range optFactories {
		if opt, useOption := optFac(conf); useOption {
			return opt, true
		}
	}
	return nil, false
}

// EvalOption indicates an evaluation option that may affect the evaluation behavior or information
// in the output result.
type EvalOption int

const (
	// OptTrackState will cause the runtime to return an immutable EvalState value in the Result.
	OptTrackState EvalOption = 1 << iota

	// OptExhaustiveEval causes the runtime to disable short-circuits and track state.
	OptExhaustiveEval EvalOption = 1<<iota | OptTrackState

	// OptOptimize precomputes functions and operators with constants as arguments at program
	// creation time. It also pre-compiles regex pattern constants passed to 'matches', reports any compilation errors
	// at program creation and uses the compiled regex pattern for all 'matches' function invocations.
	// This flag is useful when the expression will be evaluated repeatedly against
	// a series of different inputs.
	OptOptimize EvalOption = 1 << iota

	// OptPartialEval enables the evaluation of a partial state where the input data that may be
	// known to be missing, either as top-level variables, or somewhere within a variable's object
	// member graph.
	//
	// By itself, OptPartialEval does not change evaluation behavior unless the input to the
	// Program Eval() call is created via PartialVars().
	OptPartialEval EvalOption = 1 << iota

	// OptTrackCost enables the runtime cost calculation while validation and return cost within evalDetails
	// cost calculation is available via func ActualCost()
	OptTrackCost EvalOption = 1 << iota

	// OptCheckStringFormat enables compile-time checking of string.format calls for syntax/cardinality.
	//
	// Deprecated: use ext.StringsValidateFormatCalls() as this option is now a no-op.
	OptCheckStringFormat EvalOption = 1 << iota
)

// EvalOptions sets one or more evaluation options which may affect the evaluation or Result.
func EvalOptions(opts ...EvalOption) ProgramOption {
	return func(p *prog) (*prog, error) {
		for _, opt := range opts {
			p.evalOpts |= opt
		}
		return p, nil
	}
}

// InterruptCheckFrequency configures the number of iterations within a comprehension to evaluate
// before checking whether the function evaluation has been interrupted.
func InterruptCheckFrequency(checkFrequency uint) ProgramOption {
	return func(p *prog) (*prog, error) {
		p.interruptCheckFrequency = checkFrequency
		return p, nil
	}
}

// CostEstimatorOptions configure type-check time options for estimating expression cost.
func CostEstimatorOptions(costOpts ...checker.CostOption) EnvOption {
	return func(e *Env) (*Env, error) {
		e.costOptions = append(e.costOptions, costOpts...)
		return e, nil
	}
}

// CostTrackerOptions configures a set of options for cost-tracking.
//
// Note, CostTrackerOptions is a no-op unless CostTracking is also enabled.
func CostTrackerOptions(costOpts ...interpreter.CostTrackerOption) ProgramOption {
	return func(p *prog) (*prog, error) {
		p.costOptions = append(p.costOptions, costOpts...)
		return p, nil
	}
}

// CostTracking enables cost tracking and registers a ActualCostEstimator that can optionally provide a runtime cost estimate for any function calls.
func CostTracking(costEstimator interpreter.ActualCostEstimator) ProgramOption {
	return func(p *prog) (*prog, error) {
		p.callCostEstimator = costEstimator
		p.evalOpts |= OptTrackCost
		return p, nil
	}
}

// CostLimit enables cost tracking and sets configures program evaluation to exit early with a
// "runtime cost limit exceeded" error if the runtime cost exceeds the costLimit.
// The CostLimit is a metric that corresponds to the number and estimated expense of operations
// performed while evaluating an expression. It is indicative of CPU usage, not memory usage.
func CostLimit(costLimit uint64) ProgramOption {
	return func(p *prog) (*prog, error) {
		p.costLimit = &costLimit
		p.evalOpts |= OptTrackCost
		return p, nil
	}
}

func fieldToCELType(field protoreflect.FieldDescriptor) (*Type, error) {
	if field.Kind() == protoreflect.MessageKind || field.Kind() == protoreflect.GroupKind {
		msgName := (string)(field.Message().FullName())
		return ObjectType(msgName), nil
	}
	if primitiveType, found := types.ProtoCELPrimitives[field.Kind()]; found {
		return primitiveType, nil
	}
	if field.Kind() == protoreflect.EnumKind {
		return IntType, nil
	}
	return nil, fmt.Errorf("field %s type %s not implemented", field.FullName(), field.Kind().String())
}

func fieldToVariable(field protoreflect.FieldDescriptor) (*decls.VariableDecl, error) {
	name := string(field.Name())
	if field.IsMap() {
		mapKey := field.MapKey()
		mapValue := field.MapValue()
		keyType, err := fieldToCELType(mapKey)
		if err != nil {
			return nil, err
		}
		valueType, err := fieldToCELType(mapValue)
		if err != nil {
			return nil, err
		}
		return decls.NewVariable(name, MapType(keyType, valueType)), nil
	}
	if field.IsList() {
		elemType, err := fieldToCELType(field)
		if err != nil {
			return nil, err
		}
		return decls.NewVariable(name, ListType(elemType)), nil
	}
	celType, err := fieldToCELType(field)
	if err != nil {
		return nil, err
	}
	return decls.NewVariable(name, celType), nil
}

// DeclareContextProto returns an option to extend CEL environment with declarations from the given context proto.
// Each field of the proto defines a variable of the same name in the environment.
// https://github.com/google/cel-spec/blob/master/doc/langdef.md#evaluation-environment
func DeclareContextProto(descriptor protoreflect.MessageDescriptor) EnvOption {
	return func(e *Env) (*Env, error) {
		if e.contextProto != nil {
			return nil, fmt.Errorf("context proto already declared as %q, got %q",
				e.contextProto.FullName(), descriptor.FullName())
		}
		e.contextProto = descriptor
		fields := descriptor.Fields()
		vars := make([]*decls.VariableDecl, 0, fields.Len())
		for i := 0; i < fields.Len(); i++ {
			field := fields.Get(i)
			variable, err := fieldToVariable(field)
			if err != nil {
				return nil, err
			}
			vars = append(vars, variable)
		}
		var err error
		e, err = VariableDecls(vars...)(e)
		if err != nil {
			return nil, err
		}
		return Types(dynamicpb.NewMessage(descriptor))(e)
	}
}

// ContextProtoVars uses the fields of the input proto.Messages as top-level variables within an Activation.
//
// Consider using with `DeclareContextProto` to simplify variable type declarations and publishing when using
// protocol buffers.
func ContextProtoVars(ctx proto.Message) (Activation, error) {
	if ctx == nil || !ctx.ProtoReflect().IsValid() {
		return interpreter.EmptyActivation(), nil
	}
	reg, err := types.NewRegistry(ctx)
	if err != nil {
		return nil, err
	}
	pbRef := ctx.ProtoReflect()
	typeName := string(pbRef.Descriptor().FullName())
	fields := pbRef.Descriptor().Fields()
	vars := make(map[string]any, fields.Len())
	for i := 0; i < fields.Len(); i++ {
		field := fields.Get(i)
		sft, found := reg.FindStructFieldType(typeName, field.TextName())
		if !found {
			return nil, fmt.Errorf("no such field: %s", field.TextName())
		}
		fieldVal, err := sft.GetFrom(ctx)
		if err != nil {
			return nil, err
		}
		vars[field.TextName()] = fieldVal
	}
	return NewActivation(vars)
}

// EnableMacroCallTracking ensures that call expressions which are replaced by macros
// are tracked in the `SourceInfo` of parsed and checked expressions.
func EnableMacroCallTracking() EnvOption {
	return features(featureEnableMacroCallTracking, true)
}

// EnableIdentifierEscapeSyntax enables identifier escaping (`) syntax for
// fields.
func EnableIdentifierEscapeSyntax() EnvOption {
	return features(featureIdentEscapeSyntax, true)
}

// CrossTypeNumericComparisons makes it possible to compare across numeric types, e.g. double < int
func CrossTypeNumericComparisons(enabled bool) EnvOption {
	return features(featureCrossTypeNumericComparisons, enabled)
}

// DefaultUTCTimeZone ensures that time-based operations use the UTC timezone rather than the
// input time's local timezone.
func DefaultUTCTimeZone(enabled bool) EnvOption {
	return features(featureDefaultUTCTimeZone, enabled)
}

// features sets the given feature flags.  See list of Feature constants above.
func features(flag int, enabled bool) EnvOption {
	return func(e *Env) (*Env, error) {
		e.features[flag] = enabled
		return e, nil
	}
}

// ParserRecursionLimit adjusts the AST depth the parser will tolerate.
// Defaults defined in the parser package.
func ParserRecursionLimit(limit int) EnvOption {
	return func(e *Env) (*Env, error) {
		e.prsrOpts = append(e.prsrOpts, parser.MaxRecursionDepth(limit))
		return e, nil
	}
}

// ParserExpressionSizeLimit adjusts the number of code points the expression parser is allowed to parse.
// Defaults defined in the parser package.
func ParserExpressionSizeLimit(limit int) EnvOption {
	return func(e *Env) (*Env, error) {
		e.prsrOpts = append(e.prsrOpts, parser.ExpressionSizeCodePointLimit(limit))
		return e, nil
	}
}

// EnableHiddenAccumulatorName sets the parser to use the identifier '@result' for accumulators
// which is not normally accessible from CEL source.
func EnableHiddenAccumulatorName(enabled bool) EnvOption {
	return func(e *Env) (*Env, error) {
		e.prsrOpts = append(e.prsrOpts, parser.EnableHiddenAccumulatorName(enabled))
		return e, nil
	}
}

func maybeInteropProvider(provider any) (types.Provider, error) {
	switch p := provider.(type) {
	case types.Provider:
		return p, nil
	case ref.TypeProvider:
		return &interopCELTypeProvider{TypeProvider: p}, nil
	default:
		return nil, fmt.Errorf("unsupported type provider: %T", provider)
	}
}
