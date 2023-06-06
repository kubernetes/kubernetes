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
	"fmt"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/dynamicpb"

	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/types/pb"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
	"github.com/google/cel-go/interpreter/functions"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	descpb "google.golang.org/protobuf/types/descriptorpb"
)

// These constants beginning with "Feature" enable optional behavior in
// the library.  See the documentation for each constant to see its
// effects, compatibility restrictions, and standard conformance.
const (
	_ = iota

	// Disallow heterogeneous aggregate (list, map) literals.
	// Note, it is still possible to have heterogeneous aggregates when
	// provided as variables to the expression, as well as via conversion
	// of well-known dynamic types, or with unchecked expressions.
	// Affects checking.  Provides a subset of standard behavior.
	featureDisableDynamicAggregateLiterals

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

	// Enable the use of optional types in the syntax, type-system, type-checking,
	// and runtime.
	featureOptionalTypes
)

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

// CustomTypeAdapter swaps the default ref.TypeAdapter implementation with a custom one.
//
// Note: This option must be specified before the Types and TypeDescs options when used together.
func CustomTypeAdapter(adapter ref.TypeAdapter) EnvOption {
	return func(e *Env) (*Env, error) {
		e.adapter = adapter
		return e, nil
	}
}

// CustomTypeProvider swaps the default ref.TypeProvider implementation with a custom one.
//
// Note: This option must be specified before the Types and TypeDescs options when used together.
func CustomTypeProvider(provider ref.TypeProvider) EnvOption {
	return func(e *Env) (*Env, error) {
		e.provider = provider
		return e, nil
	}
}

// Declarations option extends the declaration set configured in the environment.
//
// Note: Declarations will by default be appended to the pre-existing declaration set configured
// for the environment. The NewEnv call builds on top of the standard CEL declarations. For a
// purely custom set of declarations use NewCustomEnv.
func Declarations(decls ...*exprpb.Decl) EnvOption {
	return func(e *Env) (*Env, error) {
		e.declarations = append(e.declarations, decls...)
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

// HomogeneousAggregateLiterals option ensures that list and map literal entry types must agree
// during type-checking.
//
// Note, it is still possible to have heterogeneous aggregates when provided as variables to the
// expression, as well as via conversion of well-known dynamic types, or with unchecked
// expressions.
func HomogeneousAggregateLiterals() EnvOption {
	return features(featureDisableDynamicAggregateLiterals, true)
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
		reg, isReg := e.provider.(ref.TypeRegistry)
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
		reg, isReg := e.provider.(ref.TypeRegistry)
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

func registerFileSet(reg ref.TypeRegistry, fileSet *descpb.FileDescriptorSet) error {
	files, err := protodesc.NewFiles(fileSet)
	if err != nil {
		return fmt.Errorf("protodesc.NewFiles(%v) failed: %v", fileSet, err)
	}
	return registerFiles(reg, files)
}

func registerFiles(reg ref.TypeRegistry, files *protoregistry.Files) error {
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
		p.decorators = append(p.decorators, dec)
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
// The vars value may either be an `interpreter.Activation` instance or a `map[string]any`.
func Globals(vars any) ProgramOption {
	return func(p *prog) (*prog, error) {
		defaultVars, err := interpreter.NewActivation(vars)
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

func fieldToCELType(field protoreflect.FieldDescriptor) (*exprpb.Type, error) {
	if field.Kind() == protoreflect.MessageKind || field.Kind() == protoreflect.GroupKind {
		msgName := (string)(field.Message().FullName())
		wellKnownType, found := pb.CheckedWellKnowns[msgName]
		if found {
			return wellKnownType, nil
		}
		return decls.NewObjectType(msgName), nil
	}
	if primitiveType, found := pb.CheckedPrimitives[field.Kind()]; found {
		return primitiveType, nil
	}
	if field.Kind() == protoreflect.EnumKind {
		return decls.Int, nil
	}
	return nil, fmt.Errorf("field %s type %s not implemented", field.FullName(), field.Kind().String())
}

func fieldToDecl(field protoreflect.FieldDescriptor) (*exprpb.Decl, error) {
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
		return decls.NewVar(name, decls.NewMapType(keyType, valueType)), nil
	}
	if field.IsList() {
		elemType, err := fieldToCELType(field)
		if err != nil {
			return nil, err
		}
		return decls.NewVar(name, decls.NewListType(elemType)), nil
	}
	celType, err := fieldToCELType(field)
	if err != nil {
		return nil, err
	}
	return decls.NewVar(name, celType), nil
}

// DeclareContextProto returns an option to extend CEL environment with declarations from the given context proto.
// Each field of the proto defines a variable of the same name in the environment.
// https://github.com/google/cel-spec/blob/master/doc/langdef.md#evaluation-environment
func DeclareContextProto(descriptor protoreflect.MessageDescriptor) EnvOption {
	return func(e *Env) (*Env, error) {
		var decls []*exprpb.Decl
		fields := descriptor.Fields()
		for i := 0; i < fields.Len(); i++ {
			field := fields.Get(i)
			decl, err := fieldToDecl(field)
			if err != nil {
				return nil, err
			}
			decls = append(decls, decl)
		}
		var err error
		e, err = Declarations(decls...)(e)
		if err != nil {
			return nil, err
		}
		return Types(dynamicpb.NewMessage(descriptor))(e)
	}
}

// EnableMacroCallTracking ensures that call expressions which are replaced by macros
// are tracked in the `SourceInfo` of parsed and checked expressions.
func EnableMacroCallTracking() EnvOption {
	return features(featureEnableMacroCallTracking, true)
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

// OptionalTypes enable support for optional syntax and types in CEL. The optional value type makes
// it possible to express whether variables have been provided, whether a result has been computed,
// and in the future whether an object field path, map key value, or list index has a value.
func OptionalTypes() EnvOption {
	return Lib(optionalLibrary{})
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
