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
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

// TagValidator describes a single validation tag and how to use it. To be
// findable by validation-gen, a TagValidator must be registered - see
// RegisterTagValidator.
//
// TagValidators are always evaluated before TypeValidators and
// FieldValidators. In general, TagValidators should not depend on other
// TagValidators having been run already because users might specify tags in
// the any order. The one exception to this rule is that some TagValidators may
// be designated as "late" validators (see LateTagValidator), which means they
// will be run after all non-late TagValidators.
//
// No other guarantees are made about the order of execution of TagValidators
// or LateTagValidators. Instead of relying on tag ordering, TagValidators can
// accumulate information internally and use a TypeValidator and/or
// FieldValidator to finish the work.
type TagValidator interface {
	// Init initializes the implementation.  This will be called exactly once.
	Init(cfg Config)

	// TagName returns the full tag name (without the "marker" prefix) for this
	// tag.
	TagName() string

	// ValidScopes returns the set of scopes where this tag may be used.
	ValidScopes() sets.Set[Scope]

	// GetValidations returns any validations described by this tag.
	GetValidations(context Context, tag codetags.Tag) (Validations, error)

	// Docs returns user-facing documentation for this tag.
	Docs() TagDoc
}

// LateTagValidator is an optional extension to TagValidator. Any TagValidator
// which implements this interface will be evaluated after all TagValidators
// which do not.
type LateTagValidator interface {
	LateTagValidator()
}

// TypeValidator describes a validator which runs on every type definition.
// To be findable by validation-gen, a TypeValidator must be registered - see
// RegisterTypeValidator.
//
// TypeValidators are always processed after TagValidators, and after the type
// has been fully processed (including all child fields and their types). This
// means that they can "finish" work with data that was collected by
// TagValidators.
//
// TypeValidators MUST NOT depend on other TypeValidators having been run
// already.
type TypeValidator interface {
	// Init initializes the implementation.  This will be called exactly once.
	Init(cfg Config)

	// Name returns a unique name for this validator.  This is used for sorting
	// and logging.
	Name() string

	// GetValidations returns any validations imposed by this validator for the
	// given context.
	//
	// The way gengo handles type definitions varies between structs and other
	// types.  For struct definitions (e.g. `type Foo struct {}`), the realType
	// is the struct itself (the Kind field will be `types.Struct`) and the
	// parentType will be nil.  For other types (e.g. `type Bar string`), the
	// realType will be the underlying type and the parentType will be the
	// newly defined type (the Kind field will be `types.Alias`).
	GetValidations(context Context) (Validations, error)
}

// FieldValidator describes a validator which runs on every field definition.
// To be findable by validation-gen, a FieldValidator must be registered - see
// RegisterFieldValidator.
//
// FieldValidators are always processed after TagValidators and TypeValidators,
// and after the field has been fully processed (including all child fields).
// This means that they can "finish" work with data that was collected by
// TagValidators.
//
// FieldValidators MUST NOT depend on other FieldValidators having been run
// already.
type FieldValidator interface {
	// Init initializes the implementation.  This will be called exactly once.
	Init(cfg Config)

	// Name returns a unique name for this validator.  This is used for sorting
	// and logging.
	Name() string

	// GetValidations returns any validations imposed by this validator for the
	// given context.
	//
	// The way gengo handles type definitions varies between structs and other
	// types.  For struct definitions (e.g. `type Foo struct {}`), the realType
	// is the struct itself (the Kind field will be `types.Struct`) and the
	// parentType will be nil.  For other types (e.g. `type Bar string`), the
	// realType will be the underlying type and the parentType will be the
	// newly defined type (the Kind field will be `types.Alias`).
	GetValidations(context Context) (Validations, error)
}

// Config carries optional configuration information for use by validators.
type Config struct {
	// GengoContext provides gengo's generator Context.  This allows validators
	// to look up all sorts of other information.
	GengoContext *generator.Context

	// Validator provides a way to compose validations.
	//
	// For example, it is possible to define a validation such as
	// "+myValidator=+format=IP" by using the registry to extract the
	// validation for the embedded "+format=IP" and use those to
	// create the final Validations returned by the "+myValidator" tag.
	//
	// This field MUST NOT be used during init, since other validators may not
	// be initialized yet.
	Validator Validator
}

// Scope describes where a validation (or potential validation) is located.
type Scope string

// Note: All of these values should be strings which can be used in an error
// message such as "may not be used in %s".
const (
	// ScopeType indicates a validation on a type definition, which applies to
	// all instances of that type.
	ScopeType Scope = "type definitions"

	// ScopeField indicates a validation on a particular struct field, which
	// applies only to that field of that struct.
	ScopeField Scope = "struct fields"

	// ScopeListVal indicates a validation which applies to all elements of a
	// list field or type.
	ScopeListVal Scope = "list values"

	// ScopeMapKey indicates a validation which applies to all keys of a map
	// field or type.
	ScopeMapKey Scope = "map keys"

	// ScopeMapVal indicates a validation which applies to all values of a map
	// field or type.
	ScopeMapVal Scope = "map values"

	// ScopeConst indicates a validation which applies to constant values only.
	ScopeConst Scope = "constant values"

	// TODO: It's not clear if we need to distinguish (e.g.) list values of
	// fields from list values of typedefs.  We could make {type,field} be
	// orthogonal to {scalar, list, list-value, map, map-key, map-value} (and
	// maybe even pointers?), but that seems like extra work that is not needed
	// for now.
)

// Context describes where a tag was used, so that the scope can be checked
// and so validators can handle different cases if they need.
type Context struct {
	// Scope is where the validation is being considered.
	Scope Scope

	// Type provides details about the type being validated.  When Scope is
	// ScopeType, this is the newly defined type.  When Scope is ScopeField,
	// this is the field's type (which may be a pointer, an alias, or both).
	// When Scope indicates a list-value, map-key, or map-value, this is the
	// type of that key or value (which, again, may be a pointer, and alias, or
	// both). When Scope is ScopeConst this is the constant's type.
	Type *types.Type

	// Path provides a path to the type or field being validated. This is
	// useful for identifying an exact context, e.g. to track information
	// between related tags. When Scope is ScopeType, this is the Go package
	// path and type name (e.g. "k8s.io/api/core/v1.Pod"). When Scope is
	// ScopeField, this is the field path (e.g. "spec.containers[*].image").
	// When Scope indicates a list-value, map-key, or map-value, this is the
	// type or field path, as described above, with a suffix indicating
	// that it refers to the keys or values. For ScopeConst, this will be nil.
	Path *field.Path

	// Member provides details about a field within a struct when Scope is
	// ScopeField.  For all other values of Scope, this will be nil.
	Member *types.Member

	// ListSelector provides a list of key-value pairs that represent criteria
	// for selecting one or more items from a list.  When Scope is
	// ScopeListVal, this will be non-nil.  An empty selector means that
	// all items in the list should be selected.  For all other values of
	// Scope, this will be nil.
	ListSelector []ListSelectorTerm

	// ParentPath provides a path to the parent type or field of the object
	// being validated, when applicable. enabling unique identification of
	// validation contexts for the same type in different locations.  When
	// Scope is ScopeField, this is the path to the containing struct type or
	// field (depending on where the validation tag was sepcified).  When Scope
	// indicates a list-value, map-key, or map-value, this is the path to the
	// list or map type or field (depending on where the validation tag was
	// specified). When Scope is ScopeType, this is nil.
	ParentPath *field.Path

	// Constants provides access to all constants of the type being
	// validated.  Only set when Scope is ScopeType.
	Constants []*Constant
}

// Constant represents a constant value.
type Constant struct {
	Constant *types.Type
	Tags     []codetags.Tag
}

// ListSelectorTerm represents a field name and value pair.
type ListSelectorTerm struct {
	// Field is the JSON name of the field to match.
	Field string
	// Value is the value to match.  This must be a primitive type which can
	// be used as list-map keys: string, int, or bool.
	Value any
}

// StabilityLevel indicates the stability of a validation tag.
type StabilityLevel string

const (
	// Alpha indicates that a tag's semantics may change in the future.
	Alpha StabilityLevel = "Alpha"
	// Beta indicates that a tag's semantics will remain unchanged for the
	// foreseeable future. This is used for soaking tags before qualifying to stable.
	Beta StabilityLevel = "Beta"
	// Stable indicates that a tag's semantics will remain unchanged for the
	// foreseeable future.
	Stable StabilityLevel = "Stable"
)

// TagDoc describes a comment-tag and its usage.
type TagDoc struct {
	// Tag is the tag name, without the leading '+'.
	Tag string
	// StabilityLevel is the stability level of the tag.
	StabilityLevel StabilityLevel
	// Args lists any arguments this tag might take.
	Args []TagArgDoc `json:",omitempty"`
	// Usage is how the tag is used, including arguments.
	Usage string
	// Description is a short description of this tag's purpose.
	Description string
	// Docs is a human-oriented string explaining this tag.
	Docs string
	// Warning is an optional warning about this tag.
	Warning string `json:",omitempty"`
	// Scopes lists the place or places this tag may be used.
	Scopes []Scope
	// Payloads lists zero or more varieties of value for this tag. If this tag
	// never has a payload, this list should be empty, but if the payload is
	// optional, this list should include an entry for "<none>".
	Payloads []TagPayloadDoc `json:",omitempty"`
	// PayloadsType is the type of the payloads.
	PayloadsType codetags.ValueType `json:",omitempty"`
	// PayloadsRequired is true if a payload is required.
	PayloadsRequired bool `json:",omitempty"`
	// AcceptsUnknownArgs is true if unknown args are accepted
	AcceptsUnknownArgs bool `json:",omitempty"`
}

func (td TagDoc) Arg(name string) (TagArgDoc, bool) {
	for _, arg := range td.Args {
		if arg.Name == name {
			return arg, true
		}
	}
	return TagArgDoc{}, false
}

// TagArgDoc describes an argument for a tag.
//
// For example,
//
//	`+tagName(arg)`
//	`+tagName(name1: arg1, name2: arg2)`
type TagArgDoc struct {
	// Name of this arg. Not provided for positional args.
	Name string
	// Description is a short description of this arg (e.g. `<name>`).
	Description string
	// Type is the type of the arg.
	Type codetags.ArgType
	// Required is true if the argument is required.
	Required bool
	// Default is the effective value if no value is provided.
	Default string
	// Docs is a human-oriented string explaining this arg.
	Docs string
}

// TagPayloadDoc describes a value for a tag (e.g. `+tagName=tagValue`).  Some
// tags support multiple payloads, including <none> (e.g. `+tagName`).
type TagPayloadDoc struct {
	// Description is a short description of this payload (e.g. `<number>`).
	Description string
	// Docs is a human-oriented string explaining this payload.
	Docs string
}

// Validations define the function calls and variables to generate to perform
// validation.
type Validations struct {
	// Functions hold the function calls that should be generated to perform
	// validation.  These functions may not be called in order - they may be
	// sorted based on their flags and other criteria.
	//
	// Each function's signature must be of the form:
	//   func(
	//        // standard arguments
	//        ctx context.Context
	//        op operation.Operation,
	//        fldPath field.Path,
	//        value, oldValue <ValueType>, // always nilable
	//        // additional arguments (optional)
	//        Args[0] <Args[0]Type>,
	//        Args[1] <Args[1]Type>,
	//        ...
	//        Args[N] <Args[N]Type>)
	//
	// The standard arguments are not included in the FunctionGen.Args list.
	Functions []FunctionGen

	// Variables hold any variables which must be generated to perform
	// validation.  Variables are not permitted in every context.
	Variables []VariableGen

	// Comments holds comments to emit (without the leading "//").
	Comments []string

	// OpaqueType indicates that the type being validated is opaque, and that
	// any validations defined on it should not be emitted.
	OpaqueType bool

	// OpaqueKeyType indicates that the key type of a map being validated is
	// opaque, and that any validations defined on it should not be emitted.
	OpaqueKeyType bool

	// OpaqueValType indicates that the key type of a map or slice being
	// validated is opaque, and that any validations defined on it should not
	// be emitted.
	OpaqueValType bool
}

func (v *Validations) Empty() bool {
	return v.Len() == 0
}

func (v *Validations) Len() int {
	return len(v.Functions) + len(v.Variables) + len(v.Comments)
}

func (v *Validations) AddFunction(fn FunctionGen) {
	v.Functions = append(v.Functions, fn)
}

func (v *Validations) AddVariable(vr VariableGen) {
	v.Variables = append(v.Variables, vr)
}

func (v *Validations) AddComment(comment string) {
	v.Comments = append(v.Comments, comment)
}

func (v *Validations) Add(o Validations) {
	v.Functions = append(v.Functions, o.Functions...)
	v.Variables = append(v.Variables, o.Variables...)
	v.Comments = append(v.Comments, o.Comments...)
	v.OpaqueType = v.OpaqueType || o.OpaqueType
	v.OpaqueKeyType = v.OpaqueKeyType || o.OpaqueKeyType
	v.OpaqueValType = v.OpaqueValType || o.OpaqueValType
}

// FunctionFlags define optional properties of a validator.  Most validators
// can just use DefaultFlags.
type FunctionFlags uint32

// IsSet returns true if all of the wanted flags are set.
func (ff FunctionFlags) IsSet(wanted FunctionFlags) bool {
	return (ff & wanted) == wanted
}

const (
	// DefaultFlags is defined for clarity.
	DefaultFlags FunctionFlags = 0

	// ShortCircuit indicates that further validations should be skipped if
	// this validator fails. If there are multiple validators with this flag
	// set, they will ALL run, and if any of them fail, any non-short-circuit
	// validators will be skipped.  Most validators are not fatal.
	ShortCircuit FunctionFlags = 1 << iota

	// NonError indicates that a failure of this validator should not be
	// accumulated as an error, but should trigger other aspects of the failure
	// path (e.g. early return when combined with ShortCircuit).
	NonError
)

// Conditions defines what conditions must be true for a resource to be validated.
// If any of the conditions are not true, the resource is not validated.
type Conditions struct {
	// OptionEnabled specifies an option name that must be set to true for the condition to be true.
	OptionEnabled string

	// OptionDisabled specifies an option name that must be set to false for the condition to be true.
	OptionDisabled string
}

func (c Conditions) Empty() bool {
	return len(c.OptionEnabled) == 0 && len(c.OptionDisabled) == 0
}

// Identifier is a name that the generator will output as an identifier.
// Identifiers are generated using the RawNamer strategy.
type Identifier types.Name

// PrivateVar is a variable name that the generator will output as a private identifier.
// PrivateVars are generated using the PrivateNamer strategy.
type PrivateVar types.Name

// Function creates a FunctionGen for a given function name and extraArgs.
func Function(tagName string, flags FunctionFlags, function types.Name, extraArgs ...any) FunctionGen {
	return FunctionGen{
		TagName:  tagName,
		Flags:    flags,
		Function: function,
		Args:     extraArgs,
	}
}

// FunctionGen describes a function call that should be generated.
type FunctionGen struct {
	// TagName is the tag which triggered this function.
	TagName string

	// Cohort indicates a set of related functions which are processed
	// together.
	Cohort string

	// Flags holds the options for this validator function.
	Flags FunctionFlags

	// Function is the name of the function to call.
	Function types.Name

	// Args holds arguments to pass to the function, and may conatin:
	// - data literals comprised of maps, slices, strings, ints, floats, and bools
	// - types.Type (to reference any type in the universe)
	// - types.Member (to reference members of the current value)
	// - types.Identifier (to reference any identifier in the universe)
	// - validators.WrapperFunction (to call another validation function)
	// - validators.Literal (to pass a literal value)
	// - validators.FunctionLiteral (to pass a function literal)
	// - validators.PrivateVar (to reference a variable)
	//
	// See toGolangSourceDataLiteral for details.
	Args []any

	// TypeArgs assigns types to the type parameters of the function, for
	// generic function calls which require explicit type arguments.
	TypeArgs []types.Name

	// Conditions holds any conditions that must true for a field to be
	// validated by this function.
	Conditions Conditions

	// Comments holds optional comments that should be added to the generated
	// code (without the leading "//").
	Comments []string
}

// WithTypeArgs returns a derived FunctionGen with type arguments.
func (fg FunctionGen) WithTypeArgs(typeArgs ...types.Name) FunctionGen {
	fg.TypeArgs = typeArgs
	return fg
}

// WithConditions returns a derived FunctionGen with conditions.
func (fg FunctionGen) WithConditions(conditions Conditions) FunctionGen {
	fg.Conditions = conditions
	return fg
}

// WithComments returns a new FunctionGen with a comment.
func (fg FunctionGen) WithComments(comments ...string) FunctionGen {
	fg.Comments = append(fg.Comments, comments...)
	return fg
}

// WithComment returns a new FunctionGen with a comment.
func (fg FunctionGen) WithComment(comment string) FunctionGen {
	return fg.WithComments(comment)
}

// Variable creates a VariableGen for a given variable name and init value.
func Variable(variable PrivateVar, initializer any) VariableGen {
	return VariableGen{
		Variable:    variable,
		Initializer: initializer,
	}
}

type VariableGen struct {
	// Variable holds the variable identifier.
	Variable PrivateVar

	// Initializer is the value to initialize the variable with.
	// Initializer may be any function call or literal type supported by toGolangSourceDataLiteral.
	Initializer any
}

// WrapperFunction describes a function literal which has the fingerprint of a
// regular validation function (op, fldPath, obj, oldObj) and calls another
// validation function with the same signature, plus extra args if needed.
type WrapperFunction struct {
	Function FunctionGen
	ObjType  *types.Type
}

// Literal is a literal value that, when used as an argument to a validator,
// will be emitted without any further interpretation.  Use this with caution,
// it will not be subject to Namers.
type Literal string

// FunctionLiteral describes a function-literal expression that can be used as
// an argument to a validator.  Unlike WrapperFunction, this does not
// necessarily have the same signature as a regular validation function.
type FunctionLiteral struct {
	Parameters []ParamResult
	Results    []ParamResult
	Body       string
}

// StructLiteral represents a struct literal expression that can be used as
// an argument to a validator.
type StructLiteral struct {
	// Type is the type of the struct literal to be generated.
	Type types.Name
	// TypeArgs are the generic type arguments for the struct type.
	TypeArgs []*types.Type
	Fields   []StructLiteralField
}

// SliceLiteral represents a slice literal expression that can be used as
// an argument to a validator.
type SliceLiteral struct {
	// ElementType is the type of the elements in the slice.
	ElementType types.Name
	// ElementTypeArgs are the generic type arguments for the element type.
	ElementTypeArgs []*types.Type
	Elements        []any
}

type StructLiteralField struct {
	Name  string
	Value any
}

// ParamResult represents a parameter or a result of a function.
type ParamResult struct {
	Name string
	Type *types.Type
}

// typeCheck checks that the argument and value types of the tag match the types
// declared in the doc.
func typeCheck(tag codetags.Tag, doc TagDoc) error {
	for _, docArg := range doc.Args {
		hasArg := false
		for _, tagArg := range tag.Args {
			if tagArg.Name == docArg.Name {
				hasArg = true
				if docArg.Type != tagArg.Type {
					return fmt.Errorf("argument %q has wrong type: got %s, want %s",
						tagArg, tagArg.Type, docArg.Type)
				}
				break
			}
		}
		if !hasArg && docArg.Required {
			if docArg.Name == "" {
				return fmt.Errorf("missing required positional argument of type %s", docArg.Type)
			} else {
				return fmt.Errorf("missing named argument %q of type %s", docArg.Name, docArg.Type)
			}
		}
	}

	for _, tagArg := range tag.Args {
		if _, ok := doc.Arg(tagArg.Name); !ok {
			if !doc.AcceptsUnknownArgs {
				return fmt.Errorf("unrecognized named argument %q", tagArg)
			}
		}
	}
	if tag.ValueType == codetags.ValueTypeNone {
		if doc.PayloadsRequired {
			return fmt.Errorf("missing required tag value of type %s", doc.PayloadsType)
		}
	} else if doc.PayloadsType != codetags.ValueTypeRaw && tag.ValueType != doc.PayloadsType {
		return fmt.Errorf("tag value has wrong type: got %s, want %s", tag.ValueType, doc.PayloadsType)
	}
	return nil
}
