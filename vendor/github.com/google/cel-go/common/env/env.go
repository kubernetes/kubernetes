// Copyright 2025 Google LLC
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

// Package env provides a representation of a CEL environment.
package env

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/types"
)

// NewConfig creates an instance of a YAML serializable CEL environment configuration.
func NewConfig(name string) *Config {
	return &Config{
		Name: name,
	}
}

// Config represents a serializable form of the CEL environment configuration.
//
// Note: custom validations, feature flags, and performance tuning parameters are not (yet)
// considered part of the core CEL environment configuration and should be managed separately
// until a common convention for such settings is developed.
type Config struct {
	Name            string           `yaml:"name,omitempty"`
	Description     string           `yaml:"description,omitempty"`
	Container       string           `yaml:"container,omitempty"`
	Imports         []*Import        `yaml:"imports,omitempty"`
	StdLib          *LibrarySubset   `yaml:"stdlib,omitempty"`
	Extensions      []*Extension     `yaml:"extensions,omitempty"`
	ContextVariable *ContextVariable `yaml:"context_variable,omitempty"`
	Variables       []*Variable      `yaml:"variables,omitempty"`
	Functions       []*Function      `yaml:"functions,omitempty"`
	Validators      []*Validator     `yaml:"validators,omitempty"`
	Features        []*Feature       `yaml:"features,omitempty"`
}

// Validate validates the whole configuration is well-formed.
func (c *Config) Validate() error {
	if c == nil {
		return nil
	}
	var errs []error
	for _, imp := range c.Imports {
		if err := imp.Validate(); err != nil {
			errs = append(errs, err)
		}
	}
	if err := c.StdLib.Validate(); err != nil {
		errs = append(errs, err)
	}
	for _, ext := range c.Extensions {
		if err := ext.Validate(); err != nil {
			errs = append(errs, err)
		}
	}
	if err := c.ContextVariable.Validate(); err != nil {
		errs = append(errs, err)
	}
	if c.ContextVariable != nil && len(c.Variables) != 0 {
		errs = append(errs, errors.New("invalid config: either context variable or variables may be set, but not both"))
	}
	for _, v := range c.Variables {
		if err := v.Validate(); err != nil {
			errs = append(errs, err)
		}
	}
	for _, fn := range c.Functions {
		if err := fn.Validate(); err != nil {
			errs = append(errs, err)
		}
	}
	for _, feat := range c.Features {
		if err := feat.Validate(); err != nil {
			errs = append(errs, err)
		}
	}
	for _, val := range c.Validators {
		if err := val.Validate(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// SetContainer configures the container name for this configuration.
func (c *Config) SetContainer(container string) *Config {
	c.Container = container
	return c
}

// AddVariableDecls adds one or more variables to the config, converting them to serializable values first.
//
// VariableDecl inputs are expected to be well-formed.
func (c *Config) AddVariableDecls(vars ...*decls.VariableDecl) *Config {
	convVars := make([]*Variable, len(vars))
	for i, v := range vars {
		if v == nil {
			continue
		}
		cv := NewVariable(v.Name(), SerializeTypeDesc(v.Type()))
		cv.Description = v.Description()
		convVars[i] = cv
	}
	return c.AddVariables(convVars...)
}

// AddVariables adds one or more variables to the config.
func (c *Config) AddVariables(vars ...*Variable) *Config {
	c.Variables = append(c.Variables, vars...)
	return c
}

// SetContextVariable configures the ContextVariable for this configuration.
func (c *Config) SetContextVariable(ctx *ContextVariable) *Config {
	c.ContextVariable = ctx
	return c
}

// AddFunctionDecls adds one or more functions to the config, converting them to serializable values first.
//
// FunctionDecl inputs are expected to be well-formed.
func (c *Config) AddFunctionDecls(funcs ...*decls.FunctionDecl) *Config {
	convFuncs := make([]*Function, len(funcs))
	for i, fn := range funcs {
		if fn == nil {
			continue
		}
		overloads := make([]*Overload, 0, len(fn.OverloadDecls()))
		for _, o := range fn.OverloadDecls() {
			overloadID := o.ID()
			args := make([]*TypeDesc, 0, len(o.ArgTypes()))
			for _, a := range o.ArgTypes() {
				args = append(args, SerializeTypeDesc(a))
			}
			ret := SerializeTypeDesc(o.ResultType())
			var overload *Overload
			if o.IsMemberFunction() {
				overload = NewMemberOverload(overloadID, args[0], args[1:], ret)
			} else {
				overload = NewOverload(overloadID, args, ret)
			}
			exampleCount := len(o.Examples())
			if exampleCount > 0 {
				overload.Examples = o.Examples()
			}
			overloads = append(overloads, overload)
		}
		cf := NewFunction(fn.Name(), overloads...)
		cf.Description = fn.Description()
		convFuncs[i] = cf
	}
	return c.AddFunctions(convFuncs...)
}

// AddFunctions adds one or more functions to the config.
func (c *Config) AddFunctions(funcs ...*Function) *Config {
	c.Functions = append(c.Functions, funcs...)
	return c
}

// SetStdLib configures the LibrarySubset for the standard library.
func (c *Config) SetStdLib(subset *LibrarySubset) *Config {
	c.StdLib = subset
	return c
}

// AddImports appends a set of imports to the config.
func (c *Config) AddImports(imps ...*Import) *Config {
	c.Imports = append(c.Imports, imps...)
	return c
}

// AddExtensions appends a set of extensions to the config.
func (c *Config) AddExtensions(exts ...*Extension) *Config {
	c.Extensions = append(c.Extensions, exts...)
	return c
}

// AddValidators appends one or more validators to the config.
func (c *Config) AddValidators(vals ...*Validator) *Config {
	c.Validators = append(c.Validators, vals...)
	return c
}

// AddFeatures appends one or more features to the config.
func (c *Config) AddFeatures(feats ...*Feature) *Config {
	c.Features = append(c.Features, feats...)
	return c
}

// NewImport returns a serializable import value from the qualified type name.
func NewImport(name string) *Import {
	return &Import{Name: name}
}

// Import represents a type name that will be appreviated by its simple name using
// the cel.Abbrevs() option.
type Import struct {
	Name string `yaml:"name"`
}

// Validate validates the import configuration is well-formed.
func (imp *Import) Validate() error {
	if imp == nil {
		return errors.New("invalid import: nil")
	}
	if imp.Name == "" {
		return errors.New("invalid import: missing type name")
	}
	return nil
}

// NewVariable returns a serializable variable from a name and type definition
func NewVariable(name string, t *TypeDesc) *Variable {
	return NewVariableWithDoc(name, t, "")
}

// NewVariableWithDoc returns a serializable variable from a name, type definition, and doc string.
func NewVariableWithDoc(name string, t *TypeDesc, doc string) *Variable {
	return &Variable{Name: name, TypeDesc: t, Description: doc}
}

// Variable represents a typed variable declaration which will be published via the
// cel.VariableDecls() option.
type Variable struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`

	// Type represents the type declaration for the variable.
	//
	// Deprecated: use the embedded *TypeDesc fields directly.
	Type *TypeDesc `yaml:"type,omitempty"`

	// TypeDesc is an embedded set of fields allowing for the specification of the Variable type.
	*TypeDesc `yaml:",inline"`
}

// Validate validates the variable configuration is well-formed.
func (v *Variable) Validate() error {
	if v == nil {
		return errors.New("invalid variable: nil")
	}
	if v.Name == "" {
		return errors.New("invalid variable: missing variable name")
	}
	if err := v.GetType().Validate(); err != nil {
		return fmt.Errorf("invalid variable %q: %w", v.Name, err)
	}
	return nil
}

// GetType returns the variable type description.
//
// Note, if both the embedded TypeDesc and the field Type are non-nil, the embedded TypeDesc will
// take precedence.
func (v *Variable) GetType() *TypeDesc {
	if v == nil {
		return nil
	}
	if v.TypeDesc != nil {
		return v.TypeDesc
	}
	if v.Type != nil {
		return v.Type
	}
	return nil
}

// AsCELVariable converts the serializable form of the Variable into a CEL environment declaration.
func (v *Variable) AsCELVariable(tp types.Provider) (*decls.VariableDecl, error) {
	if err := v.Validate(); err != nil {
		return nil, err
	}
	t, err := v.GetType().AsCELType(tp)
	if err != nil {
		return nil, fmt.Errorf("invalid variable %q: %w", v.Name, err)
	}
	return decls.NewVariableWithDoc(v.Name, t, v.Description), nil
}

// NewContextVariable returns a serializable context variable with a specific type name.
func NewContextVariable(typeName string) *ContextVariable {
	return &ContextVariable{TypeName: typeName}
}

// ContextVariable represents a structured message whose fields are to be treated as the top-level
// variable identifiers within CEL expressions.
type ContextVariable struct {
	// TypeName represents the fully qualified typename of the context variable.
	// Currently, only protobuf types are supported.
	TypeName string `yaml:"type_name"`
}

// Validate validates the context-variable configuration is well-formed.
func (ctx *ContextVariable) Validate() error {
	if ctx == nil {
		return nil
	}
	if ctx.TypeName == "" {
		return errors.New("invalid context variable: missing type name")
	}
	return nil
}

// NewFunction creates a serializable function and overload set.
func NewFunction(name string, overloads ...*Overload) *Function {
	return &Function{Name: name, Overloads: overloads}
}

// NewFunctionWithDoc creates a serializable function and overload set.
func NewFunctionWithDoc(name, doc string, overloads ...*Overload) *Function {
	return &Function{Name: name, Description: doc, Overloads: overloads}
}

// Function represents the serializable format of a function and its overloads.
type Function struct {
	Name        string      `yaml:"name"`
	Description string      `yaml:"description,omitempty"`
	Overloads   []*Overload `yaml:"overloads,omitempty"`
}

// Validate validates the function configuration is well-formed.
func (fn *Function) Validate() error {
	if fn == nil {
		return errors.New("invalid function: nil")
	}
	if fn.Name == "" {
		return errors.New("invalid function: missing function name")
	}
	if len(fn.Overloads) == 0 {
		return fmt.Errorf("invalid function %q: missing overloads", fn.Name)
	}
	var errs []error
	for _, o := range fn.Overloads {
		if err := o.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("invalid function %q: %w", fn.Name, err))
		}
	}
	return errors.Join(errs...)
}

// AsCELFunction converts the serializable form of the Function into CEL environment declaration.
func (fn *Function) AsCELFunction(tp types.Provider) (*decls.FunctionDecl, error) {
	if err := fn.Validate(); err != nil {
		return nil, err
	}
	opts := make([]decls.FunctionOpt, 0, len(fn.Overloads)+1)
	for _, o := range fn.Overloads {
		opt, err := o.AsFunctionOption(tp)
		opts = append(opts, opt)
		if err != nil {
			return nil, fmt.Errorf("invalid function %q: %w", fn.Name, err)
		}
	}
	if len(fn.Description) != 0 {
		opts = append(opts, decls.FunctionDocs(fn.Description))
	}
	return decls.NewFunction(fn.Name, opts...)
}

// NewOverload returns a new serializable representation of a global overload.
func NewOverload(id string, args []*TypeDesc, ret *TypeDesc, examples ...string) *Overload {
	return &Overload{ID: id, Args: args, Return: ret, Examples: examples}
}

// NewMemberOverload returns a new serializable representation of a member (receiver) overload.
func NewMemberOverload(id string, target *TypeDesc, args []*TypeDesc, ret *TypeDesc, examples ...string) *Overload {
	return &Overload{ID: id, Target: target, Args: args, Return: ret, Examples: examples}
}

// Overload represents the serializable format of a function overload.
type Overload struct {
	ID       string      `yaml:"id"`
	Examples []string    `yaml:"examples,omitempty"`
	Target   *TypeDesc   `yaml:"target,omitempty"`
	Args     []*TypeDesc `yaml:"args,omitempty"`
	Return   *TypeDesc   `yaml:"return,omitempty"`
}

// Validate validates the overload configuration is well-formed.
func (od *Overload) Validate() error {
	if od == nil {
		return errors.New("invalid overload: nil")
	}
	if od.ID == "" {
		return errors.New("invalid overload: missing overload id")
	}
	var errs []error
	if od.Target != nil {
		if err := od.Target.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("invalid overload %q target: %w", od.ID, err))
		}
	}
	for i, arg := range od.Args {
		if err := arg.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("invalid overload %q arg[%d]: %w", od.ID, i, err))
		}
	}
	if err := od.Return.Validate(); err != nil {
		errs = append(errs, fmt.Errorf("invalid overload %q return: %w", od.ID, err))
	}
	return errors.Join(errs...)
}

// AsFunctionOption converts the serializable form of the Overload into a function declaration option.
func (od *Overload) AsFunctionOption(tp types.Provider) (decls.FunctionOpt, error) {
	if err := od.Validate(); err != nil {
		return nil, err
	}
	args := make([]*types.Type, len(od.Args))
	var err error
	var errs []error
	for i, a := range od.Args {
		args[i], err = a.AsCELType(tp)
		if err != nil {
			errs = append(errs, err)
		}
	}
	result, err := od.Return.AsCELType(tp)
	if err != nil {
		errs = append(errs, err)
	}
	if od.Target != nil {
		t, err := od.Target.AsCELType(tp)
		if err != nil {
			return nil, errors.Join(append(errs, err)...)
		}
		args = append([]*types.Type{t}, args...)
		return decls.MemberOverload(od.ID, args, result), nil
	}
	if len(errs) != 0 {
		return nil, errors.Join(errs...)
	}
	return decls.Overload(od.ID, args, result, decls.OverloadExamples(od.Examples...)), nil
}

// NewExtension creates a serializable Extension from a name and version string.
func NewExtension(name string, version uint32) *Extension {
	versionString := "latest"
	if version < math.MaxUint32 {
		versionString = strconv.FormatUint(uint64(version), 10)
	}
	return &Extension{
		Name:    name,
		Version: versionString,
	}
}

// Extension represents a named and optionally versioned extension library configured in the environment.
type Extension struct {
	// Name is either the LibraryName() or some short-hand simple identifier which is understood by the config-handler.
	Name string `yaml:"name"`

	// Version may either be an unsigned long value or the string 'latest'. If empty, the value is treated as '0'.
	Version string `yaml:"version,omitempty"`
}

// Validate validates the extension configuration is well-formed.
func (e *Extension) Validate() error {
	_, err := e.VersionNumber()
	return err
}

// VersionNumber returns the parsed version string, or an error if the version cannot be parsed.
func (e *Extension) VersionNumber() (uint32, error) {
	if e == nil {
		return 0, fmt.Errorf("invalid extension: nil")
	}
	if e.Name == "" {
		return 0, fmt.Errorf("invalid extension: missing name")
	}
	if e.Version == "latest" {
		return math.MaxUint32, nil
	}
	if e.Version == "" {
		return 0, nil
	}
	ver, err := strconv.ParseUint(e.Version, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid extension %q version: %w", e.Name, err)
	}
	return uint32(ver), nil
}

// NewLibrarySubset returns an empty library subsetting config which permits all library features.
func NewLibrarySubset() *LibrarySubset {
	return &LibrarySubset{}
}

// LibrarySubset indicates a subset of the macros and function supported by a subsettable library.
type LibrarySubset struct {
	// Disabled indicates whether the library has been disabled, typically only used for
	// default-enabled libraries like stdlib.
	Disabled bool `yaml:"disabled,omitempty"`

	// DisableMacros disables macros for the given library.
	DisableMacros bool `yaml:"disable_macros,omitempty"`

	// IncludeMacros specifies a set of macro function names to include in the subset.
	IncludeMacros []string `yaml:"include_macros,omitempty"`

	// ExcludeMacros specifies a set of macro function names to exclude from the subset.
	// Note: if IncludeMacros is non-empty, then ExcludeFunctions is ignored.
	ExcludeMacros []string `yaml:"exclude_macros,omitempty"`

	// IncludeFunctions specifies a set of functions to include in the subset.
	//
	// Note: the overloads specified in the subset need only specify their ID.
	// Note: if IncludeFunctions is non-empty, then ExcludeFunctions is ignored.
	IncludeFunctions []*Function `yaml:"include_functions,omitempty"`

	// ExcludeFunctions specifies the set of functions to exclude from the subset.
	//
	// Note: the overloads specified in the subset need only specify their ID.
	ExcludeFunctions []*Function `yaml:"exclude_functions,omitempty"`
}

// Validate validates the library configuration is well-formed.
//
// For example, setting both the IncludeMacros and ExcludeMacros together could be confusing
// and create a broken expectation, likewise for IncludeFunctions and ExcludeFunctions.
func (lib *LibrarySubset) Validate() error {
	if lib == nil {
		return nil
	}
	var errs []error
	if len(lib.IncludeMacros) != 0 && len(lib.ExcludeMacros) != 0 {
		errs = append(errs, errors.New("invalid subset: cannot both include and exclude macros"))
	}
	if len(lib.IncludeFunctions) != 0 && len(lib.ExcludeFunctions) != 0 {
		errs = append(errs, errors.New("invalid subset: cannot both include and exclude functions"))
	}
	return errors.Join(errs...)
}

// SubsetFunction produces a function declaration which matches the supported subset, or nil
// if the function is not supported by the LibrarySubset.
//
// For IncludeFunctions, if the function does not specify a set of overloads to include, the
// whole function definition is included. If overloads are set, then a new function which
// includes only the specified overloads is produced.
//
// For ExcludeFunctions, if the function does not specify a set of overloads to exclude, the
// whole function definition is excluded. If overloads are set, then a new function which
// includes only the permitted overloads is produced.
func (lib *LibrarySubset) SubsetFunction(fn *decls.FunctionDecl) (*decls.FunctionDecl, bool) {
	// When lib is null, it should indicate that all values are included in the subset.
	if lib == nil {
		return fn, true
	}
	if lib.Disabled {
		return nil, false
	}
	if len(lib.IncludeFunctions) != 0 {
		for _, include := range lib.IncludeFunctions {
			if include.Name != fn.Name() {
				continue
			}
			if len(include.Overloads) == 0 {
				return fn, true
			}
			overloadIDs := make([]string, len(include.Overloads))
			for i, o := range include.Overloads {
				overloadIDs[i] = o.ID
			}
			return fn.Subset(decls.IncludeOverloads(overloadIDs...)), true
		}
		return nil, false
	}
	if len(lib.ExcludeFunctions) != 0 {
		for _, exclude := range lib.ExcludeFunctions {
			if exclude.Name != fn.Name() {
				continue
			}
			if len(exclude.Overloads) == 0 {
				return nil, false
			}
			overloadIDs := make([]string, len(exclude.Overloads))
			for i, o := range exclude.Overloads {
				overloadIDs[i] = o.ID
			}
			return fn.Subset(decls.ExcludeOverloads(overloadIDs...)), true
		}
		return fn, true
	}
	return fn, true
}

// SubsetMacro indicates whether the macro function should be included in the library subset.
func (lib *LibrarySubset) SubsetMacro(macroFunction string) bool {
	// When lib is null, it should indicate that all values are included in the subset.
	if lib == nil {
		return true
	}
	if lib.Disabled || lib.DisableMacros {
		return false
	}
	if len(lib.IncludeMacros) != 0 {
		for _, name := range lib.IncludeMacros {
			if name == macroFunction {
				return true
			}
		}
		return false
	}
	if len(lib.ExcludeMacros) != 0 {
		for _, name := range lib.ExcludeMacros {
			if name == macroFunction {
				return false
			}
		}
		return true
	}
	return true
}

// SetDisabled disables or enables the library.
func (lib *LibrarySubset) SetDisabled(value bool) *LibrarySubset {
	lib.Disabled = value
	return lib
}

// SetDisableMacros disables the macros for the library.
func (lib *LibrarySubset) SetDisableMacros(value bool) *LibrarySubset {
	lib.DisableMacros = value
	return lib
}

// AddIncludedMacros allow-lists one or more macros by function name.
//
// Note, this option will override any excluded macros.
func (lib *LibrarySubset) AddIncludedMacros(macros ...string) *LibrarySubset {
	lib.IncludeMacros = append(lib.IncludeMacros, macros...)
	return lib
}

// AddExcludedMacros deny-lists one or more macros by function name.
func (lib *LibrarySubset) AddExcludedMacros(macros ...string) *LibrarySubset {
	lib.ExcludeMacros = append(lib.ExcludeMacros, macros...)
	return lib
}

// AddIncludedFunctions allow-lists one or more functions from the subset.
//
// Note, this option will override any excluded functions.
func (lib *LibrarySubset) AddIncludedFunctions(funcs ...*Function) *LibrarySubset {
	lib.IncludeFunctions = append(lib.IncludeFunctions, funcs...)
	return lib
}

// AddExcludedFunctions deny-lists one or more functions from the subset.
func (lib *LibrarySubset) AddExcludedFunctions(funcs ...*Function) *LibrarySubset {
	lib.ExcludeFunctions = append(lib.ExcludeFunctions, funcs...)
	return lib
}

// NewValidator returns a named Validator instance.
func NewValidator(name string) *Validator {
	return &Validator{Name: name}
}

// Validator represents a named validator with an optional map-based configuration object.
//
// Note: the map-keys must directly correspond to the internal representation of the original
// validator, and should only use primitive scalar types as values at this time.
type Validator struct {
	Name   string         `yaml:"name"`
	Config map[string]any `yaml:"config,omitempty"`
}

// Validate validates the configuration of the validator object.
func (v *Validator) Validate() error {
	if v == nil {
		return errors.New("invalid validator: nil")
	}
	if v.Name == "" {
		return errors.New("invalid validator: missing name")
	}
	return nil
}

// SetConfig sets the set of map key-value pairs associated with this validator's configuration.
func (v *Validator) SetConfig(config map[string]any) *Validator {
	v.Config = config
	return v
}

// ConfigValue retrieves the value associated with the config key name, if one exists.
func (v *Validator) ConfigValue(name string) (any, bool) {
	if v == nil {
		return nil, false
	}
	value, found := v.Config[name]
	return value, found
}

// NewFeature creates a new feature flag with a boolean enablement flag.
func NewFeature(name string, enabled bool) *Feature {
	return &Feature{Name: name, Enabled: enabled}
}

// Feature represents a named boolean feature flag supported by CEL.
type Feature struct {
	Name    string `yaml:"name"`
	Enabled bool   `yaml:"enabled"`
}

// Validate validates whether the feature is well-configured.
func (feat *Feature) Validate() error {
	if feat == nil {
		return errors.New("invalid feature: nil")
	}
	if feat.Name == "" {
		return errors.New("invalid feature: missing name")
	}
	return nil
}

// NewTypeDesc describes a simple or complex type with parameters.
func NewTypeDesc(typeName string, params ...*TypeDesc) *TypeDesc {
	return &TypeDesc{TypeName: typeName, Params: params}
}

// NewTypeParam describe a type-param type.
func NewTypeParam(paramName string) *TypeDesc {
	return &TypeDesc{TypeName: paramName, IsTypeParam: true}
}

// TypeDesc represents the serializable format of a CEL *types.Type value.
type TypeDesc struct {
	TypeName    string      `yaml:"type_name"`
	Params      []*TypeDesc `yaml:"params,omitempty"`
	IsTypeParam bool        `yaml:"is_type_param,omitempty"`
}

// String implements the strings.Stringer interface method.
func (td *TypeDesc) String() string {
	ps := make([]string, len(td.Params))
	for i, p := range td.Params {
		ps[i] = p.String()
	}
	typeName := td.TypeName
	if len(ps) != 0 {
		typeName = fmt.Sprintf("%s(%s)", typeName, strings.Join(ps, ","))
	}
	return typeName
}

// Validate validates the type configuration is well-formed.
func (td *TypeDesc) Validate() error {
	if td == nil {
		return errors.New("invalid type: nil")
	}
	if td.TypeName == "" {
		return errors.New("invalid type: missing type name")
	}
	if td.IsTypeParam && len(td.Params) != 0 {
		return errors.New("invalid type: param type cannot have parameters")
	}
	switch td.TypeName {
	case "list":
		if len(td.Params) != 1 {
			return fmt.Errorf("invalid type: list expects 1 parameter, got %d", len(td.Params))
		}
		return td.Params[0].Validate()
	case "map":
		if len(td.Params) != 2 {
			return fmt.Errorf("invalid type: map expects 2 parameters, got %d", len(td.Params))
		}
		if err := td.Params[0].Validate(); err != nil {
			return err
		}
		if err := td.Params[1].Validate(); err != nil {
			return err
		}
	case "optional_type":
		if len(td.Params) != 1 {
			return fmt.Errorf("invalid type: optional_type expects 1 parameter, got %d", len(td.Params))
		}
		return td.Params[0].Validate()
	default:
	}
	return nil
}

// AsCELType converts the serializable object to a *types.Type value.
func (td *TypeDesc) AsCELType(tp types.Provider) (*types.Type, error) {
	err := td.Validate()
	if err != nil {
		return nil, err
	}
	switch td.TypeName {
	case "dyn":
		return types.DynType, nil
	case "map":
		kt, err := td.Params[0].AsCELType(tp)
		if err != nil {
			return nil, err
		}
		vt, err := td.Params[1].AsCELType(tp)
		if err != nil {
			return nil, err
		}
		return types.NewMapType(kt, vt), nil
	case "list":
		et, err := td.Params[0].AsCELType(tp)
		if err != nil {
			return nil, err
		}
		return types.NewListType(et), nil
	case "optional_type":
		et, err := td.Params[0].AsCELType(tp)
		if err != nil {
			return nil, err
		}
		return types.NewOptionalType(et), nil
	default:
		if td.IsTypeParam {
			return types.NewTypeParamType(td.TypeName), nil
		}
		if msgType, found := tp.FindStructType(td.TypeName); found {
			// First parameter is the type name.
			return msgType.Parameters()[0], nil
		}
		t, found := tp.FindIdent(td.TypeName)
		if !found {
			return nil, fmt.Errorf("undefined type name: %q", td.TypeName)
		}
		_, ok := t.(*types.Type)
		if ok && len(td.Params) == 0 {
			return t.(*types.Type), nil
		}
		params := make([]*types.Type, len(td.Params))
		for i, p := range td.Params {
			params[i], err = p.AsCELType(tp)
			if err != nil {
				return nil, err
			}
		}
		return types.NewOpaqueType(td.TypeName, params...), nil
	}
}

// SerializeTypeDesc converts a CEL native *types.Type to a serializable TypeDesc.
func SerializeTypeDesc(t *types.Type) *TypeDesc {
	typeName := t.TypeName()
	if t.Kind() == types.TypeParamKind {
		return NewTypeParam(typeName)
	}
	if t != types.NullType && t.IsAssignableType(types.NullType) {
		if wrapperTypeName, found := wrapperTypes[t.Kind()]; found {
			return NewTypeDesc(wrapperTypeName)
		}
	}
	var params []*TypeDesc
	for _, p := range t.Parameters() {
		params = append(params, SerializeTypeDesc(p))
	}
	return NewTypeDesc(typeName, params...)
}

var wrapperTypes = map[types.Kind]string{
	types.BoolKind:   "google.protobuf.BoolValue",
	types.BytesKind:  "google.protobuf.BytesValue",
	types.DoubleKind: "google.protobuf.DoubleValue",
	types.IntKind:    "google.protobuf.Int64Value",
	types.StringKind: "google.protobuf.StringValue",
	types.UintKind:   "google.protobuf.UInt64Value",
}
