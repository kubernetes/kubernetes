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
	"sync"

	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Source interface representing a user-provided expression.
type Source interface {
	common.Source
}

// Ast representing the checked or unchecked expression, its source, and related metadata such as
// source position information.
type Ast struct {
	expr    *exprpb.Expr
	info    *exprpb.SourceInfo
	source  Source
	refMap  map[int64]*exprpb.Reference
	typeMap map[int64]*exprpb.Type
}

// Expr returns the proto serializable instance of the parsed/checked expression.
func (ast *Ast) Expr() *exprpb.Expr {
	return ast.expr
}

// IsChecked returns whether the Ast value has been successfully type-checked.
func (ast *Ast) IsChecked() bool {
	return ast.typeMap != nil && len(ast.typeMap) > 0
}

// SourceInfo returns character offset and newling position information about expression elements.
func (ast *Ast) SourceInfo() *exprpb.SourceInfo {
	return ast.info
}

// ResultType returns the output type of the expression if the Ast has been type-checked, else
// returns decls.Dyn as the parse step cannot infer the type.
func (ast *Ast) ResultType() *exprpb.Type {
	if !ast.IsChecked() {
		return decls.Dyn
	}
	return ast.typeMap[ast.expr.Id]
}

// Source returns a view of the input used to create the Ast. This source may be complete or
// constructed from the SourceInfo.
func (ast *Ast) Source() Source {
	return ast.source
}

// FormatType converts a type message into a string representation.
func FormatType(t *exprpb.Type) string {
	return checker.FormatCheckedType(t)
}

// Env encapsulates the context necessary to perform parsing, type checking, or generation of
// evaluable programs for different expressions.
type Env struct {
	Container    *containers.Container
	declarations []*exprpb.Decl
	macros       []parser.Macro
	adapter      ref.TypeAdapter
	provider     ref.TypeProvider
	features     map[int]bool
	// program options tied to the environment.
	progOpts []ProgramOption

	// Internal checker representation
	chk    *checker.Env
	chkErr error
	once   sync.Once
}

// NewEnv creates a program environment configured with the standard library of CEL functions and
// macros. The Env value returned can parse and check any CEL program which builds upon the core
// features documented in the CEL specification.
//
// See the EnvOption helper functions for the options that can be used to configure the
// environment.
func NewEnv(opts ...EnvOption) (*Env, error) {
	stdOpts := append([]EnvOption{StdLib()}, opts...)
	return NewCustomEnv(stdOpts...)
}

// NewCustomEnv creates a custom program environment which is not automatically configured with the
// standard library of functions and macros documented in the CEL spec.
//
// The purpose for using a custom environment might be for subsetting the standard library produced
// by the cel.StdLib() function. Subsetting CEL is a core aspect of its design that allows users to
// limit the compute and memory impact of a CEL program by controlling the functions and macros
// that may appear in a given expression.
//
// See the EnvOption helper functions for the options that can be used to configure the
// environment.
func NewCustomEnv(opts ...EnvOption) (*Env, error) {
	registry, err := types.NewRegistry()
	if err != nil {
		return nil, err
	}
	return (&Env{
		declarations: []*exprpb.Decl{},
		macros:       []parser.Macro{},
		Container:    containers.DefaultContainer,
		adapter:      registry,
		provider:     registry,
		features:     map[int]bool{},
		progOpts:     []ProgramOption{},
	}).configure(opts)
}

// Check performs type-checking on the input Ast and yields a checked Ast and/or set of Issues.
//
// Checking has failed if the returned Issues value and its Issues.Err() value are non-nil.
// Issues should be inspected if they are non-nil, but may not represent a fatal error.
//
// It is possible to have both non-nil Ast and Issues values returned from this call: however,
// the mere presence of an Ast does not imply that it is valid for use.
func (e *Env) Check(ast *Ast) (*Ast, *Issues) {
	// Note, errors aren't currently possible on the Ast to ParsedExpr conversion.
	pe, _ := AstToParsedExpr(ast)

	// Construct the internal checker env, erroring if there is an issue adding the declarations.
	e.once.Do(func() {
		ce := checker.NewEnv(e.Container, e.provider)
		ce.EnableDynamicAggregateLiterals(true)
		if e.HasFeature(FeatureDisableDynamicAggregateLiterals) {
			ce.EnableDynamicAggregateLiterals(false)
		}
		err := ce.Add(e.declarations...)
		if err != nil {
			e.chkErr = err
		} else {
			e.chk = ce
		}
	})
	// The once call will ensure that this value is set or nil for all invocations.
	if e.chkErr != nil {
		errs := common.NewErrors(ast.Source())
		errs.ReportError(common.NoLocation, e.chkErr.Error())
		return nil, NewIssues(errs)
	}

	res, errs := checker.Check(pe, ast.Source(), e.chk)
	if len(errs.GetErrors()) > 0 {
		return nil, NewIssues(errs)
	}
	// Manually create the Ast to ensure that the Ast source information (which may be more
	// detailed than the information provided by Check), is returned to the caller.
	return &Ast{
		source:  ast.Source(),
		expr:    res.GetExpr(),
		info:    res.GetSourceInfo(),
		refMap:  res.GetReferenceMap(),
		typeMap: res.GetTypeMap()}, nil
}

// Compile combines the Parse and Check phases CEL program compilation to produce an Ast and
// associated issues.
//
// If an error is encountered during parsing the Compile step will not continue with the Check
// phase. If non-error issues are encountered during Parse, they may be combined with any issues
// discovered during Check.
//
// Note, for parse-only uses of CEL use Parse.
func (e *Env) Compile(txt string) (*Ast, *Issues) {
	return e.CompileSource(common.NewTextSource(txt))
}

// CompileSource combines the Parse and Check phases CEL program compilation to produce an Ast and
// associated issues.
//
// If an error is encountered during parsing the CompileSource step will not continue with the
// Check phase. If non-error issues are encountered during Parse, they may be combined with any
// issues discovered during Check.
//
// Note, for parse-only uses of CEL use Parse.
func (e *Env) CompileSource(src common.Source) (*Ast, *Issues) {
	ast, iss := e.ParseSource(src)
	if iss.Err() != nil {
		return nil, iss
	}
	checked, iss2 := e.Check(ast)
	iss = iss.Append(iss2)
	if iss.Err() != nil {
		return nil, iss
	}
	return checked, iss
}

// Extend the current environment with additional options to produce a new Env.
//
// Note, the extended Env value should not share memory with the original. It is possible, however,
// that a CustomTypeAdapter or CustomTypeProvider options could provide values which are mutable.
// To ensure separation of state between extended environments either make sure the TypeAdapter and
// TypeProvider are immutable, or that their underlying implementations are based on the
// ref.TypeRegistry which provides a Copy method which will be invoked by this method.
func (e *Env) Extend(opts ...EnvOption) (*Env, error) {
	if e.chkErr != nil {
		return nil, e.chkErr
	}
	// Copy slices.
	decsCopy := make([]*exprpb.Decl, len(e.declarations))
	macsCopy := make([]parser.Macro, len(e.macros))
	progOptsCopy := make([]ProgramOption, len(e.progOpts))
	copy(decsCopy, e.declarations)
	copy(macsCopy, e.macros)
	copy(progOptsCopy, e.progOpts)

	// Copy the adapter / provider if they appear to be mutable.
	adapter := e.adapter
	provider := e.provider
	adapterReg, isAdapterReg := e.adapter.(ref.TypeRegistry)
	providerReg, isProviderReg := e.provider.(ref.TypeRegistry)
	// In most cases the provider and adapter will be a ref.TypeRegistry;
	// however, in the rare cases where they are not, they are assumed to
	// be immutable. Since it is possible to set the TypeProvider separately
	// from the TypeAdapter, the possible configurations which could use a
	// TypeRegistry as the base implementation are captured below.
	if isAdapterReg && isProviderReg {
		reg := providerReg.Copy()
		provider = reg
		// If the adapter and provider are the same object, set the adapter
		// to the same ref.TypeRegistry as the provider.
		if adapterReg == providerReg {
			adapter = reg
		} else {
			// Otherwise, make a copy of the adapter.
			adapter = adapterReg.Copy()
		}
	} else if isProviderReg {
		provider = providerReg.Copy()
	} else if isAdapterReg {
		adapter = adapterReg.Copy()
	}

	featuresCopy := make(map[int]bool, len(e.features))
	for k, v := range e.features {
		featuresCopy[k] = v
	}

	ext := &Env{
		Container:    e.Container,
		declarations: decsCopy,
		macros:       macsCopy,
		progOpts:     progOptsCopy,
		adapter:      adapter,
		features:     featuresCopy,
		provider:     provider,
	}
	return ext.configure(opts)
}

// HasFeature checks whether the environment enables the given feature
// flag, as enumerated in options.go.
func (e *Env) HasFeature(flag int) bool {
	_, has := e.features[flag]
	return has
}

// Parse parses the input expression value `txt` to a Ast and/or a set of Issues.
//
// This form of Parse creates a common.Source value for the input `txt` and forwards to the
// ParseSource method.
func (e *Env) Parse(txt string) (*Ast, *Issues) {
	src := common.NewTextSource(txt)
	return e.ParseSource(src)
}

// ParseSource parses the input source to an Ast and/or set of Issues.
//
// Parsing has failed if the returned Issues value and its Issues.Err() value is non-nil.
// Issues should be inspected if they are non-nil, but may not represent a fatal error.
//
// It is possible to have both non-nil Ast and Issues values returned from this call; however,
// the mere presence of an Ast does not imply that it is valid for use.
func (e *Env) ParseSource(src common.Source) (*Ast, *Issues) {
	res, errs := parser.ParseWithMacros(src, e.macros)
	if len(errs.GetErrors()) > 0 {
		return nil, &Issues{errs: errs}
	}
	// Manually create the Ast to ensure that the text source information is propagated on
	// subsequent calls to Check.
	return &Ast{
		source: Source(src),
		expr:   res.GetExpr(),
		info:   res.GetSourceInfo()}, nil
}

// Program generates an evaluable instance of the Ast within the environment (Env).
func (e *Env) Program(ast *Ast, opts ...ProgramOption) (Program, error) {
	optSet := e.progOpts
	if len(opts) != 0 {
		mergedOpts := []ProgramOption{}
		mergedOpts = append(mergedOpts, e.progOpts...)
		mergedOpts = append(mergedOpts, opts...)
		optSet = mergedOpts
	}
	return newProgram(e, ast, optSet)
}

// SetFeature sets the given feature flag, as enumerated in options.go.
func (e *Env) SetFeature(flag int) {
	e.features[flag] = true
}

// TypeAdapter returns the `ref.TypeAdapter` configured for the environment.
func (e *Env) TypeAdapter() ref.TypeAdapter {
	return e.adapter
}

// TypeProvider returns the `ref.TypeProvider` configured for the environment.
func (e *Env) TypeProvider() ref.TypeProvider {
	return e.provider
}

// UnknownVars returns an interpreter.PartialActivation which marks all variables
// declared in the Env as unknown AttributePattern values.
//
// Note, the UnknownVars will behave the same as an interpreter.EmptyActivation
// unless the PartialAttributes option is provided as a ProgramOption.
func (e *Env) UnknownVars() interpreter.PartialActivation {
	var unknownPatterns []*interpreter.AttributePattern
	for _, d := range e.declarations {
		switch d.GetDeclKind().(type) {
		case *exprpb.Decl_Ident:
			unknownPatterns = append(unknownPatterns,
				interpreter.NewAttributePattern(d.GetName()))
		}
	}
	part, _ := PartialVars(
		interpreter.EmptyActivation(),
		unknownPatterns...)
	return part
}

// ResidualAst takes an Ast and its EvalDetails to produce a new Ast which only contains the
// attribute references which are unknown.
//
// Residual expressions are beneficial in a few scenarios:
//
// - Optimizing constant expression evaluations away.
// - Indexing and pruning expressions based on known input arguments.
// - Surfacing additional requirements that are needed in order to complete an evaluation.
// - Sharing the evaluation of an expression across multiple machines/nodes.
//
// For example, if an expression targets a 'resource' and 'request' attribute and the possible
// values for the resource are known, a PartialActivation could mark the 'request' as an unknown
// interpreter.AttributePattern and the resulting ResidualAst would be reduced to only the parts
// of the expression that reference the 'request'.
//
// Note, the expression ids within the residual AST generated through this method have no
// correlation to the expression ids of the original AST.
//
// See the PartialVars helper for how to construct a PartialActivation.
//
// TODO: Consider adding an option to generate a Program.Residual to avoid round-tripping to an
// Ast format and then Program again.
func (e *Env) ResidualAst(a *Ast, details *EvalDetails) (*Ast, error) {
	pruned := interpreter.PruneAst(a.Expr(), details.State())
	expr, err := AstToString(ParsedExprToAst(&exprpb.ParsedExpr{Expr: pruned}))
	if err != nil {
		return nil, err
	}
	parsed, iss := e.Parse(expr)
	if iss != nil && iss.Err() != nil {
		return nil, iss.Err()
	}
	if !a.IsChecked() {
		return parsed, nil
	}
	checked, iss := e.Check(parsed)
	if iss != nil && iss.Err() != nil {
		return nil, iss.Err()
	}
	return checked, nil
}

// configure applies a series of EnvOptions to the current environment.
func (e *Env) configure(opts []EnvOption) (*Env, error) {
	// Customized the environment using the provided EnvOption values. If an error is
	// generated at any step this, will be returned as a nil Env with a non-nil error.
	var err error
	for _, opt := range opts {
		e, err = opt(e)
		if err != nil {
			return nil, err
		}
	}
	return e, nil
}

// Issues defines methods for inspecting the error details of parse and check calls.
//
// Note: in the future, non-fatal warnings and notices may be inspectable via the Issues struct.
type Issues struct {
	errs *common.Errors
}

// NewIssues returns an Issues struct from a common.Errors object.
func NewIssues(errs *common.Errors) *Issues {
	return &Issues{
		errs: errs,
	}
}

// Err returns an error value if the issues list contains one or more errors.
func (i *Issues) Err() error {
	if i == nil {
		return nil
	}
	if len(i.Errors()) > 0 {
		return errors.New(i.String())
	}
	return nil
}

// Errors returns the collection of errors encountered in more granular detail.
func (i *Issues) Errors() []common.Error {
	if i == nil {
		return []common.Error{}
	}
	return i.errs.GetErrors()
}

// Append collects the issues from another Issues struct into a new Issues object.
func (i *Issues) Append(other *Issues) *Issues {
	if i == nil {
		return other
	}
	return NewIssues(i.errs.Append(other.errs.GetErrors()))
}

// String converts the issues to a suitable display string.
func (i *Issues) String() string {
	if i == nil {
		return ""
	}
	return i.errs.ToDisplayString()
}
