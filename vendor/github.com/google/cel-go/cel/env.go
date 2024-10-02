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
	chkdecls "github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common"
	celast "github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Source interface representing a user-provided expression.
type Source = common.Source

// Ast representing the checked or unchecked expression, its source, and related metadata such as
// source position information.
type Ast struct {
	source Source
	impl   *celast.AST
}

// NativeRep converts the AST to a Go-native representation.
func (ast *Ast) NativeRep() *celast.AST {
	if ast == nil {
		return nil
	}
	return ast.impl
}

// Expr returns the proto serializable instance of the parsed/checked expression.
//
// Deprecated: prefer cel.AstToCheckedExpr() or cel.AstToParsedExpr() and call GetExpr()
// the result instead.
func (ast *Ast) Expr() *exprpb.Expr {
	if ast == nil {
		return nil
	}
	pbExpr, _ := celast.ExprToProto(ast.NativeRep().Expr())
	return pbExpr
}

// IsChecked returns whether the Ast value has been successfully type-checked.
func (ast *Ast) IsChecked() bool {
	return ast.NativeRep().IsChecked()
}

// SourceInfo returns character offset and newline position information about expression elements.
func (ast *Ast) SourceInfo() *exprpb.SourceInfo {
	if ast == nil {
		return nil
	}
	pbInfo, _ := celast.SourceInfoToProto(ast.NativeRep().SourceInfo())
	return pbInfo
}

// ResultType returns the output type of the expression if the Ast has been type-checked, else
// returns chkdecls.Dyn as the parse step cannot infer the type.
//
// Deprecated: use OutputType
func (ast *Ast) ResultType() *exprpb.Type {
	out := ast.OutputType()
	t, err := TypeToExprType(out)
	if err != nil {
		return chkdecls.Dyn
	}
	return t
}

// OutputType returns the output type of the expression if the Ast has been type-checked, else
// returns cel.DynType as the parse step cannot infer types.
func (ast *Ast) OutputType() *Type {
	if ast == nil {
		return types.ErrorType
	}
	return ast.NativeRep().GetType(ast.NativeRep().Expr().ID())
}

// Source returns a view of the input used to create the Ast. This source may be complete or
// constructed from the SourceInfo.
func (ast *Ast) Source() Source {
	if ast == nil {
		return nil
	}
	return ast.source
}

// FormatType converts a type message into a string representation.
//
// Deprecated: prefer FormatCELType
func FormatType(t *exprpb.Type) string {
	return checker.FormatCheckedType(t)
}

// FormatCELType formats a cel.Type value to a string representation.
//
// The type formatting is identical to FormatType.
func FormatCELType(t *Type) string {
	return checker.FormatCELType(t)
}

// Env encapsulates the context necessary to perform parsing, type checking, or generation of
// evaluable programs for different expressions.
type Env struct {
	Container       *containers.Container
	variables       []*decls.VariableDecl
	functions       map[string]*decls.FunctionDecl
	macros          []parser.Macro
	adapter         types.Adapter
	provider        types.Provider
	features        map[int]bool
	appliedFeatures map[int]bool
	libraries       map[string]bool
	validators      []ASTValidator
	costOptions     []checker.CostOption

	// Internal parser representation
	prsr     *parser.Parser
	prsrOpts []parser.Option

	// Internal checker representation
	chkMutex sync.Mutex
	chk      *checker.Env
	chkErr   error
	chkOnce  sync.Once
	chkOpts  []checker.Option

	// Program options tied to the environment
	progOpts []ProgramOption
}

// NewEnv creates a program environment configured with the standard library of CEL functions and
// macros. The Env value returned can parse and check any CEL program which builds upon the core
// features documented in the CEL specification.
//
// See the EnvOption helper functions for the options that can be used to configure the
// environment.
func NewEnv(opts ...EnvOption) (*Env, error) {
	// Extend the statically configured standard environment, disabling eager validation to ensure
	// the cost of setup for the environment is still just as cheap as it is in v0.11.x and earlier
	// releases. The user provided options can easily re-enable the eager validation as they are
	// processed after this default option.
	stdOpts := append([]EnvOption{EagerlyValidateDeclarations(false)}, opts...)
	env, err := getStdEnv()
	if err != nil {
		return nil, err
	}
	return env.Extend(stdOpts...)
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
		variables:       []*decls.VariableDecl{},
		functions:       map[string]*decls.FunctionDecl{},
		macros:          []parser.Macro{},
		Container:       containers.DefaultContainer,
		adapter:         registry,
		provider:        registry,
		features:        map[int]bool{},
		appliedFeatures: map[int]bool{},
		libraries:       map[string]bool{},
		validators:      []ASTValidator{},
		progOpts:        []ProgramOption{},
		costOptions:     []checker.CostOption{},
	}).configure(opts)
}

// Check performs type-checking on the input Ast and yields a checked Ast and/or set of Issues.
// If any `ASTValidators` are configured on the environment, they will be applied after a valid
// type-check result. If any issues are detected, the validators will provide them on the
// output Issues object.
//
// Either checking or validation has failed if the returned Issues value and its Issues.Err()
// value are non-nil. Issues should be inspected if they are non-nil, but may not represent a
// fatal error.
//
// It is possible to have both non-nil Ast and Issues values returned from this call: however,
// the mere presence of an Ast does not imply that it is valid for use.
func (e *Env) Check(ast *Ast) (*Ast, *Issues) {
	// Construct the internal checker env, erroring if there is an issue adding the declarations.
	chk, err := e.initChecker()
	if err != nil {
		errs := common.NewErrors(ast.Source())
		errs.ReportError(common.NoLocation, err.Error())
		return nil, NewIssuesWithSourceInfo(errs, ast.NativeRep().SourceInfo())
	}

	checked, errs := checker.Check(ast.NativeRep(), ast.Source(), chk)
	if len(errs.GetErrors()) > 0 {
		return nil, NewIssuesWithSourceInfo(errs, ast.NativeRep().SourceInfo())
	}
	// Manually create the Ast to ensure that the Ast source information (which may be more
	// detailed than the information provided by Check), is returned to the caller.
	ast = &Ast{
		source: ast.Source(),
		impl:   checked}

	// Avoid creating a validator config if it's not needed.
	if len(e.validators) == 0 {
		return ast, nil
	}

	// Generate a validator configuration from the set of configured validators.
	vConfig := newValidatorConfig()
	for _, v := range e.validators {
		if cv, ok := v.(ASTValidatorConfigurer); ok {
			cv.Configure(vConfig)
		}
	}
	// Apply additional validators on the type-checked result.
	iss := NewIssuesWithSourceInfo(errs, ast.NativeRep().SourceInfo())
	for _, v := range e.validators {
		v.Validate(e, vConfig, checked, iss)
	}
	if iss.Err() != nil {
		return nil, iss
	}
	return ast, nil
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
func (e *Env) CompileSource(src Source) (*Ast, *Issues) {
	ast, iss := e.ParseSource(src)
	if iss.Err() != nil {
		return nil, iss
	}
	checked, iss2 := e.Check(ast)
	if iss2.Err() != nil {
		return nil, iss2
	}
	return checked, iss2
}

// Extend the current environment with additional options to produce a new Env.
//
// Note, the extended Env value should not share memory with the original. It is possible, however,
// that a CustomTypeAdapter or CustomTypeProvider options could provide values which are mutable.
// To ensure separation of state between extended environments either make sure the TypeAdapter and
// TypeProvider are immutable, or that their underlying implementations are based on the
// ref.TypeRegistry which provides a Copy method which will be invoked by this method.
func (e *Env) Extend(opts ...EnvOption) (*Env, error) {
	chk, chkErr := e.getCheckerOrError()
	if chkErr != nil {
		return nil, chkErr
	}

	prsrOptsCopy := make([]parser.Option, len(e.prsrOpts))
	copy(prsrOptsCopy, e.prsrOpts)

	// The type-checker is configured with Declarations. The declarations may either be provided
	// as options which have not yet been validated, or may come from a previous checker instance
	// whose types have already been validated.
	chkOptsCopy := make([]checker.Option, len(e.chkOpts))
	copy(chkOptsCopy, e.chkOpts)

	// Copy the declarations if needed.
	if chk != nil {
		// If the type-checker has already been instantiated, then the e.declarations have been
		// validated within the chk instance.
		chkOptsCopy = append(chkOptsCopy, checker.ValidatedDeclarations(chk))
	}
	varsCopy := make([]*decls.VariableDecl, len(e.variables))
	copy(varsCopy, e.variables)

	// Copy macros and program options
	macsCopy := make([]parser.Macro, len(e.macros))
	progOptsCopy := make([]ProgramOption, len(e.progOpts))
	copy(macsCopy, e.macros)
	copy(progOptsCopy, e.progOpts)

	// Copy the adapter / provider if they appear to be mutable.
	adapter := e.adapter
	provider := e.provider
	adapterReg, isAdapterReg := e.adapter.(*types.Registry)
	providerReg, isProviderReg := e.provider.(*types.Registry)
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
	appliedFeaturesCopy := make(map[int]bool, len(e.appliedFeatures))
	for k, v := range e.appliedFeatures {
		appliedFeaturesCopy[k] = v
	}
	funcsCopy := make(map[string]*decls.FunctionDecl, len(e.functions))
	for k, v := range e.functions {
		funcsCopy[k] = v
	}
	libsCopy := make(map[string]bool, len(e.libraries))
	for k, v := range e.libraries {
		libsCopy[k] = v
	}
	validatorsCopy := make([]ASTValidator, len(e.validators))
	copy(validatorsCopy, e.validators)
	costOptsCopy := make([]checker.CostOption, len(e.costOptions))
	copy(costOptsCopy, e.costOptions)

	ext := &Env{
		Container:       e.Container,
		variables:       varsCopy,
		functions:       funcsCopy,
		macros:          macsCopy,
		progOpts:        progOptsCopy,
		adapter:         adapter,
		features:        featuresCopy,
		appliedFeatures: appliedFeaturesCopy,
		libraries:       libsCopy,
		validators:      validatorsCopy,
		provider:        provider,
		chkOpts:         chkOptsCopy,
		prsrOpts:        prsrOptsCopy,
		costOptions:     costOptsCopy,
	}
	return ext.configure(opts)
}

// HasFeature checks whether the environment enables the given feature
// flag, as enumerated in options.go.
func (e *Env) HasFeature(flag int) bool {
	enabled, has := e.features[flag]
	return has && enabled
}

// HasLibrary returns whether a specific SingletonLibrary has been configured in the environment.
func (e *Env) HasLibrary(libName string) bool {
	configured, exists := e.libraries[libName]
	return exists && configured
}

// Libraries returns a list of SingletonLibrary that have been configured in the environment.
func (e *Env) Libraries() []string {
	libraries := make([]string, 0, len(e.libraries))
	for libName := range e.libraries {
		libraries = append(libraries, libName)
	}
	return libraries
}

// HasFunction returns whether a specific function has been configured in the environment
func (e *Env) HasFunction(functionName string) bool {
	_, ok := e.functions[functionName]
	return ok
}

// Functions returns map of Functions, keyed by function name, that have been configured in the environment.
func (e *Env) Functions() map[string]*decls.FunctionDecl {
	return e.functions
}

// HasValidator returns whether a specific ASTValidator has been configured in the environment.
func (e *Env) HasValidator(name string) bool {
	for _, v := range e.validators {
		if v.Name() == name {
			return true
		}
	}
	return false
}

// Parse parses the input expression value `txt` to a Ast and/or a set of Issues.
//
// This form of Parse creates a Source value for the input `txt` and forwards to the
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
func (e *Env) ParseSource(src Source) (*Ast, *Issues) {
	parsed, errs := e.prsr.Parse(src)
	if len(errs.GetErrors()) > 0 {
		return nil, &Issues{errs: errs}
	}
	return &Ast{source: src, impl: parsed}, nil
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

// CELTypeAdapter returns the `types.Adapter` configured for the environment.
func (e *Env) CELTypeAdapter() types.Adapter {
	return e.adapter
}

// CELTypeProvider returns the `types.Provider` configured for the environment.
func (e *Env) CELTypeProvider() types.Provider {
	return e.provider
}

// TypeAdapter returns the `ref.TypeAdapter` configured for the environment.
//
// Deprecated: use CELTypeAdapter()
func (e *Env) TypeAdapter() ref.TypeAdapter {
	return e.adapter
}

// TypeProvider returns the `ref.TypeProvider` configured for the environment.
//
// Deprecated: use CELTypeProvider()
func (e *Env) TypeProvider() ref.TypeProvider {
	if legacyProvider, ok := e.provider.(ref.TypeProvider); ok {
		return legacyProvider
	}
	return &interopLegacyTypeProvider{Provider: e.provider}
}

// UnknownVars returns an interpreter.PartialActivation which marks all variables declared in the
// Env as unknown AttributePattern values.
//
// Note, the UnknownVars will behave the same as an interpreter.EmptyActivation unless the
// PartialAttributes option is provided as a ProgramOption.
func (e *Env) UnknownVars() interpreter.PartialActivation {
	act := interpreter.EmptyActivation()
	part, _ := PartialVars(act, e.computeUnknownVars(act)...)
	return part
}

// PartialVars returns an interpreter.PartialActivation where all variables not in the input variable
// set, but which have been configured in the environment, are marked as unknown.
//
// The `vars` value may either be an interpreter.Activation or any valid input to the
// interpreter.NewActivation call.
//
// Note, this is equivalent to calling cel.PartialVars and manually configuring the set of unknown
// variables. For more advanced use cases of partial state where portions of an object graph, rather
// than top-level variables, are missing the PartialVars() method may be a more suitable choice.
//
// Note, the PartialVars will behave the same as an interpreter.EmptyActivation unless the
// PartialAttributes option is provided as a ProgramOption.
func (e *Env) PartialVars(vars any) (interpreter.PartialActivation, error) {
	act, err := interpreter.NewActivation(vars)
	if err != nil {
		return nil, err
	}
	return PartialVars(act, e.computeUnknownVars(act)...)
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
	pruned := interpreter.PruneAst(a.impl.Expr(), a.impl.SourceInfo().MacroCalls(), details.State())
	newAST := &Ast{source: a.Source(), impl: pruned}
	expr, err := AstToString(newAST)
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

// EstimateCost estimates the cost of a type checked CEL expression using the length estimates of input data and
// extension functions provided by estimator.
func (e *Env) EstimateCost(ast *Ast, estimator checker.CostEstimator, opts ...checker.CostOption) (checker.CostEstimate, error) {
	extendedOpts := make([]checker.CostOption, 0, len(e.costOptions))
	extendedOpts = append(extendedOpts, opts...)
	extendedOpts = append(extendedOpts, e.costOptions...)
	return checker.Cost(ast.impl, estimator, extendedOpts...)
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

	// If the default UTC timezone fix has been enabled, make sure the library is configured
	e, err = e.maybeApplyFeature(featureDefaultUTCTimeZone, Lib(timeUTCLibrary{}))
	if err != nil {
		return nil, err
	}

	// Configure the parser.
	prsrOpts := []parser.Option{}
	prsrOpts = append(prsrOpts, e.prsrOpts...)
	prsrOpts = append(prsrOpts, parser.Macros(e.macros...))

	if e.HasFeature(featureEnableMacroCallTracking) {
		prsrOpts = append(prsrOpts, parser.PopulateMacroCalls(true))
	}
	if e.HasFeature(featureVariadicLogicalASTs) {
		prsrOpts = append(prsrOpts, parser.EnableVariadicOperatorASTs(true))
	}
	e.prsr, err = parser.NewParser(prsrOpts...)
	if err != nil {
		return nil, err
	}

	// Ensure that the checker init happens eagerly rather than lazily.
	if e.HasFeature(featureEagerlyValidateDeclarations) {
		_, err := e.initChecker()
		if err != nil {
			return nil, err
		}
	}

	return e, nil
}

func (e *Env) initChecker() (*checker.Env, error) {
	e.chkOnce.Do(func() {
		chkOpts := []checker.Option{}
		chkOpts = append(chkOpts, e.chkOpts...)
		chkOpts = append(chkOpts,
			checker.CrossTypeNumericComparisons(
				e.HasFeature(featureCrossTypeNumericComparisons)))

		ce, err := checker.NewEnv(e.Container, e.provider, chkOpts...)
		if err != nil {
			e.setCheckerOrError(nil, err)
			return
		}
		// Add the statically configured declarations.
		err = ce.AddIdents(e.variables...)
		if err != nil {
			e.setCheckerOrError(nil, err)
			return
		}
		// Add the function declarations which are derived from the FunctionDecl instances.
		for _, fn := range e.functions {
			if fn.IsDeclarationDisabled() {
				continue
			}
			err = ce.AddFunctions(fn)
			if err != nil {
				e.setCheckerOrError(nil, err)
				return
			}
		}
		// Add function declarations here separately.
		e.setCheckerOrError(ce, nil)
	})
	return e.getCheckerOrError()
}

// setCheckerOrError sets the checker.Env or error state in a concurrency-safe manner
func (e *Env) setCheckerOrError(chk *checker.Env, chkErr error) {
	e.chkMutex.Lock()
	e.chk = chk
	e.chkErr = chkErr
	e.chkMutex.Unlock()
}

// getCheckerOrError gets the checker.Env or error state in a concurrency-safe manner
func (e *Env) getCheckerOrError() (*checker.Env, error) {
	e.chkMutex.Lock()
	defer e.chkMutex.Unlock()
	return e.chk, e.chkErr
}

// maybeApplyFeature determines whether the feature-guarded option is enabled, and if so applies
// the feature if it has not already been enabled.
func (e *Env) maybeApplyFeature(feature int, option EnvOption) (*Env, error) {
	if !e.HasFeature(feature) {
		return e, nil
	}
	_, applied := e.appliedFeatures[feature]
	if applied {
		return e, nil
	}
	e, err := option(e)
	if err != nil {
		return nil, err
	}
	// record that the feature has been applied since it will generate declarations
	// and functions which will be propagated on Extend() calls and which should only
	// be registered once.
	e.appliedFeatures[feature] = true
	return e, nil
}

// computeUnknownVars determines a set of missing variables based on the input activation and the
// environment's configured declaration set.
func (e *Env) computeUnknownVars(vars interpreter.Activation) []*interpreter.AttributePattern {
	var unknownPatterns []*interpreter.AttributePattern
	for _, v := range e.variables {
		varName := v.Name()
		if _, found := vars.ResolveName(varName); found {
			continue
		}
		unknownPatterns = append(unknownPatterns, interpreter.NewAttributePattern(varName))
	}
	return unknownPatterns
}

// Error type which references an expression id, a location within source, and a message.
type Error = common.Error

// Issues defines methods for inspecting the error details of parse and check calls.
//
// Note: in the future, non-fatal warnings and notices may be inspectable via the Issues struct.
type Issues struct {
	errs *common.Errors
	info *celast.SourceInfo
}

// NewIssues returns an Issues struct from a common.Errors object.
func NewIssues(errs *common.Errors) *Issues {
	return NewIssuesWithSourceInfo(errs, nil)
}

// NewIssuesWithSourceInfo returns an Issues struct from a common.Errors object with SourceInfo metatata
// which can be used with the `ReportErrorAtID` method for additional error reports within the context
// information that's inferred from an expression id.
func NewIssuesWithSourceInfo(errs *common.Errors, info *celast.SourceInfo) *Issues {
	return &Issues{
		errs: errs,
		info: info,
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
func (i *Issues) Errors() []*Error {
	if i == nil {
		return []*Error{}
	}
	return i.errs.GetErrors()
}

// Append collects the issues from another Issues struct into a new Issues object.
func (i *Issues) Append(other *Issues) *Issues {
	if i == nil {
		return other
	}
	if other == nil || i == other {
		return i
	}
	return NewIssuesWithSourceInfo(i.errs.Append(other.errs.GetErrors()), i.info)
}

// String converts the issues to a suitable display string.
func (i *Issues) String() string {
	if i == nil {
		return ""
	}
	return i.errs.ToDisplayString()
}

// ReportErrorAtID reports an error message with an optional set of formatting arguments.
//
// The source metadata for the expression at `id`, if present, is attached to the error report.
// To ensure that source metadata is attached to error reports, use NewIssuesWithSourceInfo.
func (i *Issues) ReportErrorAtID(id int64, message string, args ...any) {
	i.errs.ReportErrorAtID(id, i.info.GetStartLocation(id), message, args...)
}

// getStdEnv lazy initializes the CEL standard environment.
func getStdEnv() (*Env, error) {
	stdEnvInit.Do(func() {
		stdEnv, stdEnvErr = NewCustomEnv(StdLib(), EagerlyValidateDeclarations(true))
	})
	return stdEnv, stdEnvErr
}

// interopCELTypeProvider layers support for the types.Provider interface on top of a ref.TypeProvider.
type interopCELTypeProvider struct {
	ref.TypeProvider
}

// FindStructType returns a types.Type instance for the given fully-qualified typeName if one exists.
//
// This method proxies to the underlying ref.TypeProvider's FindType method and converts protobuf type
// into a native type representation. If the conversion fails, the type is listed as not found.
func (p *interopCELTypeProvider) FindStructType(typeName string) (*types.Type, bool) {
	if et, found := p.FindType(typeName); found {
		t, err := types.ExprTypeToType(et)
		if err != nil {
			return nil, false
		}
		return t, true
	}
	return nil, false
}

// FindStructFieldNames returns an empty set of field for the interop provider.
//
// To inspect the field names, migrate to a `types.Provider` implementation.
func (p *interopCELTypeProvider) FindStructFieldNames(typeName string) ([]string, bool) {
	return []string{}, false
}

// FindStructFieldType returns a types.FieldType instance for the given fully-qualified typeName and field
// name, if one exists.
//
// This method proxies to the underlying ref.TypeProvider's FindFieldType method and converts protobuf type
// into a native type representation. If the conversion fails, the type is listed as not found.
func (p *interopCELTypeProvider) FindStructFieldType(structType, fieldName string) (*types.FieldType, bool) {
	if ft, found := p.FindFieldType(structType, fieldName); found {
		t, err := types.ExprTypeToType(ft.Type)
		if err != nil {
			return nil, false
		}
		return &types.FieldType{
			Type:    t,
			IsSet:   ft.IsSet,
			GetFrom: ft.GetFrom,
		}, true
	}
	return nil, false
}

// interopLegacyTypeProvider layers support for the ref.TypeProvider interface on top of a types.Provider.
type interopLegacyTypeProvider struct {
	types.Provider
}

// FindType retruns the protobuf Type representation for the input type name if one exists.
//
// This method proxies to the underlying types.Provider FindStructType method and converts the types.Type
// value to a protobuf Type representation.
//
// Failure to convert the type will result in the type not being found.
func (p *interopLegacyTypeProvider) FindType(typeName string) (*exprpb.Type, bool) {
	if t, found := p.FindStructType(typeName); found {
		et, err := types.TypeToExprType(t)
		if err != nil {
			return nil, false
		}
		return et, true
	}
	return nil, false
}

// FindFieldType returns the protobuf-based FieldType representation for the input type name and field,
// if one exists.
//
// This call proxies to the types.Provider FindStructFieldType method and converts the types.FIeldType
// value to a protobuf-based ref.FieldType representation if found.
//
// Failure to convert the FieldType will result in the field not being found.
func (p *interopLegacyTypeProvider) FindFieldType(structType, fieldName string) (*ref.FieldType, bool) {
	if cft, found := p.FindStructFieldType(structType, fieldName); found {
		et, err := types.TypeToExprType(cft.Type)
		if err != nil {
			return nil, false
		}
		return &ref.FieldType{
			Type:    et,
			IsSet:   cft.IsSet,
			GetFrom: cft.GetFrom,
		}, true
	}
	return nil, false
}

var (
	stdEnvInit sync.Once
	stdEnv     *Env
	stdEnvErr  error
)
