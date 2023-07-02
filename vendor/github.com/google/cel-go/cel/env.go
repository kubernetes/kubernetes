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
type Source = common.Source

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

// SourceInfo returns character offset and newline position information about expression elements.
func (ast *Ast) SourceInfo() *exprpb.SourceInfo {
	return ast.info
}

// ResultType returns the output type of the expression if the Ast has been type-checked, else
// returns decls.Dyn as the parse step cannot infer the type.
//
// Deprecated: use OutputType
func (ast *Ast) ResultType() *exprpb.Type {
	if !ast.IsChecked() {
		return decls.Dyn
	}
	return ast.typeMap[ast.expr.GetId()]
}

// OutputType returns the output type of the expression if the Ast has been type-checked, else
// returns cel.DynType as the parse step cannot infer types.
func (ast *Ast) OutputType() *Type {
	t, err := ExprTypeToType(ast.ResultType())
	if err != nil {
		return DynType
	}
	return t
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
	Container       *containers.Container
	functions       map[string]*functionDecl
	declarations    []*exprpb.Decl
	macros          []parser.Macro
	adapter         ref.TypeAdapter
	provider        ref.TypeProvider
	features        map[int]bool
	appliedFeatures map[int]bool
	libraries       map[string]bool

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
		declarations:    []*exprpb.Decl{},
		functions:       map[string]*functionDecl{},
		macros:          []parser.Macro{},
		Container:       containers.DefaultContainer,
		adapter:         registry,
		provider:        registry,
		features:        map[int]bool{},
		appliedFeatures: map[int]bool{},
		libraries:       map[string]bool{},
		progOpts:        []ProgramOption{},
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
	chk, err := e.initChecker()
	if err != nil {
		errs := common.NewErrors(ast.Source())
		errs.ReportError(common.NoLocation, err.Error())
		return nil, NewIssues(errs)
	}

	res, errs := checker.Check(pe, ast.Source(), chk)
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
	decsCopy := []*exprpb.Decl{}
	if chk != nil {
		// If the type-checker has already been instantiated, then the e.declarations have been
		// validated within the chk instance.
		chkOptsCopy = append(chkOptsCopy, checker.ValidatedDeclarations(chk))
	} else {
		// If the type-checker has not been instantiated, ensure the unvalidated declarations are
		// provided to the extended Env instance.
		decsCopy = make([]*exprpb.Decl, len(e.declarations))
		copy(decsCopy, e.declarations)
	}

	// Copy macros and program options
	macsCopy := make([]parser.Macro, len(e.macros))
	progOptsCopy := make([]ProgramOption, len(e.progOpts))
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
	appliedFeaturesCopy := make(map[int]bool, len(e.appliedFeatures))
	for k, v := range e.appliedFeatures {
		appliedFeaturesCopy[k] = v
	}
	funcsCopy := make(map[string]*functionDecl, len(e.functions))
	for k, v := range e.functions {
		funcsCopy[k] = v
	}
	libsCopy := make(map[string]bool, len(e.libraries))
	for k, v := range e.libraries {
		libsCopy[k] = v
	}

	ext := &Env{
		Container:       e.Container,
		declarations:    decsCopy,
		functions:       funcsCopy,
		macros:          macsCopy,
		progOpts:        progOptsCopy,
		adapter:         adapter,
		features:        featuresCopy,
		appliedFeatures: appliedFeaturesCopy,
		libraries:       libsCopy,
		provider:        provider,
		chkOpts:         chkOptsCopy,
		prsrOpts:        prsrOptsCopy,
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
	res, errs := e.prsr.Parse(src)
	if len(errs.GetErrors()) > 0 {
		return nil, &Issues{errs: errs}
	}
	// Manually create the Ast to ensure that the text source information is propagated on
	// subsequent calls to Check.
	return &Ast{
		source: src,
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
	pruned := interpreter.PruneAst(a.Expr(), a.SourceInfo().GetMacroCalls(), details.State())
	expr, err := AstToString(ParsedExprToAst(pruned))
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
	checked, err := AstToCheckedExpr(ast)
	if err != nil {
		return checker.CostEstimate{}, fmt.Errorf("EsimateCost could not inspect Ast: %v", err)
	}
	return checker.Cost(checked, estimator, opts...)
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

	// Initialize all of the functions configured within the environment.
	for _, fn := range e.functions {
		err = fn.init()
		if err != nil {
			return nil, err
		}
	}

	// Configure the parser.
	prsrOpts := []parser.Option{}
	prsrOpts = append(prsrOpts, e.prsrOpts...)
	prsrOpts = append(prsrOpts, parser.Macros(e.macros...))

	if e.HasFeature(featureEnableMacroCallTracking) {
		prsrOpts = append(prsrOpts, parser.PopulateMacroCalls(true))
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
			checker.HomogeneousAggregateLiterals(
				e.HasFeature(featureDisableDynamicAggregateLiterals)),
			checker.CrossTypeNumericComparisons(
				e.HasFeature(featureCrossTypeNumericComparisons)))

		ce, err := checker.NewEnv(e.Container, e.provider, chkOpts...)
		if err != nil {
			e.setCheckerOrError(nil, err)
			return
		}
		// Add the statically configured declarations.
		err = ce.Add(e.declarations...)
		if err != nil {
			e.setCheckerOrError(nil, err)
			return
		}
		// Add the function declarations which are derived from the FunctionDecl instances.
		for _, fn := range e.functions {
			fnDecl, err := functionDeclToExprDecl(fn)
			if err != nil {
				e.setCheckerOrError(nil, err)
				return
			}
			err = ce.Add(fnDecl)
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
	if other == nil {
		return i
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

// getStdEnv lazy initializes the CEL standard environment.
func getStdEnv() (*Env, error) {
	stdEnvInit.Do(func() {
		stdEnv, stdEnvErr = NewCustomEnv(StdLib(), EagerlyValidateDeclarations(true))
	})
	return stdEnv, stdEnvErr
}

var (
	stdEnvInit sync.Once
	stdEnv     *Env
	stdEnvErr  error
)
