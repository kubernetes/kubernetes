package ruleguard

import (
	"go/ast"
	"go/build"
	"go/token"
	"go/types"
	"io"

	"github.com/quasilyte/go-ruleguard/ruleguard/ir"
)

// Engine is the main ruleguard package API object.
//
// First, load some ruleguard files with Load() to build a rule set.
// Then use Run() to execute the rules.
//
// It's advised to have only 1 engine per application as it does a lot of caching.
// The Run() method is synchronized, so it can be used concurrently.
//
// An Engine must be created with NewEngine() function.
type Engine struct {
	impl *engine

	// BuildContext can be used as an override for build.Default context.
	// Used during the Go packages resolving.
	//
	// Use Engine.InferBuildContext() to create a sensible default
	// for this field that is better than build.Default.
	// We're not using this by default to avoid the excessive work
	// if you already have a properly initialized build.Context object.
	//
	// nil will result in build.Default usage.
	BuildContext *build.Context
}

// NewEngine creates an engine with empty rule set.
func NewEngine() *Engine {
	return &Engine{impl: newEngine()}
}

func (e *Engine) InferBuildContext() {
	e.BuildContext = inferBuildContext()
}

// Load reads a ruleguard file from r and adds it to the engine rule set.
//
// Load() is not thread-safe, especially if used concurrently with Run() method.
// It's advised to Load() all ruleguard files under a critical section (like sync.Once)
// and then use Run() to execute all of them.
func (e *Engine) Load(ctx *LoadContext, filename string, r io.Reader) error {
	return e.impl.Load(ctx, e.BuildContext, filename, r)
}

// LoadFromIR is like Load(), but it takes already parsed IR file as an input.
//
// This method can be useful if you're trying to embed a precompiled rules file
// into your binary.
func (e *Engine) LoadFromIR(ctx *LoadContext, filename string, f *ir.File) error {
	return e.impl.LoadFromIR(ctx, e.BuildContext, filename, f)
}

// LoadedGroups returns information about all currently loaded rule groups.
func (e *Engine) LoadedGroups() []GoRuleGroup {
	return e.impl.LoadedGroups()
}

// Run executes all loaded rules on a given file.
// Matched rules invoke `RunContext.Report()` method.
//
// Run() is thread-safe, unless used in parallel with Load(),
// which modifies the engine state.
func (e *Engine) Run(ctx *RunContext, f *ast.File) error {
	return e.impl.Run(ctx, e.BuildContext, f)
}

type LoadContext struct {
	DebugFilter  string
	DebugImports bool
	DebugPrint   func(string)

	// GroupFilter is called for every rule group being parsed.
	// If function returns false, that group will not be included
	// in the resulting rules set.
	// Nil filter accepts all rule groups.
	GroupFilter func(string) bool

	Fset *token.FileSet
}

type RunContext struct {
	Debug        string
	DebugImports bool
	DebugPrint   func(string)

	Types *types.Info
	Sizes types.Sizes
	Fset  *token.FileSet
	Pkg   *types.Package

	// Report is a function that is called for every successful ruleguard match.
	// The pointer to ReportData is reused, it should not be kept.
	// If you want to keep it after Report() returns, make a copy.
	Report func(*ReportData)

	GoVersion GoVersion
}

type ReportData struct {
	RuleInfo   GoRuleInfo
	Node       ast.Node
	Message    string
	Suggestion *Suggestion

	// Experimental: fields below are part of the experiment.
	// They'll probably be removed or changed over time.

	Func *ast.FuncDecl
}

type Suggestion struct {
	From        token.Pos
	To          token.Pos
	Replacement []byte
}

type GoRuleInfo struct {
	// Line is a line inside a file that defined this rule.
	Line int

	// Group is a function that contains this rule.
	Group *GoRuleGroup
}

type GoRuleGroup struct {
	// Name is a function name associated with this rule group.
	Name string

	// Pos is a location where this rule group was defined.
	Pos token.Position

	// Line is a source code line number inside associated file.
	// A pair of Filename:Line form a conventional location string.
	Line int

	// Filename is a file that defined this rule group.
	Filename string

	// DocTags contains a list of keys from the `gorules:tags` comment.
	DocTags []string

	// DocSummary is a short one sentence description.
	// Filled from the `doc:summary` pragma content.
	DocSummary string

	// DocBefore is a code snippet of code that will violate rule.
	// Filled from the `doc:before` pragma content.
	DocBefore string

	// DocAfter is a code snippet of fixed code that complies to the rule.
	// Filled from the `doc:after` pragma content.
	DocAfter string

	// DocNote is an optional caution message or advice.
	// Usually, it's used to reference some external resource, like
	// issue on the GitHub.
	// Filled from the `doc:note` pragma content.
	DocNote string
}

// ImportError is returned when a ruleguard file references a package that cannot be imported.
type ImportError struct {
	msg string
	err error
}

func (e *ImportError) Error() string { return e.msg }
func (e *ImportError) Unwrap() error { return e.err }
