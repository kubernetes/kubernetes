/*
Copyright 2021 The Kubernetes Authors.

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

package pkg

import (
	"flag"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"path"
	"strconv"
	"strings"

	"golang.org/x/exp/utf8string"
	"golang.org/x/tools/go/analysis"
)

const (
	structuredCheck  = "structured"
	parametersCheck  = "parameters"
	contextualCheck  = "contextual"
	withHelpersCheck = "with-helpers"
)

type checks map[string]*bool

type config struct {
	enabled       checks
	fileOverrides RegexpFilter
}

func (c config) isEnabled(check string, filename string) bool {
	return c.fileOverrides.Enabled(check, *c.enabled[check], filename)
}

// Analyser creates a new logcheck analyser.
func Analyser() *analysis.Analyzer {
	c := config{
		enabled: checks{
			structuredCheck:  new(bool),
			parametersCheck:  new(bool),
			contextualCheck:  new(bool),
			withHelpersCheck: new(bool),
		},
	}
	c.fileOverrides.validChecks = map[string]bool{}
	for key := range c.enabled {
		c.fileOverrides.validChecks[key] = true
	}
	logcheckFlags := flag.NewFlagSet("", flag.ExitOnError)
	prefix := "check-"
	logcheckFlags.BoolVar(c.enabled[structuredCheck], prefix+structuredCheck, true, `When true, logcheck will warn about calls to unstructured
klog methods (Info, Infof, Error, Errorf, Warningf, etc).`)
	logcheckFlags.BoolVar(c.enabled[parametersCheck], prefix+parametersCheck, true, `When true, logcheck will check parameters of structured logging calls.`)
	logcheckFlags.BoolVar(c.enabled[contextualCheck], prefix+contextualCheck, false, `When true, logcheck will only allow log calls for contextual logging (retrieving a Logger from klog or the context and logging through that) and warn about all others.`)
	logcheckFlags.BoolVar(c.enabled[withHelpersCheck], prefix+withHelpersCheck, false, `When true, logcheck will warn about direct calls to WithName, WithValues and NewContext.`)
	logcheckFlags.Var(&c.fileOverrides, "config", `A file which overrides the global settings for checks on a per-file basis via regular expressions.`)

	// Use env variables as defaults. This is necessary when used as plugin
	// for golangci-lint because of
	// https://github.com/golangci/golangci-lint/issues/1512.
	for key, enabled := range c.enabled {
		envVarName := "LOGCHECK_" + strings.ToUpper(strings.ReplaceAll(string(key), "-", "_"))
		if value, ok := os.LookupEnv(envVarName); ok {
			v, err := strconv.ParseBool(value)
			if err != nil {
				panic(fmt.Errorf("%s=%q: %v", envVarName, value, err))
			}
			*enabled = v
		}
	}
	if value, ok := os.LookupEnv("LOGCHECK_CONFIG"); ok {
		if err := c.fileOverrides.Set(value); err != nil {
			panic(fmt.Errorf("LOGCHECK_CONFIG=%q: %v", value, err))
		}
	}

	return &analysis.Analyzer{
		Name: "logcheck",
		Doc:  "Tool to check logging calls.",
		Run: func(pass *analysis.Pass) (interface{}, error) {
			return run(pass, &c)
		},
		Flags: *logcheckFlags,
	}
}

func run(pass *analysis.Pass, c *config) (interface{}, error) {
	for _, file := range pass.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.CallExpr:
				// We are intrested in function calls, as we want to detect klog.* calls
				// passing all function calls to checkForFunctionExpr
				checkForFunctionExpr(n, pass, c)
			case *ast.FuncType:
				checkForContextAndLogger(n, n.Params, pass, c)
			case *ast.IfStmt:
				checkForIfEnabled(n, pass, c)
			}

			return true
		})
	}
	return nil, nil
}

// checkForFunctionExpr checks for unstructured logging function, prints error if found any.
func checkForFunctionExpr(fexpr *ast.CallExpr, pass *analysis.Pass, c *config) {
	fun := fexpr.Fun
	args := fexpr.Args

	/* we are extracting external package function calls e.g. klog.Infof fmt.Printf
	   and eliminating calls like setLocalHost()
	   basically function calls that has selector expression like .
	*/
	if selExpr, ok := fun.(*ast.SelectorExpr); ok {
		// extracting function Name like Infof
		fName := selExpr.Sel.Name

		filename := pass.Pkg.Path() + "/" + path.Base(pass.Fset.Position(fexpr.Pos()).Filename)

		// Now we need to determine whether it is coming from klog.
		if isKlog(selExpr.X, pass) {
			if c.isEnabled(contextualCheck, filename) && !isContextualCall(fName) {
				pass.Report(analysis.Diagnostic{
					Pos:     fun.Pos(),
					Message: fmt.Sprintf("function %q should not be used, convert to contextual logging", fName),
				})
				return
			}

			// Matching if any unstructured logging function is used.
			if c.isEnabled(structuredCheck, filename) && isUnstructured(fName) {
				pass.Report(analysis.Diagnostic{
					Pos:     fun.Pos(),
					Message: fmt.Sprintf("unstructured logging function %q should not be used", fName),
				})
				return
			}

			if c.isEnabled(parametersCheck, filename) {
				// if format specifier is used, check for arg length will most probably fail
				// so check for format specifier first and skip if found
				if checkForFormatSpecifier(fexpr, pass) {
					return
				}
				if fName == "InfoS" {
					isKeysValid(args[1:], fun, pass, fName)
				} else if fName == "ErrorS" {
					isKeysValid(args[2:], fun, pass, fName)
				}

				// Also check structured calls.
				if c.isEnabled(parametersCheck, filename) {
					checkForFormatSpecifier(fexpr, pass)
				}
			}
		} else if isGoLogger(selExpr.X, pass) {
			if c.isEnabled(parametersCheck, filename) {
				checkForFormatSpecifier(fexpr, pass)
				switch fName {
				case "WithValues":
					isKeysValid(args, fun, pass, fName)
				case "Info":
					isKeysValid(args[1:], fun, pass, fName)
				case "Error":
					isKeysValid(args[2:], fun, pass, fName)
				}
			}
			if c.isEnabled(withHelpersCheck, filename) {
				switch fName {
				case "WithValues", "WithName":
					pass.Report(analysis.Diagnostic{
						Pos:     fun.Pos(),
						Message: fmt.Sprintf("function %q should be called through klogr.Logger%s", fName, fName),
					})
				}
			}
		} else if fName == "NewContext" &&
			isPackage(selExpr.X, "github.com/go-logr/logr", pass) &&
			c.isEnabled(withHelpersCheck, filename) {
			pass.Report(analysis.Diagnostic{
				Pos:     fun.Pos(),
				Message: fmt.Sprintf("function %q should be called through klogr.NewContext", fName),
			})
		}

	}
}

// isKlogVerbose returns true if the type of the expression is klog.Verbose (=
// the result of klog.V).
func isKlogVerbose(expr ast.Expr, pass *analysis.Pass) bool {
	if typeAndValue, ok := pass.TypesInfo.Types[expr]; ok {
		switch t := typeAndValue.Type.(type) {
		case *types.Named:
			if typeName := t.Obj(); typeName != nil {
				if pkg := typeName.Pkg(); pkg != nil {
					if typeName.Name() == "Verbose" && pkg.Path() == "k8s.io/klog/v2" {
						return true
					}
				}
			}
		}
	}
	return false
}

// isKlog checks whether an expression is klog.Verbose or the klog package itself.
func isKlog(expr ast.Expr, pass *analysis.Pass) bool {
	// For klog.V(1) and klogV := klog.V(1) we can decide based on the type.
	if isKlogVerbose(expr, pass) {
		return true
	}

	// In "klog.Info", "klog" is a package identifier. It doesn't need to
	// be "klog" because here we look up the actual package.
	return isPackage(expr, "k8s.io/klog/v2", pass)
}

// isPackage checks whether an expression is an identifier that refers
// to a specific package like k8s.io/klog/v2.
func isPackage(expr ast.Expr, packagePath string, pass *analysis.Pass) bool {
	if ident, ok := expr.(*ast.Ident); ok {
		if object, ok := pass.TypesInfo.Uses[ident]; ok {
			switch object := object.(type) {
			case *types.PkgName:
				pkg := object.Imported()
				if pkg.Path() == packagePath {
					return true
				}
			}
		}
	}

	return false
}

// isGoLogger checks whether an expression is logr.Logger.
func isGoLogger(expr ast.Expr, pass *analysis.Pass) bool {
	if typeAndValue, ok := pass.TypesInfo.Types[expr]; ok {
		switch t := typeAndValue.Type.(type) {
		case *types.Named:
			if typeName := t.Obj(); typeName != nil {
				if pkg := typeName.Pkg(); pkg != nil {
					if typeName.Name() == "Logger" && pkg.Path() == "github.com/go-logr/logr" {
						return true
					}
				}
			}
		}
	}
	return false
}

func isUnstructured(fName string) bool {
	// List of klog functions we do not want to use after migration to structured logging.
	unstrucured := []string{
		"Infof", "Info", "Infoln", "InfoDepth",
		"Warning", "Warningf", "Warningln", "WarningDepth",
		"Error", "Errorf", "Errorln", "ErrorDepth",
		"Fatal", "Fatalf", "Fatalln", "FatalDepth",
		"Exit", "Exitf", "Exitln", "ExitDepth",
	}

	for _, name := range unstrucured {
		if fName == name {
			return true
		}
	}

	return false
}

func isContextualCall(fName string) bool {
	// List of klog functions we still want to use after migration to
	// contextual logging. This is an allow list, so any new acceptable
	// klog call has to be added here.
	contextual := []string{
		"Background",
		"ClearLogger",
		"ContextualLogger",
		"EnableContextualLogging",
		"FlushAndExit",
		"FlushLogger",
		"FromContext",
		"KObj",
		"KObjs",
		"KRef",
		"LoggerWithName",
		"LoggerWithValues",
		"NewContext",
		"SetLogger",
		"SetLoggerWithOptions",
		"StartFlushDaemon",
		"StopFlushDaemon",
		"TODO",
	}
	for _, name := range contextual {
		if fName == name {
			return true
		}
	}

	return false
}

// isKeysValid check if all keys in keyAndValues is string type
func isKeysValid(keyValues []ast.Expr, fun ast.Expr, pass *analysis.Pass, funName string) {
	if len(keyValues)%2 != 0 {
		pass.Report(analysis.Diagnostic{
			Pos:     fun.Pos(),
			Message: fmt.Sprintf("Additional arguments to %s should always be Key Value pairs. Please check if there is any key or value missing.", funName),
		})
		return
	}

	for index, arg := range keyValues {
		if index%2 != 0 {
			continue
		}
		lit, ok := arg.(*ast.BasicLit)
		if !ok {
			pass.Report(analysis.Diagnostic{
				Pos:     fun.Pos(),
				Message: fmt.Sprintf("Key positional arguments are expected to be inlined constant strings. Please replace %v provided with string value.", arg),
			})
			continue
		}
		if lit.Kind != token.STRING {
			pass.Report(analysis.Diagnostic{
				Pos:     fun.Pos(),
				Message: fmt.Sprintf("Key positional arguments are expected to be inlined constant strings. Please replace %v provided with string value.", lit.Value),
			})
			continue
		}
		isASCII := utf8string.NewString(lit.Value).IsASCII()
		if !isASCII {
			pass.Report(analysis.Diagnostic{
				Pos:     fun.Pos(),
				Message: fmt.Sprintf("Key positional arguments %s are expected to be lowerCamelCase alphanumeric strings. Please remove any non-Latin characters.", lit.Value),
			})
		}
	}
}

func checkForFormatSpecifier(expr *ast.CallExpr, pass *analysis.Pass) bool {
	if selExpr, ok := expr.Fun.(*ast.SelectorExpr); ok {
		// extracting function Name like Infof
		fName := selExpr.Sel.Name
		if strings.HasSuffix(fName, "f") {
			// Allowed for calls like Infof.
			return false
		}
		if specifier, found := hasFormatSpecifier(expr.Args); found {
			msg := fmt.Sprintf("logging function %q should not use format specifier %q", fName, specifier)
			pass.Report(analysis.Diagnostic{
				Pos:     expr.Fun.Pos(),
				Message: msg,
			})
			return true
		}
	}
	return false
}

func hasFormatSpecifier(fArgs []ast.Expr) (string, bool) {
	formatSpecifiers := []string{
		"%v", "%+v", "%#v", "%T",
		"%t", "%b", "%c", "%d", "%o", "%O", "%q", "%x", "%X", "%U",
		"%e", "%E", "%f", "%F", "%g", "%G", "%s", "%q", "%p",
	}
	for _, fArg := range fArgs {
		if arg, ok := fArg.(*ast.BasicLit); ok {
			for _, specifier := range formatSpecifiers {
				if strings.Contains(arg.Value, specifier) {
					return specifier, true
				}
			}
		}
	}
	return "", false
}

// checkForContextAndLogger ensures that a function doesn't accept both a
// context and a logger. That is problematic because it leads to ambiguity:
// does the context already contain the logger? That matters when passing it on
// without the logger.
func checkForContextAndLogger(n ast.Node, params *ast.FieldList, pass *analysis.Pass, c *config) {
	var haveLogger, haveContext bool

	for _, param := range params.List {
		if typeAndValue, ok := pass.TypesInfo.Types[param.Type]; ok {
			switch t := typeAndValue.Type.(type) {
			case *types.Named:
				if typeName := t.Obj(); typeName != nil {
					if pkg := typeName.Pkg(); pkg != nil {
						if typeName.Name() == "Logger" && pkg.Path() == "github.com/go-logr/logr" {
							haveLogger = true
						} else if typeName.Name() == "Context" && pkg.Path() == "context" {
							haveContext = true
						}
					}
				}
			}
		}
	}

	if haveLogger && haveContext {
		pass.Report(analysis.Diagnostic{
			Pos:     n.Pos(),
			End:     n.End(),
			Message: `A function should accept either a context or a logger, but not both. Having both makes calling the function harder because it must be defined whether the context must contain the logger and callers have to follow that.`,
		})
	}
}

// checkForIfEnabled detects `if klog.V(..).Enabled() { ...` and `if
// logger.V(...).Enabled()` and suggests capturing the result of V.
func checkForIfEnabled(i *ast.IfStmt, pass *analysis.Pass, c *config) {
	// if i.Init == nil {
	// A more complex if statement, let's assume it's okay.
	// return
	// }

	// Must be a method call.
	callExpr, ok := i.Cond.(*ast.CallExpr)
	if !ok {
		return
	}
	selExpr, ok := callExpr.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}

	// We only care about calls to Enabled().
	if selExpr.Sel.Name != "Enabled" {
		return
	}

	// And it must be Enabled for klog or logr.Logger.
	if !isKlogVerbose(selExpr.X, pass) &&
		!isGoLogger(selExpr.X, pass) {
		return
	}

	// logger.Enabled() is okay, logger.V(1).Enabled() is not.
	// That means we need to check for another selector expression
	// with V as method name.
	subCallExpr, ok := selExpr.X.(*ast.CallExpr)
	if !ok {
		return
	}
	subSelExpr, ok := subCallExpr.Fun.(*ast.SelectorExpr)
	if !ok || subSelExpr.Sel.Name != "V" {
		return
	}

	// klogV is recommended as replacement for klog.V(). For logr.Logger
	// let's use the root of the selector, which should be a variable.
	varName := "klogV"
	funcCall := "klog.V"
	if isGoLogger(subSelExpr.X, pass) {
		varName = "logger"
		root := subSelExpr
		for s, ok := root.X.(*ast.SelectorExpr); ok; s, ok = root.X.(*ast.SelectorExpr) {
			root = s
		}
		if id, ok := root.X.(*ast.Ident); ok {
			varName = id.Name
		}
		funcCall = varName + ".V"
	}

	pass.Report(analysis.Diagnostic{
		Pos: i.Pos(),
		End: i.End(),
		Message: fmt.Sprintf("the result of %s should be stored in a variable and then be used multiple times: if %s := %s(); %s.Enabled() { ... %s.Info ... }",
			funcCall, varName, funcCall, varName, varName),
	})
}
