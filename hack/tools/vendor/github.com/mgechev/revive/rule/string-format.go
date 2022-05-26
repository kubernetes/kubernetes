package rule

import (
	"fmt"
	"go/ast"
	"go/token"
	"regexp"
	"strconv"

	"github.com/mgechev/revive/lint"
)

// #region Revive API

// StringFormatRule lints strings and/or comments according to a set of regular expressions given as Arguments
type StringFormatRule struct{}

// Apply applies the rule to the given file.
func (r *StringFormatRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintStringFormatRule{onFailure: onFailure}
	w.parseArguments(arguments)
	ast.Walk(w, file.AST)

	return failures
}

// Name returns the rule name.
func (r *StringFormatRule) Name() string {
	return "string-format"
}

// ParseArgumentsTest is a public wrapper around w.parseArguments used for testing. Returns the error message provided to panic, or nil if no error was encountered
func (r *StringFormatRule) ParseArgumentsTest(arguments lint.Arguments) *string {
	w := lintStringFormatRule{}
	c := make(chan interface{})
	// Parse the arguments in a goroutine, defer a recover() call, return the error encountered (or nil if there was no error)
	go func() {
		defer func() {
			err := recover()
			c <- err
		}()
		w.parseArguments(arguments)
	}()
	err := <-c
	if err != nil {
		e := fmt.Sprintf("%s", err)
		return &e
	}
	return nil
}

// #endregion

// #region Internal structure

type lintStringFormatRule struct {
	onFailure func(lint.Failure)

	rules              []stringFormatSubrule
	stringDeclarations map[string]string
}

type stringFormatSubrule struct {
	parent       *lintStringFormatRule
	scope        stringFormatSubruleScope
	regexp       *regexp.Regexp
	errorMessage string
}

type stringFormatSubruleScope struct {
	funcName string // Function name the rule is scoped to
	argument int    // (optional) Which argument in calls to the function is checked against the rule (the first argument is checked by default)
	field    string // (optional) If the argument to be checked is a struct, which member of the struct is checked against the rule (top level members only)
}

// Regex inserted to match valid function/struct field identifiers
const identRegex = "[_A-Za-z][_A-Za-z0-9]*"

var parseStringFormatScope = regexp.MustCompile(
	fmt.Sprintf("^(%s(?:\\.%s)?)(?:\\[([0-9]+)\\](?:\\.(%s))?)?$", identRegex, identRegex, identRegex))

// #endregion

// #region Argument parsing

func (w *lintStringFormatRule) parseArguments(arguments lint.Arguments) {
	for i, argument := range arguments {
		scope, regex, errorMessage := w.parseArgument(argument, i)
		w.rules = append(w.rules, stringFormatSubrule{
			parent:       w,
			scope:        scope,
			regexp:       regex,
			errorMessage: errorMessage,
		})
	}
}

func (w lintStringFormatRule) parseArgument(argument interface{}, ruleNum int) (scope stringFormatSubruleScope, regex *regexp.Regexp, errorMessage string) {
	g, ok := argument.([]interface{}) // Cast to generic slice first
	if !ok {
		w.configError("argument is not a slice", ruleNum, 0)
	}
	if len(g) < 2 {
		w.configError("less than two slices found in argument, scope and regex are required", ruleNum, len(g)-1)
	}
	rule := make([]string, len(g))
	for i, obj := range g {
		val, ok := obj.(string)
		if !ok {
			w.configError("unexpected value, string was expected", ruleNum, i)
		}
		rule[i] = val
	}

	// Validate scope and regex length
	if rule[0] == "" {
		w.configError("empty scope provided", ruleNum, 0)
	} else if len(rule[1]) < 2 {
		w.configError("regex is too small (regexes should begin and end with '/')", ruleNum, 1)
	}

	// Parse rule scope
	scope = stringFormatSubruleScope{}
	matches := parseStringFormatScope.FindStringSubmatch(rule[0])
	if matches == nil {
		// The rule's scope didn't match the parsing regex at all, probably a configuration error
		w.parseError("unable to parse rule scope", ruleNum, 0)
	} else if len(matches) != 4 {
		// The rule's scope matched the parsing regex, but an unexpected number of submatches was returned, probably a bug
		w.parseError(fmt.Sprintf("unexpected number of submatches when parsing scope: %d, expected 4", len(matches)), ruleNum, 0)
	}
	scope.funcName = matches[1]
	if len(matches[2]) > 0 {
		var err error
		scope.argument, err = strconv.Atoi(matches[2])
		if err != nil {
			w.parseError("unable to parse argument number in rule scope", ruleNum, 0)
		}
	}
	if len(matches[3]) > 0 {
		scope.field = matches[3]
	}

	// Strip / characters from the beginning and end of rule[1] before compiling
	regex, err := regexp.Compile(rule[1][1 : len(rule[1])-1])
	if err != nil {
		w.parseError(fmt.Sprintf("unable to compile %s as regexp", rule[1]), ruleNum, 1)
	}

	// Use custom error message if provided
	if len(rule) == 3 {
		errorMessage = rule[2]
	}
	return scope, regex, errorMessage
}

// Report an invalid config, this is specifically the user's fault
func (w lintStringFormatRule) configError(msg string, ruleNum, option int) {
	panic(fmt.Sprintf("invalid configuration for string-format: %s [argument %d, option %d]", msg, ruleNum, option))
}

// Report a general config parsing failure, this may be the user's fault, but it isn't known for certain
func (w lintStringFormatRule) parseError(msg string, ruleNum, option int) {
	panic(fmt.Sprintf("failed to parse configuration for string-format: %s [argument %d, option %d]", msg, ruleNum, option))
}

// #endregion

// #region Node traversal

func (w lintStringFormatRule) Visit(node ast.Node) ast.Visitor {
	// First, check if node is a call expression
	call, ok := node.(*ast.CallExpr)
	if !ok {
		return w
	}

	// Get the name of the call expression to check against rule scope
	callName, ok := w.getCallName(call)
	if !ok {
		return w
	}

	for _, rule := range w.rules {
		if rule.scope.funcName == callName {
			rule.Apply(call)
		}
	}

	return w
}

// Return the name of a call expression in the form of package.Func or Func
func (w lintStringFormatRule) getCallName(call *ast.CallExpr) (callName string, ok bool) {
	if ident, ok := call.Fun.(*ast.Ident); ok {
		// Local function call
		return ident.Name, true
	}

	if selector, ok := call.Fun.(*ast.SelectorExpr); ok {
		// Scoped function call
		scope, ok := selector.X.(*ast.Ident)
		if !ok {
			return "", false
		}
		return scope.Name + "." + selector.Sel.Name, true
	}

	return "", false
}

// #endregion

// #region Linting logic

// Apply a single format rule to a call expression (should be done after verifying the that the call expression matches the rule's scope)
func (rule stringFormatSubrule) Apply(call *ast.CallExpr) {
	if len(call.Args) <= rule.scope.argument {
		return
	}

	arg := call.Args[rule.scope.argument]
	var lit *ast.BasicLit
	if len(rule.scope.field) > 0 {
		// Try finding the scope's Field, treating arg as a composite literal
		composite, ok := arg.(*ast.CompositeLit)
		if !ok {
			return
		}
		for _, el := range composite.Elts {
			kv, ok := el.(*ast.KeyValueExpr)
			if !ok {
				continue
			}
			key, ok := kv.Key.(*ast.Ident)
			if !ok || key.Name != rule.scope.field {
				continue
			}

			// We're now dealing with the exact field in the rule's scope, so if anything fails, we can safely return instead of continuing the loop
			lit, ok = kv.Value.(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				return
			}
		}
	} else {
		var ok bool
		// Treat arg as a string literal
		lit, ok = arg.(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return
		}
	}
	// Unquote the string literal before linting
	unquoted := lit.Value[1 : len(lit.Value)-1]
	rule.lintMessage(unquoted, lit)
}

func (rule stringFormatSubrule) lintMessage(s string, node ast.Node) {
	// Fail if the string doesn't match the user's regex
	if rule.regexp.MatchString(s) {
		return
	}
	var failure string
	if len(rule.errorMessage) > 0 {
		failure = rule.errorMessage
	} else {
		failure = fmt.Sprintf("string literal doesn't match user defined regex /%s/", rule.regexp.String())
	}
	rule.parent.onFailure(lint.Failure{
		Confidence: 1,
		Failure:    failure,
		Node:       node})
}

// #endregion
