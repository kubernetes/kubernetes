package lint

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"math"
	"regexp"
	"strings"
)

// File abstraction used for representing files.
type File struct {
	Name    string
	Pkg     *Package
	content []byte
	AST     *ast.File
}

// IsTest returns if the file contains tests.
func (f *File) IsTest() bool { return strings.HasSuffix(f.Name, "_test.go") }

// Content returns the file's content.
func (f *File) Content() []byte {
	return f.content
}

// NewFile creates a new file
func NewFile(name string, content []byte, pkg *Package) (*File, error) {
	f, err := parser.ParseFile(pkg.fset, name, content, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	return &File{
		Name:    name,
		content: content,
		Pkg:     pkg,
		AST:     f,
	}, nil
}

// ToPosition returns line and column for given position.
func (f *File) ToPosition(pos token.Pos) token.Position {
	return f.Pkg.fset.Position(pos)
}

// Render renders a node.
func (f *File) Render(x interface{}) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, f.Pkg.fset, x); err != nil {
		panic(err)
	}
	return buf.String()
}

// CommentMap builds a comment map for the file.
func (f *File) CommentMap() ast.CommentMap {
	return ast.NewCommentMap(f.Pkg.fset, f.AST, f.AST.Comments)
}

var basicTypeKinds = map[types.BasicKind]string{
	types.UntypedBool:    "bool",
	types.UntypedInt:     "int",
	types.UntypedRune:    "rune",
	types.UntypedFloat:   "float64",
	types.UntypedComplex: "complex128",
	types.UntypedString:  "string",
}

// IsUntypedConst reports whether expr is an untyped constant,
// and indicates what its default type is.
// scope may be nil.
func (f *File) IsUntypedConst(expr ast.Expr) (defType string, ok bool) {
	// Re-evaluate expr outside its context to see if it's untyped.
	// (An expr evaluated within, for example, an assignment context will get the type of the LHS.)
	exprStr := f.Render(expr)
	tv, err := types.Eval(f.Pkg.fset, f.Pkg.TypesPkg, expr.Pos(), exprStr)
	if err != nil {
		return "", false
	}
	if b, ok := tv.Type.(*types.Basic); ok {
		if dt, ok := basicTypeKinds[b.Kind()]; ok {
			return dt, true
		}
	}

	return "", false
}

func (f *File) isMain() bool {
	return f.AST.Name.Name == "main"
}

const directiveSpecifyDisableReason = "specify-disable-reason"

func (f *File) lint(rules []Rule, config Config, failures chan Failure) {
	rulesConfig := config.Rules
	_, mustSpecifyDisableReason := config.Directives[directiveSpecifyDisableReason]
	disabledIntervals := f.disabledIntervals(rules, mustSpecifyDisableReason, failures)
	for _, currentRule := range rules {
		ruleConfig := rulesConfig[currentRule.Name()]
		currentFailures := currentRule.Apply(f, ruleConfig.Arguments)
		for idx, failure := range currentFailures {
			if failure.RuleName == "" {
				failure.RuleName = currentRule.Name()
			}
			if failure.Node != nil {
				failure.Position = ToFailurePosition(failure.Node.Pos(), failure.Node.End(), f)
			}
			currentFailures[idx] = failure
		}
		currentFailures = f.filterFailures(currentFailures, disabledIntervals)
		for _, failure := range currentFailures {
			if failure.Confidence >= config.Confidence {
				failures <- failure
			}
		}
	}
}

type enableDisableConfig struct {
	enabled  bool
	position int
}

const directiveRE = `^//[\s]*revive:(enable|disable)(?:-(line|next-line))?(?::([^\s]+))?[\s]*(?: (.+))?$`
const directivePos = 1
const modifierPos = 2
const rulesPos = 3
const reasonPos = 4

var re = regexp.MustCompile(directiveRE)

func (f *File) disabledIntervals(rules []Rule, mustSpecifyDisableReason bool, failures chan Failure) disabledIntervalsMap {
	enabledDisabledRulesMap := make(map[string][]enableDisableConfig)

	getEnabledDisabledIntervals := func() disabledIntervalsMap {
		result := make(disabledIntervalsMap)

		for ruleName, disabledArr := range enabledDisabledRulesMap {
			ruleResult := []DisabledInterval{}
			for i := 0; i < len(disabledArr); i++ {
				interval := DisabledInterval{
					RuleName: ruleName,
					From: token.Position{
						Filename: f.Name,
						Line:     disabledArr[i].position,
					},
					To: token.Position{
						Filename: f.Name,
						Line:     math.MaxInt32,
					},
				}
				if i%2 == 0 {
					ruleResult = append(ruleResult, interval)
				} else {
					ruleResult[len(ruleResult)-1].To.Line = disabledArr[i].position
				}
			}
			result[ruleName] = ruleResult
		}

		return result
	}

	handleConfig := func(isEnabled bool, line int, name string) {
		existing, ok := enabledDisabledRulesMap[name]
		if !ok {
			existing = []enableDisableConfig{}
			enabledDisabledRulesMap[name] = existing
		}
		if (len(existing) > 1 && existing[len(existing)-1].enabled == isEnabled) ||
			(len(existing) == 0 && isEnabled) {
			return
		}
		existing = append(existing, enableDisableConfig{
			enabled:  isEnabled,
			position: line,
		})
		enabledDisabledRulesMap[name] = existing
	}

	handleRules := func(filename, modifier string, isEnabled bool, line int, ruleNames []string) []DisabledInterval {
		var result []DisabledInterval
		for _, name := range ruleNames {
			if modifier == "line" {
				handleConfig(isEnabled, line, name)
				handleConfig(!isEnabled, line, name)
			} else if modifier == "next-line" {
				handleConfig(isEnabled, line+1, name)
				handleConfig(!isEnabled, line+1, name)
			} else {
				handleConfig(isEnabled, line, name)
			}
		}
		return result
	}

	handleComment := func(filename string, c *ast.CommentGroup, line int) {
		comments := c.List
		for _, c := range comments {
			match := re.FindStringSubmatch(c.Text)
			if len(match) == 0 {
				continue
			}
			ruleNames := []string{}
			tempNames := strings.Split(match[rulesPos], ",")

			for _, name := range tempNames {
				name = strings.Trim(name, "\n")
				if len(name) > 0 {
					ruleNames = append(ruleNames, name)
				}
			}

			mustCheckDisablingReason := mustSpecifyDisableReason && match[directivePos] == "disable"
			if mustCheckDisablingReason && strings.Trim(match[reasonPos], " ") == "" {
				failures <- Failure{
					Confidence: 1,
					RuleName:   directiveSpecifyDisableReason,
					Failure:    "reason of lint disabling not found",
					Position:   ToFailurePosition(c.Pos(), c.End(), f),
					Node:       c,
				}
				continue // skip this linter disabling directive
			}

			// TODO: optimize
			if len(ruleNames) == 0 {
				for _, rule := range rules {
					ruleNames = append(ruleNames, rule.Name())
				}
			}

			handleRules(filename, match[modifierPos], match[directivePos] == "enable", line, ruleNames)
		}
	}

	comments := f.AST.Comments
	for _, c := range comments {
		handleComment(f.Name, c, f.ToPosition(c.End()).Line)
	}

	return getEnabledDisabledIntervals()
}

func (f *File) filterFailures(failures []Failure, disabledIntervals disabledIntervalsMap) []Failure {
	result := []Failure{}
	for _, failure := range failures {
		fStart := failure.Position.Start.Line
		fEnd := failure.Position.End.Line
		intervals, ok := disabledIntervals[failure.RuleName]
		if !ok {
			result = append(result, failure)
		} else {
			include := true
			for _, interval := range intervals {
				intStart := interval.From.Line
				intEnd := interval.To.Line
				if (fStart >= intStart && fStart <= intEnd) ||
					(fEnd >= intStart && fEnd <= intEnd) {
					include = false
					break
				}
			}
			if include {
				result = append(result, failure)
			}
		}
	}
	return result
}
