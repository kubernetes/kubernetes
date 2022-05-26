// (c) Copyright 2016 Hewlett Packard Enterprise Development LP
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rules

import (
	"go/ast"
	"regexp"
	"strings"

	"github.com/securego/gosec/v2"
)

type sqlStatement struct {
	gosec.MetaData
	gosec.CallList

	// Contains a list of patterns which must all match for the rule to match.
	patterns []*regexp.Regexp
}

func (s *sqlStatement) ID() string {
	return s.MetaData.ID
}

// See if the string matches the patterns for the statement.
func (s *sqlStatement) MatchPatterns(str string) bool {
	for _, pattern := range s.patterns {
		if !pattern.MatchString(str) {
			return false
		}
	}
	return true
}

type sqlStrConcat struct {
	sqlStatement
}

func (s *sqlStrConcat) ID() string {
	return s.MetaData.ID
}

// see if we can figure out what it is
func (s *sqlStrConcat) checkObject(n *ast.Ident, c *gosec.Context) bool {
	if n.Obj != nil {
		return n.Obj.Kind != ast.Var && n.Obj.Kind != ast.Fun
	}

	// Try to resolve unresolved identifiers using other files in same package
	for _, file := range c.PkgFiles {
		if node, ok := file.Scope.Objects[n.String()]; ok {
			return node.Kind != ast.Var && node.Kind != ast.Fun
		}
	}
	return false
}

// checkQuery verifies if the query parameters is a string concatenation
func (s *sqlStrConcat) checkQuery(call *ast.CallExpr, ctx *gosec.Context) (*gosec.Issue, error) {
	_, fnName, err := gosec.GetCallInfo(call, ctx)
	if err != nil {
		return nil, err
	}
	var query ast.Node
	if strings.HasSuffix(fnName, "Context") {
		query = call.Args[1]
	} else {
		query = call.Args[0]
	}

	if be, ok := query.(*ast.BinaryExpr); ok {
		operands := gosec.GetBinaryExprOperands(be)
		if start, ok := operands[0].(*ast.BasicLit); ok {
			if str, e := gosec.GetString(start); e == nil {
				if !s.MatchPatterns(str) {
					return nil, nil
				}
			}
			for _, op := range operands[1:] {
				if _, ok := op.(*ast.BasicLit); ok {
					continue
				}
				if op, ok := op.(*ast.Ident); ok && s.checkObject(op, ctx) {
					continue
				}
				return gosec.NewIssue(ctx, be, s.ID(), s.What, s.Severity, s.Confidence), nil
			}
		}
	}

	return nil, nil
}

// Checks SQL query concatenation issues such as "SELECT * FROM table WHERE " + " ' OR 1=1"
func (s *sqlStrConcat) Match(n ast.Node, ctx *gosec.Context) (*gosec.Issue, error) {
	switch stmt := n.(type) {
	case *ast.AssignStmt:
		for _, expr := range stmt.Rhs {
			if sqlQueryCall, ok := expr.(*ast.CallExpr); ok && s.ContainsCallExpr(expr, ctx) != nil {
				return s.checkQuery(sqlQueryCall, ctx)
			}
		}
	case *ast.ExprStmt:
		if sqlQueryCall, ok := stmt.X.(*ast.CallExpr); ok && s.ContainsCallExpr(stmt.X, ctx) != nil {
			return s.checkQuery(sqlQueryCall, ctx)
		}
	}
	return nil, nil
}

// NewSQLStrConcat looks for cases where we are building SQL strings via concatenation
func NewSQLStrConcat(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	rule := &sqlStrConcat{
		sqlStatement: sqlStatement{
			patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)(SELECT|DELETE|INSERT|UPDATE|INTO|FROM|WHERE) `),
			},
			MetaData: gosec.MetaData{
				ID:         id,
				Severity:   gosec.Medium,
				Confidence: gosec.High,
				What:       "SQL string concatenation",
			},
			CallList: gosec.NewCallList(),
		},
	}

	rule.AddAll("*database/sql.DB", "Query", "QueryContext", "QueryRow", "QueryRowContext", "Exec", "ExecContext", "Prepare", "PrepareContext")
	rule.AddAll("*database/sql.Tx", "Query", "QueryContext", "QueryRow", "QueryRowContext", "Exec", "ExecContext", "Prepare", "PrepareContext")
	return rule, []ast.Node{(*ast.AssignStmt)(nil), (*ast.ExprStmt)(nil)}
}

type sqlStrFormat struct {
	gosec.CallList
	sqlStatement
	fmtCalls      gosec.CallList
	noIssue       gosec.CallList
	noIssueQuoted gosec.CallList
}

// see if we can figure out what it is
func (s *sqlStrFormat) constObject(e ast.Expr, c *gosec.Context) bool {
	n, ok := e.(*ast.Ident)
	if !ok {
		return false
	}

	if n.Obj != nil {
		return n.Obj.Kind == ast.Con
	}

	// Try to resolve unresolved identifiers using other files in same package
	for _, file := range c.PkgFiles {
		if node, ok := file.Scope.Objects[n.String()]; ok {
			return node.Kind == ast.Con
		}
	}
	return false
}

func (s *sqlStrFormat) checkQuery(call *ast.CallExpr, ctx *gosec.Context) (*gosec.Issue, error) {
	_, fnName, err := gosec.GetCallInfo(call, ctx)
	if err != nil {
		return nil, err
	}
	var query ast.Node
	if strings.HasSuffix(fnName, "Context") {
		query = call.Args[1]
	} else {
		query = call.Args[0]
	}

	if ident, ok := query.(*ast.Ident); ok && ident.Obj != nil {
		decl := ident.Obj.Decl
		if assign, ok := decl.(*ast.AssignStmt); ok {
			for _, expr := range assign.Rhs {
				issue := s.checkFormatting(expr, ctx)
				if issue != nil {
					return issue, err
				}
			}
		}
	}

	return nil, nil
}

func (s *sqlStrFormat) checkFormatting(n ast.Node, ctx *gosec.Context) *gosec.Issue {
	// argIndex changes the function argument which gets matched to the regex
	argIndex := 0
	if node := s.fmtCalls.ContainsPkgCallExpr(n, ctx, false); node != nil {
		// if the function is fmt.Fprintf, search for SQL statement in Args[1] instead
		if sel, ok := node.Fun.(*ast.SelectorExpr); ok {
			if sel.Sel.Name == "Fprintf" {
				// if os.Stderr or os.Stdout is in Arg[0], mark as no issue
				if arg, ok := node.Args[0].(*ast.SelectorExpr); ok {
					if ident, ok := arg.X.(*ast.Ident); ok {
						if s.noIssue.Contains(ident.Name, arg.Sel.Name) {
							return nil
						}
					}
				}
				// the function is Fprintf so set argIndex = 1
				argIndex = 1
			}
		}

		// no formatter
		if len(node.Args) == 0 {
			return nil
		}

		var formatter string

		// concats callexpr arg strings together if needed before regex evaluation
		if argExpr, ok := node.Args[argIndex].(*ast.BinaryExpr); ok {
			if fullStr, ok := gosec.ConcatString(argExpr); ok {
				formatter = fullStr
			}
		} else if arg, e := gosec.GetString(node.Args[argIndex]); e == nil {
			formatter = arg
		}
		if len(formatter) <= 0 {
			return nil
		}

		// If all formatter args are quoted or constant, then the SQL construction is safe
		if argIndex+1 < len(node.Args) {
			allSafe := true
			for _, arg := range node.Args[argIndex+1:] {
				if n := s.noIssueQuoted.ContainsPkgCallExpr(arg, ctx, true); n == nil && !s.constObject(arg, ctx) {
					allSafe = false
					break
				}
			}
			if allSafe {
				return nil
			}
		}
		if s.MatchPatterns(formatter) {
			return gosec.NewIssue(ctx, n, s.ID(), s.What, s.Severity, s.Confidence)
		}
	}
	return nil
}

// Check SQL query formatting issues such as "fmt.Sprintf("SELECT * FROM foo where '%s', userInput)"
func (s *sqlStrFormat) Match(n ast.Node, ctx *gosec.Context) (*gosec.Issue, error) {
	switch stmt := n.(type) {
	case *ast.AssignStmt:
		for _, expr := range stmt.Rhs {
			if call, ok := expr.(*ast.CallExpr); ok {
				selector, ok := call.Fun.(*ast.SelectorExpr)
				if !ok {
					continue
				}
				sqlQueryCall, ok := selector.X.(*ast.CallExpr)
				if ok && s.ContainsCallExpr(sqlQueryCall, ctx) != nil {
					issue, err := s.checkQuery(sqlQueryCall, ctx)
					if err == nil && issue != nil {
						return issue, err
					}
				}
			}
			if sqlQueryCall, ok := expr.(*ast.CallExpr); ok && s.ContainsCallExpr(expr, ctx) != nil {
				return s.checkQuery(sqlQueryCall, ctx)
			}
		}
	case *ast.ExprStmt:
		if sqlQueryCall, ok := stmt.X.(*ast.CallExpr); ok && s.ContainsCallExpr(stmt.X, ctx) != nil {
			return s.checkQuery(sqlQueryCall, ctx)
		}
	}
	return nil, nil
}

// NewSQLStrFormat looks for cases where we're building SQL query strings using format strings
func NewSQLStrFormat(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	rule := &sqlStrFormat{
		CallList:      gosec.NewCallList(),
		fmtCalls:      gosec.NewCallList(),
		noIssue:       gosec.NewCallList(),
		noIssueQuoted: gosec.NewCallList(),
		sqlStatement: sqlStatement{
			patterns: []*regexp.Regexp{
				regexp.MustCompile("(?i)(SELECT|DELETE|INSERT|UPDATE|INTO|FROM|WHERE)( |\n|\r|\t)"),
				regexp.MustCompile("%[^bdoxXfFp]"),
			},
			MetaData: gosec.MetaData{
				ID:         id,
				Severity:   gosec.Medium,
				Confidence: gosec.High,
				What:       "SQL string formatting",
			},
		},
	}
	rule.AddAll("*database/sql.DB", "Query", "QueryContext", "QueryRow", "QueryRowContext", "Exec", "ExecContext", "Prepare", "PrepareContext")
	rule.AddAll("*database/sql.Tx", "Query", "QueryContext", "QueryRow", "QueryRowContext", "Exec", "ExecContext", "Prepare", "PrepareContext")
	rule.fmtCalls.AddAll("fmt", "Sprint", "Sprintf", "Sprintln", "Fprintf")
	rule.noIssue.AddAll("os", "Stdout", "Stderr")
	rule.noIssueQuoted.Add("github.com/lib/pq", "QuoteIdentifier")

	return rule, []ast.Node{(*ast.AssignStmt)(nil), (*ast.ExprStmt)(nil)}
}
