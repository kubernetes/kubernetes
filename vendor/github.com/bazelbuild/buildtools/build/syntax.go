/*
Copyright 2016 Google Inc. All Rights Reserved.

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

// Package build implements parsing and printing of BUILD files.
package build

// Syntax data structure definitions.

import (
	"strings"
	"unicode/utf8"
)

// A Position describes the position between two bytes of input.
type Position struct {
	Line     int // line in input (starting at 1)
	LineRune int // rune in line (starting at 1)
	Byte     int // byte in input (starting at 0)
}

// add returns the position at the end of s, assuming it starts at p.
func (p Position) add(s string) Position {
	p.Byte += len(s)
	if n := strings.Count(s, "\n"); n > 0 {
		p.Line += n
		s = s[strings.LastIndex(s, "\n")+1:]
		p.LineRune = 1
	}
	p.LineRune += utf8.RuneCountInString(s)
	return p
}

// An Expr represents an input element.
type Expr interface {
	// Span returns the start and end position of the expression,
	// excluding leading or trailing comments.
	Span() (start, end Position)

	// Comment returns the comments attached to the expression.
	// This method would normally be named 'Comments' but that
	// would interfere with embedding a type of the same name.
	Comment() *Comments
}

// A Comment represents a single # comment.
type Comment struct {
	Start Position
	Token string // without trailing newline
}

// Comments collects the comments associated with an expression.
type Comments struct {
	Before []Comment // whole-line comments before this expression
	Suffix []Comment // end-of-line comments after this expression

	// For top-level expressions only, After lists whole-line
	// comments following the expression.
	After []Comment
}

// Comment returns the receiver. This isn't useful by itself, but
// a Comments struct is embedded into all the expression
// implementation types, and this gives each of those a Comment
// method to satisfy the Expr interface.
func (c *Comments) Comment() *Comments {
	return c
}

// A File represents an entire BUILD file.
type File struct {
	Path string // file path, relative to workspace directory
	Comments
	Stmt []Expr
}

func (f *File) Span() (start, end Position) {
	if len(f.Stmt) == 0 {
		return
	}
	start, _ = f.Stmt[0].Span()
	_, end = f.Stmt[len(f.Stmt)-1].Span()
	return start, end
}

// A CommentBlock represents a top-level block of comments separate
// from any rule.
type CommentBlock struct {
	Comments
	Start Position
}

func (x *CommentBlock) Span() (start, end Position) {
	return x.Start, x.Start
}

// A PythonBlock represents a blob of Python code, typically a def or for loop.
type PythonBlock struct {
	Comments
	Start Position
	Token string // raw Python code, including final newline
}

func (x *PythonBlock) Span() (start, end Position) {
	return x.Start, x.Start.add(x.Token)
}

// A LiteralExpr represents a literal identifier or number.
type LiteralExpr struct {
	Comments
	Start Position
	Token string // identifier token
}

func (x *LiteralExpr) Span() (start, end Position) {
	return x.Start, x.Start.add(x.Token)
}

// A StringExpr represents a single literal string.
type StringExpr struct {
	Comments
	Start       Position
	Value       string // string value (decoded)
	TripleQuote bool   // triple quote output
	End         Position

	// To allow specific formatting of string literals,
	// at least within our requirements, record the
	// preferred form of Value. This field is a hint:
	// it is only used if it is a valid quoted form for Value.
	Token string
}

func (x *StringExpr) Span() (start, end Position) {
	return x.Start, x.End
}

// An End represents the end of a parenthesized or bracketed expression.
// It is a place to hang comments.
type End struct {
	Comments
	Pos Position
}

func (x *End) Span() (start, end Position) {
	return x.Pos, x.Pos.add(")")
}

// A CallExpr represents a function call expression: X(List).
type CallExpr struct {
	Comments
	X              Expr
	ListStart      Position // position of (
	List           []Expr
	End                 // position of )
	ForceCompact   bool // force compact (non-multiline) form when printing
	ForceMultiLine bool // force multiline form when printing
}

func (x *CallExpr) Span() (start, end Position) {
	start, _ = x.X.Span()
	return start, x.End.Pos.add(")")
}

// A DotExpr represents a field selector: X.Name.
type DotExpr struct {
	Comments
	X       Expr
	Dot     Position
	NamePos Position
	Name    string
}

func (x *DotExpr) Span() (start, end Position) {
	start, _ = x.X.Span()
	return start, x.NamePos.add(x.Name)
}

// A ListForExpr represents a list comprehension expression: [X for ... if ...].
type ListForExpr struct {
	Comments
	ForceMultiLine bool   // split expression across multiple lines
	Brack          string // "", "()", or "[]"
	Start          Position
	X              Expr
	For            []*ForClauseWithIfClausesOpt
	End
}

func (x *ListForExpr) Span() (start, end Position) {
	return x.Start, x.End.Pos.add("]")
}

// A ForClause represents a for clause in a list comprehension: for Var in Expr.
type ForClause struct {
	Comments
	For  Position
	Var  []Expr
	In   Position
	Expr Expr
}

func (x *ForClause) Span() (start, end Position) {
	_, end = x.Expr.Span()
	return x.For, end
}

// An IfClause represents an if clause in a list comprehension: if Cond.
type IfClause struct {
	Comments
	If   Position
	Cond Expr
}

func (x *IfClause) Span() (start, end Position) {
	_, end = x.Cond.Span()
	return x.If, end
}

// A ForClauseWithIfClausesOpt represents a for clause in a list comprehension followed by optional
// if expressions: for ... in ... [if ... if ...]
type ForClauseWithIfClausesOpt struct {
	Comments
	For *ForClause
	Ifs []*IfClause
}

func (x *ForClauseWithIfClausesOpt) Span() (start, end Position) {
	start, end = x.For.Span()
	if len(x.Ifs) > 0 {
		_, end = x.Ifs[len(x.Ifs)-1].Span()
	}

	return start, end
}

// A KeyValueExpr represents a dictionary entry: Key: Value.
type KeyValueExpr struct {
	Comments
	Key   Expr
	Colon Position
	Value Expr
}

func (x *KeyValueExpr) Span() (start, end Position) {
	start, _ = x.Key.Span()
	_, end = x.Value.Span()
	return start, end
}

// A DictExpr represents a dictionary literal: { List }.
type DictExpr struct {
	Comments
	Start Position
	List  []Expr   // all *KeyValueExprs
	Comma Position // position of trailing comma, if any
	End
	ForceMultiLine bool // force multiline form when printing
}

func (x *DictExpr) Span() (start, end Position) {
	return x.Start, x.End.Pos.add("}")
}

// A ListExpr represents a list literal: [ List ].
type ListExpr struct {
	Comments
	Start Position
	List  []Expr
	Comma Position // position of trailing comma, if any
	End
	ForceMultiLine bool // force multiline form when printing
}

func (x *ListExpr) Span() (start, end Position) {
	return x.Start, x.End.Pos.add("]")
}

// A SetExpr represents a set literal: { List }.
type SetExpr struct {
	Comments
	Start Position
	List  []Expr
	Comma Position // position of trailing comma, if any
	End
	ForceMultiLine bool // force multiline form when printing
}

func (x *SetExpr) Span() (start, end Position) {
	return x.Start, x.End.Pos.add("}")
}

// A TupleExpr represents a tuple literal: (List)
type TupleExpr struct {
	Comments
	Start Position
	List  []Expr
	Comma Position // position of trailing comma, if any
	End
	ForceCompact   bool // force compact (non-multiline) form when printing
	ForceMultiLine bool // force multiline form when printing
}

func (x *TupleExpr) Span() (start, end Position) {
	return x.Start, x.End.Pos.add(")")
}

// A UnaryExpr represents a unary expression: Op X.
type UnaryExpr struct {
	Comments
	OpStart Position
	Op      string
	X       Expr
}

func (x *UnaryExpr) Span() (start, end Position) {
	_, end = x.X.Span()
	return x.OpStart, end
}

// A BinaryExpr represents a binary expression: X Op Y.
type BinaryExpr struct {
	Comments
	X         Expr
	OpStart   Position
	Op        string
	LineBreak bool // insert line break between Op and Y
	Y         Expr
}

func (x *BinaryExpr) Span() (start, end Position) {
	start, _ = x.X.Span()
	_, end = x.Y.Span()
	return start, end
}

// A ParenExpr represents a parenthesized expression: (X).
type ParenExpr struct {
	Comments
	Start Position
	X     Expr
	End
	ForceMultiLine bool // insert line break after opening ( and before closing )
}

func (x *ParenExpr) Span() (start, end Position) {
	return x.Start, x.End.Pos.add(")")
}

// A SliceExpr represents a slice expression: expr[from:to] or expr[from:to:step] .
type SliceExpr struct {
	Comments
	X           Expr
	SliceStart  Position
	From        Expr
	FirstColon  Position
	To          Expr
	SecondColon Position
	Step        Expr
	End         Position
}

func (x *SliceExpr) Span() (start, end Position) {
	start, _ = x.X.Span()
	return start, x.End
}

// An IndexExpr represents an index expression: X[Y].
type IndexExpr struct {
	Comments
	X          Expr
	IndexStart Position
	Y          Expr
	End        Position
}

func (x *IndexExpr) Span() (start, end Position) {
	start, _ = x.X.Span()
	return start, x.End
}

// A LambdaExpr represents a lambda expression: lambda Var: Expr.
type LambdaExpr struct {
	Comments
	Lambda Position
	Var    []Expr
	Colon  Position
	Expr   Expr
}

func (x *LambdaExpr) Span() (start, end Position) {
	_, end = x.Expr.Span()
	return x.Lambda, end
}

// ConditionalExpr represents the conditional: X if TEST else ELSE.
type ConditionalExpr struct {
	Comments
	Then      Expr
	IfStart   Position
	Test      Expr
	ElseStart Position
	Else      Expr
}

// Span returns the start and end position of the expression,
// excluding leading or trailing comments.
func (x *ConditionalExpr) Span() (start, end Position) {
	start, _ = x.Then.Span()
	_, end = x.Else.Span()
	return start, end
}

// A CodeBlock represents an indented code block.
type CodeBlock struct {
	Statements []Expr
	Start      Position
	End
}

func (x *CodeBlock) Span() (start, end Position) {
	return x.Start, x.End.Pos
}

// A FuncDef represents a function definition expression: def foo(List):.
type FuncDef struct {
	Comments
	Start          Position // position of def
	Name           string
	ListStart      Position // position of (
	Args           []Expr
	Body           CodeBlock
	End                 // position of the end
	ForceCompact   bool // force compact (non-multiline) form when printing
	ForceMultiLine bool // force multiline form when printing
}

func (x *FuncDef) Span() (start, end Position) {
	return x.Start, x.End.Pos
}

// A ReturnExpr represents a return statement: return f(x).
type ReturnExpr struct {
	Comments
	Start Position
	X     Expr
	End   Position
}

func (x *ReturnExpr) Span() (start, end Position) {
	return x.Start, x.End
}

// A ForLoop represents a for loop block: for x in range(10):.
type ForLoop struct {
	Comments
	Start    Position // position of for
	LoopVars []Expr
	Iterable Expr
	Body     CodeBlock
	End      // position of the end
}

func (x *ForLoop) Span() (start, end Position) {
	return x.Start, x.End.Pos
}

// An IfElse represents an if-else blocks sequence: if x: ... elif y: ... else: ... .
type IfElse struct {
	Comments
	Start      Position // position of if
	Conditions []Condition
	End        // position of the end
}

type Condition struct {
	If   Expr
	Then CodeBlock
}

func (x *IfElse) Span() (start, end Position) {
	return x.Start, x.End.Pos
}
