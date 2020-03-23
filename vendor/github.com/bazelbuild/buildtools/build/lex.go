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
// Lexical scanning for BUILD file parser.

package build

import (
	"bytes"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// FileType represents a type of a file (default (for .bzl files), BUILD, or WORKSPACE).
// Certain formatting or refactoring rules can be applied to several file types, so they support
// bitwise operations: `type1 | type2` can represent a scope (e.g. BUILD and WORKSPACE files) and
// `scope & fileType` can be used to check whether a file type belongs to a scope.
type FileType int

const (
	// TypeDefault represents general Starlark files
	TypeDefault FileType = 1 << iota
	// TypeBuild represents BUILD files
	TypeBuild
	// TypeWorkspace represents WORKSPACE files
	TypeWorkspace
	// TypeBzl represents .bzl files
	TypeBzl
)

func (t FileType) String() string {
	switch t {
	case TypeDefault:
		return "default"
	case TypeBuild:
		return "BUILD"
	case TypeWorkspace:
		return "WORKSPACE"
	case TypeBzl:
		return ".bzl"
	}
	return "unknown"
}

// ParseBuild parses a file, marks it as a BUILD file and returns the corresponding parse tree.
//
// The filename is used only for generating error messages.
func ParseBuild(filename string, data []byte) (*File, error) {
	in := newInput(filename, data)
	f, err := in.parse()
	if f != nil {
		f.Type = TypeBuild
	}
	return f, err
}

// ParseWorkspace parses a file, marks it as a WORKSPACE file and returns the corresponding parse tree.
//
// The filename is used only for generating error messages.
func ParseWorkspace(filename string, data []byte) (*File, error) {
	in := newInput(filename, data)
	f, err := in.parse()
	if f != nil {
		f.Type = TypeWorkspace
	}
	return f, err
}

// ParseBzl parses a file, marks it as a .bzl file and returns the corresponding parse tree.
//
// The filename is used only for generating error messages.
func ParseBzl(filename string, data []byte) (*File, error) {
	in := newInput(filename, data)
	f, err := in.parse()
	if f != nil {
		f.Type = TypeBzl
	}
	return f, err
}

// ParseDefault parses a file, marks it as a generic Starlark file and returns the corresponding parse tree.
//
// The filename is used only for generating error messages.
func ParseDefault(filename string, data []byte) (*File, error) {
	in := newInput(filename, data)
	f, err := in.parse()
	if f != nil {
		f.Type = TypeDefault
	}
	return f, err
}

func getFileType(filename string) FileType {
	if filename == "" { // stdin
		return TypeDefault
	}
	basename := strings.ToLower(filepath.Base(filename))
	if strings.HasSuffix(basename, ".oss") {
		basename = basename[:len(basename)-4]
	}
	ext := filepath.Ext(basename)
	switch ext {
	case ".bzl":
		return TypeBzl
	case ".sky":
		return TypeDefault
	}
	base := basename[:len(basename)-len(ext)]
	switch {
	case ext == ".build" || base == "build" || strings.HasPrefix(base, "build."):
		return TypeBuild
	case ext == ".workspace" || base == "workspace" || strings.HasPrefix(base, "workspace."):
		return TypeWorkspace
	}
	return TypeDefault
}

// Parse parses the input data and returns the corresponding parse tree.
//
// Uses the filename to detect the formatting type (build, workspace, or default) and calls
// ParseBuild, ParseWorkspace, or ParseDefault correspondingly.
func Parse(filename string, data []byte) (*File, error) {
	switch getFileType(filename) {
	case TypeBuild:
		return ParseBuild(filename, data)
	case TypeWorkspace:
		return ParseWorkspace(filename, data)
	case TypeBzl:
		return ParseBzl(filename, data)
	}
	return ParseDefault(filename, data)
}

// ParseError contains information about the error encountered during parsing.
type ParseError struct {
	Message  string
	Filename string
	Pos      Position
}

// Error returns a string representation of the parse error.
func (e ParseError) Error() string {
	filename := e.Filename
	if filename == "" {
		filename = "<stdin>"
	}
	return fmt.Sprintf("%s:%d:%d: %v", filename, e.Pos.Line, e.Pos.LineRune, e.Message)
}

// An input represents a single input file being parsed.
type input struct {
	// Lexing state.
	filename       string    // name of input file, for errors
	complete       []byte    // entire input
	remaining      []byte    // remaining input
	token          []byte    // token being scanned
	lastToken      string    // most recently returned token, for error messages
	pos            Position  // current input position
	lineComments   []Comment // accumulated line comments
	suffixComments []Comment // accumulated suffix comments
	depth          int       // nesting of [ ] { } ( )
	cleanLine      bool      // true if the current line only contains whitespace before the current position
	indent         int       // current line indentation in spaces
	indents        []int     // stack of indentation levels in spaces

	// Parser state.
	file       *File // returned top-level syntax tree
	parseError error // error encountered during parsing

	// Comment assignment state.
	pre  []Expr // all expressions, in preorder traversal
	post []Expr // all expressions, in postorder traversal
}

func newInput(filename string, data []byte) *input {
	// The syntax requires that each simple statement ends with '\n', however it's optional at EOF.
	// If `data` doesn't end with '\n' we add it here to keep parser simple.
	// It shouldn't affect neither the parsed tree nor its formatting.
	data = append(data, '\n')

	return &input{
		filename:  filename,
		complete:  data,
		remaining: data,
		pos:       Position{Line: 1, LineRune: 1, Byte: 0},
		cleanLine: true,
		indents:   []int{0},
	}
}

func (in *input) currentIndent() int {
	return in.indents[len(in.indents)-1]
}

// parse parses the input file.
func (in *input) parse() (f *File, err error) {
	// The parser panics for both routine errors like syntax errors
	// and for programmer bugs like array index errors.
	// Turn both into error returns. Catching bug panics is
	// especially important when processing many files.
	defer func() {
		if e := recover(); e != nil {
			if e == in.parseError {
				err = in.parseError
			} else {
				err = ParseError{Message: fmt.Sprintf("internal error: %v", e), Filename: in.filename, Pos: in.pos}
			}
		}
	}()

	// Invoke the parser generated from parse.y.
	yyParse(in)
	if in.parseError != nil {
		return nil, in.parseError
	}
	in.file.Path = in.filename

	// Assign comments to nearby syntax.
	in.assignComments()

	return in.file, nil
}

// Error is called to report an error.
// When called by the generated code s is always "syntax error".
// Error does not return: it panics.
func (in *input) Error(s string) {
	if s == "syntax error" && in.lastToken != "" {
		s += " near " + in.lastToken
	}
	in.parseError = ParseError{Message: s, Filename: in.filename, Pos: in.pos}
	panic(in.parseError)
}

// eof reports whether the input has reached end of file.
func (in *input) eof() bool {
	return len(in.remaining) == 0
}

// peekRune returns the next rune in the input without consuming it.
func (in *input) peekRune() int {
	if len(in.remaining) == 0 {
		return 0
	}
	r, _ := utf8.DecodeRune(in.remaining)
	return int(r)
}

// readRune consumes and returns the next rune in the input.
func (in *input) readRune() int {
	if len(in.remaining) == 0 {
		in.Error("internal lexer error: readRune at EOF")
	}
	r, size := utf8.DecodeRune(in.remaining)
	in.remaining = in.remaining[size:]
	if r == '\n' {
		in.pos.Line++
		in.pos.LineRune = 1
	} else {
		in.pos.LineRune++
	}
	in.pos.Byte += size
	return int(r)
}

// startToken marks the beginning of the next input token.
// It must be followed by a call to endToken, once the token has
// been consumed using readRune.
func (in *input) startToken(val *yySymType) {
	in.token = in.remaining
	val.tok = ""
	val.pos = in.pos
}

// yySymType (used in the next few functions) is defined by the
// generated parser. It is a struct containing all the fields listed
// in parse.y's %union [sic] section.

// endToken marks the end of an input token.
// It records the actual token string in val.tok if the caller
// has not done that already.
func (in *input) endToken(val *yySymType) {
	if val.tok == "" {
		tok := string(in.token[:len(in.token)-len(in.remaining)])
		val.tok = tok
		in.lastToken = val.tok
	}
}

// Lex is called from the generated parser to obtain the next input token.
// It returns the token value (either a rune like '+' or a symbolic token _FOR)
// and sets val to the data associated with the token.
//
// For all our input tokens, the associated data is
// val.Pos (the position where the token begins)
// and val.Token (the input string corresponding to the token).
func (in *input) Lex(val *yySymType) int {
	// Skip past spaces, stopping at non-space or EOF.
	countNL := 0 // number of newlines we've skipped past
	for !in.eof() {
		// Skip over spaces. Count newlines so we can give the parser
		// information about where top-level blank lines are,
		// for top-level comment assignment.
		c := in.peekRune()
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			if c == '\n' {
				in.indent = 0
				in.cleanLine = true
				if in.depth == 0 {
					// Not in a statememt. Tell parser about top-level blank line.
					in.startToken(val)
					in.readRune()
					in.endToken(val)
					return '\n'
				}
				countNL++
			} else if c == ' ' && in.cleanLine {
				in.indent++
			}
			in.readRune()
			continue
		}

		// Comment runs to end of line.
		if c == '#' {
			// If a line contains just a comment its indentation level doesn't matter.
			// Reset it to zero.
			in.indent = 0
			isLineComment := in.cleanLine
			in.cleanLine = true

			// Is this comment the only thing on its line?
			// Find the last \n before this # and see if it's all
			// spaces from there to here.
			// If it's a suffix comment but the last non-space symbol before
			// it is one of (, [, or {, or it's a suffix comment to "):"
			// (e.g. trailing closing bracket or a function definition),
			// treat it as a line comment that should be
			// put inside the corresponding block.
			i := bytes.LastIndex(in.complete[:in.pos.Byte], []byte("\n"))
			prefix := bytes.TrimSpace(in.complete[i+1 : in.pos.Byte])
			prefix = bytes.Replace(prefix, []byte{' '}, []byte{}, -1)
			isSuffix := true
			if len(prefix) == 0 ||
				(len(prefix) == 2 && prefix[0] == ')' && prefix[1] == ':') ||
				prefix[len(prefix)-1] == '[' ||
				prefix[len(prefix)-1] == '(' ||
				prefix[len(prefix)-1] == '{' {
				isSuffix = false
			}

			// Consume comment without the \n it ends with.
			in.startToken(val)
			for len(in.remaining) > 0 && in.peekRune() != '\n' {
				in.readRune()
			}

			in.endToken(val)

			val.tok = strings.TrimRight(val.tok, "\n")
			in.lastToken = "comment"

			// If we are at top level (not in a rule), hand the comment to
			// the parser as a _COMMENT token. The grammar is written
			// to handle top-level comments itself.
			if in.depth == 0 && isLineComment {
				// Not in a statement. Tell parser about top-level comment.
				return _COMMENT
			}

			// Otherwise, save comment for later attachment to syntax tree.
			if countNL > 1 {
				in.lineComments = append(in.lineComments, Comment{val.pos, ""})
			}
			if isSuffix {
				in.suffixComments = append(in.suffixComments, Comment{val.pos, val.tok})
			} else {
				in.lineComments = append(in.lineComments, Comment{val.pos, val.tok})
			}
			countNL = 0
			continue
		}

		if c == '\\' && len(in.remaining) >= 2 && in.remaining[1] == '\n' {
			// We can ignore a trailing \ at end of line together with the \n.
			in.readRune()
			in.readRune()
			continue
		}

		// Found non-space non-comment.
		break
	}

	// Check for changes in indentation
	// Skip if we're inside a statement, or if there were non-space
	// characters before in the current line.
	if in.depth == 0 && in.cleanLine {
		if in.indent > in.currentIndent() {
			// A new indentation block starts
			in.indents = append(in.indents, in.indent)
			in.lastToken = "indent"
			in.cleanLine = false
			return _INDENT
		} else if in.indent < in.currentIndent() {
			// An indentation block ends
			in.indents = in.indents[:len(in.indents)-1]

			// It's a syntax error if the current line indentation level in now greater than
			// currentIndent(), should be either equal (a parent block continues) or still less
			// (need to unindent more).
			if in.indent > in.currentIndent() {
				in.pos = val.pos
				in.Error("unexpected indentation")
			}
			in.lastToken = "unindent"
			return _UNINDENT
		}
	}

	in.cleanLine = false

	// If the file ends with an indented block, return the corresponding amounts of unindents.
	if in.eof() && in.currentIndent() > 0 {
		in.indents = in.indents[:len(in.indents)-1]
		in.lastToken = "unindent"
		return _UNINDENT
	}

	// Found the beginning of the next token.
	in.startToken(val)
	defer in.endToken(val)

	// End of file.
	if in.eof() {
		in.lastToken = "EOF"
		return _EOF
	}

	// Punctuation tokens.
	switch c := in.peekRune(); c {
	case '[', '(', '{':
		in.depth++
		in.readRune()
		return c

	case ']', ')', '}':
		in.depth--
		in.readRune()
		return c

	case '.', ':', ';', ',': // single-char tokens
		in.readRune()
		return c

	case '<', '>', '=', '!', '+', '-', '*', '/', '%', '|', '&', '~', '^': // possibly followed by =
		in.readRune()

		if c == '~' {
			// unary bitwise not, shouldn't be followed by anything
			return c
		}

		if c == '*' && in.peekRune() == '*' {
			// double asterisk
			in.readRune()
			return _STAR_STAR
		}

		if c == in.peekRune() {
			switch c {
			case '/':
				// integer division
				in.readRune()
				c = _INT_DIV
			case '<':
				// left shift
				in.readRune()
				c = _BIT_LSH
			case '>':
				// right shift
				in.readRune()
				c = _BIT_RSH
			}
		}

		if in.peekRune() == '=' {
			in.readRune()
			switch c {
			case '<':
				return _LE
			case '>':
				return _GE
			case '=':
				return _EQ
			case '!':
				return _NE
			default:
				return _AUGM
			}
		}
		return c

	case 'r': // possible beginning of raw quoted string
		if len(in.remaining) < 2 || in.remaining[1] != '"' && in.remaining[1] != '\'' {
			break
		}
		in.readRune()
		c = in.peekRune()
		fallthrough

	case '"', '\'': // quoted string
		quote := c
		if len(in.remaining) >= 3 && in.remaining[0] == byte(quote) && in.remaining[1] == byte(quote) && in.remaining[2] == byte(quote) {
			// Triple-quoted string.
			in.readRune()
			in.readRune()
			in.readRune()
			var c1, c2, c3 int
			for {
				if in.eof() {
					in.pos = val.pos
					in.Error("unexpected EOF in string")
				}
				c1, c2, c3 = c2, c3, in.readRune()
				if c1 == quote && c2 == quote && c3 == quote {
					break
				}
				if c3 == '\\' {
					if in.eof() {
						in.pos = val.pos
						in.Error("unexpected EOF in string")
					}
					in.readRune()
				}
			}
		} else {
			in.readRune()
			for {
				if in.eof() {
					in.pos = val.pos
					in.Error("unexpected EOF in string")
				}
				if in.peekRune() == '\n' {
					in.Error("unexpected newline in string")
				}
				c := in.readRune()
				if c == quote {
					break
				}
				if c == '\\' {
					if in.eof() {
						in.pos = val.pos
						in.Error("unexpected EOF in string")
					}
					in.readRune()
				}
			}
		}
		in.endToken(val)
		s, triple, err := Unquote(val.tok)
		if err != nil {
			in.Error(fmt.Sprint(err))
		}
		val.str = s
		val.triple = triple
		return _STRING
	}

	// Checked all punctuation. Must be identifier token.
	if c := in.peekRune(); !isIdent(c) {
		in.Error(fmt.Sprintf("unexpected input character %#q", c))
	}

	// Scan over alphanumeric identifier.
	for {
		c := in.peekRune()
		if !isIdent(c) {
			break
		}
		in.readRune()
	}

	// Call endToken to set val.tok to identifier we just scanned,
	// so we can look to see if val.tok is a keyword.
	in.endToken(val)
	if k := keywordToken[val.tok]; k != 0 {
		return k
	}
	switch val.tok {
	case "pass":
		return _PASS
	case "break":
		return _BREAK
	case "continue":
		return _CONTINUE
	}
	if len(val.tok) > 0 && val.tok[0] >= '0' && val.tok[0] <= '9' {
		return _NUMBER
	}
	return _IDENT
}

// isIdent reports whether c is an identifier rune.
// We treat all non-ASCII runes as identifier runes.
func isIdent(c int) bool {
	return '0' <= c && c <= '9' ||
		'A' <= c && c <= 'Z' ||
		'a' <= c && c <= 'z' ||
		c == '_' ||
		c >= 0x80
}

// keywordToken records the special tokens for
// strings that should not be treated as ordinary identifiers.
var keywordToken = map[string]int{
	"and":    _AND,
	"for":    _FOR,
	"if":     _IF,
	"else":   _ELSE,
	"elif":   _ELIF,
	"in":     _IN,
	"is":     _IS,
	"lambda": _LAMBDA,
	"load":   _LOAD,
	"not":    _NOT,
	"or":     _OR,
	"def":    _DEF,
	"return": _RETURN,
}

// Comment assignment.
// We build two lists of all subexpressions, preorder and postorder.
// The preorder list is ordered by start location, with outer expressions first.
// The postorder list is ordered by end location, with outer expressions last.
// We use the preorder list to assign each whole-line comment to the syntax
// immediately following it, and we use the postorder list to assign each
// end-of-line comment to the syntax immediately preceding it.

// order walks the expression adding it and its subexpressions to the
// preorder and postorder lists.
func (in *input) order(v Expr) {
	if v != nil {
		in.pre = append(in.pre, v)
	}
	switch v := v.(type) {
	default:
		panic(fmt.Errorf("order: unexpected type %T", v))
	case nil:
		// nothing
	case *End:
		// nothing
	case *File:
		for _, stmt := range v.Stmt {
			in.order(stmt)
		}
	case *CommentBlock:
		// nothing
	case *CallExpr:
		in.order(v.X)
		for _, x := range v.List {
			in.order(x)
		}
		in.order(&v.End)
	case *LoadStmt:
		in.order(v.Module)
		for i := range v.From {
			in.order(v.To[i])
			in.order(v.From[i])
		}
		in.order(&v.Rparen)
	case *LiteralExpr:
		// nothing
	case *StringExpr:
		// nothing
	case *Ident:
		// nothing
	case *BranchStmt:
		// nothing
	case *DotExpr:
		in.order(v.X)
	case *ListExpr:
		for _, x := range v.List {
			in.order(x)
		}
		in.order(&v.End)
	case *Comprehension:
		in.order(v.Body)
		for _, c := range v.Clauses {
			in.order(c)
		}
		in.order(&v.End)
	case *SetExpr:
		for _, x := range v.List {
			in.order(x)
		}
		in.order(&v.End)
	case *ForClause:
		in.order(v.Vars)
		in.order(v.X)
	case *IfClause:
		in.order(v.Cond)
	case *KeyValueExpr:
		in.order(v.Key)
		in.order(v.Value)
	case *DictExpr:
		for _, x := range v.List {
			in.order(x)
		}
		in.order(&v.End)
	case *TupleExpr:
		for _, x := range v.List {
			in.order(x)
		}
		if !v.NoBrackets {
			in.order(&v.End)
		}
	case *UnaryExpr:
		in.order(v.X)
	case *BinaryExpr:
		in.order(v.X)
		in.order(v.Y)
	case *AssignExpr:
		in.order(v.LHS)
		in.order(v.RHS)
	case *ConditionalExpr:
		in.order(v.Then)
		in.order(v.Test)
		in.order(v.Else)
	case *ParenExpr:
		in.order(v.X)
		in.order(&v.End)
	case *SliceExpr:
		in.order(v.X)
		in.order(v.From)
		in.order(v.To)
		in.order(v.Step)
	case *IndexExpr:
		in.order(v.X)
		in.order(v.Y)
	case *LambdaExpr:
		for _, param := range v.Params {
			in.order(param)
		}
		for _, expr := range v.Body {
			in.order(expr)
		}
	case *ReturnStmt:
		if v.Result != nil {
			in.order(v.Result)
		}
	case *DefStmt:
		for _, x := range v.Params {
			in.order(x)
		}
		for _, x := range v.Body {
			in.order(x)
		}
	case *ForStmt:
		in.order(v.Vars)
		in.order(v.X)
		for _, x := range v.Body {
			in.order(x)
		}
	case *IfStmt:
		in.order(v.Cond)
		for _, s := range v.True {
			in.order(s)
		}
		if len(v.False) > 0 {
			in.order(&v.ElsePos)
		}
		for _, s := range v.False {
			in.order(s)
		}
	}
	if v != nil {
		in.post = append(in.post, v)
	}
}

// assignComments attaches comments to nearby syntax.
func (in *input) assignComments() {
	// Generate preorder and postorder lists.
	in.order(in.file)
	in.assignSuffixComments()
	in.assignLineComments()
}

func (in *input) assignSuffixComments() {
	// Assign suffix comments to syntax immediately before.
	suffix := in.suffixComments
	for i := len(in.post) - 1; i >= 0; i-- {
		x := in.post[i]

		// Do not assign suffix comments to file or to block statements
		switch x.(type) {
		case *File, *DefStmt, *IfStmt, *ForStmt, *CommentBlock:
			continue
		}

		_, end := x.Span()
		xcom := x.Comment()
		for len(suffix) > 0 && end.Byte <= suffix[len(suffix)-1].Start.Byte {
			xcom.Suffix = append(xcom.Suffix, suffix[len(suffix)-1])
			suffix = suffix[:len(suffix)-1]
		}
	}

	// We assigned suffix comments in reverse.
	// If multiple suffix comments were appended to the same
	// expression node, they are now in reverse. Fix that.
	for _, x := range in.post {
		reverseComments(x.Comment().Suffix)
	}

	// Remaining suffix comments go at beginning of file.
	in.file.Before = append(in.file.Before, suffix...)
}

func (in *input) assignLineComments() {
	// Assign line comments to syntax immediately following.
	line := in.lineComments
	for _, x := range in.pre {
		start, _ := x.Span()
		xcom := x.Comment()
		for len(line) > 0 && start.Byte >= line[0].Start.Byte {
			xcom.Before = append(xcom.Before, line[0])
			line = line[1:]
		}
		// Line comments can be sorted in a wrong order because they get assigned from different
		// parts of the lexer and the parser. Restore the original order.
		sort.SliceStable(xcom.Before, func(i, j int) bool {
			return xcom.Before[i].Start.Byte < xcom.Before[j].Start.Byte
		})
	}

	// Remaining line comments go at end of file.
	in.file.After = append(in.file.After, line...)
}

// reverseComments reverses the []Comment list.
func reverseComments(list []Comment) {
	for i, j := 0, len(list)-1; i < j; i, j = i+1, j-1 {
		list[i], list[j] = list[j], list[i]
	}
}
