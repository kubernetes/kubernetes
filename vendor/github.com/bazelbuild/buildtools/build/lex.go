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
	"strings"
	"unicode/utf8"

	"github.com/bazelbuild/buildtools/tables"
)

// Parse parses the input data and returns the corresponding parse tree.
//
// The filename is used only for generating error messages.
func Parse(filename string, data []byte) (*File, error) {
	in := newInput(filename, data)
	return in.parse()
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
	endStmt        int       // position of the end of the current statement
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
		endStmt:   -1, // -1 denotes it's not inside a statement
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
				err = fmt.Errorf("%s:%d:%d: internal error: %v", in.filename, in.pos.Line, in.pos.LineRune, e)
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
	in.parseError = fmt.Errorf("%s:%d:%d: %v", in.filename, in.pos.Line, in.pos.LineRune, s)
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
		// If a single statement is split into multiple lines, we don't need
		// to track indentations and unindentations within these lines. For example:
		//
		// def f(
		//     # This indentation should be ignored
		//     x):
		//  # This unindentation should be ignored
		//  # Actual indentation is from 0 to 2 spaces here
		//  return x
		//
		// To handle this case, when we reach the beginning of a statement  we scan forward to see where
		// it should end and record the number of input bytes remaining at that endpoint.
		//
		// If --format_bzl is set to false, top level blocks (e.g. an entire function definition)
		// is considered as a single statement.
		if in.endStmt != -1 && len(in.remaining) == in.endStmt {
			in.endStmt = -1
		}

		// Skip over spaces. Count newlines so we can give the parser
		// information about where top-level blank lines are,
		// for top-level comment assignment.
		c := in.peekRune()
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			if c == '\n' {
				in.indent = 0
				in.cleanLine = true
				if in.endStmt == -1 {
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
			in.cleanLine = true

			// Is this comment the only thing on its line?
			// Find the last \n before this # and see if it's all
			// spaces from there to here.
			// If it's a suffix comment but the last non-space symbol before
			// it is one of (, [, or {, treat it as a line comment that should be
			// put inside the corresponding block.
			i := bytes.LastIndex(in.complete[:in.pos.Byte], []byte("\n"))
			prefix := bytes.TrimSpace(in.complete[i+1 : in.pos.Byte])
			isSuffix := true
			if len(prefix) == 0 ||
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
			if in.endStmt == -1 {
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
	// Skip if --format_bzl is set to false, if we're inside a statement, or if there were non-space
	// characters before in the current line.
	if tables.FormatBzlFiles && in.endStmt == -1 && in.cleanLine {
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

	// If endStmt is 0, we need to recompute where the end of the next statement is.
	if in.endStmt == -1 {
		in.endStmt = len(in.skipStmt(in.remaining))
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

	case '<', '>', '=', '!', '+', '-', '*', '/', '%': // possibly followed by =
		in.readRune()
		if c == '/' && in.peekRune() == '/' {
			// integer division
			in.readRune()
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
		s, triple, err := unquote(val.tok)
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

	if !tables.FormatBzlFiles {
		// Look for raw Python block (class, def, if, etc at beginning of line) and pass through.
		if in.depth == 0 && in.pos.LineRune == 1 && hasPythonPrefix(in.remaining) {
			// Find end of Python block and advance input beyond it.
			// Have to loop calling readRune in order to maintain line number info.
			rest := in.skipStmt(in.remaining)
			for len(in.remaining) > len(rest) {
				in.readRune()
			}
			return _PYTHON
		}
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

// Python scanning.
// About 1% of BUILD files embed arbitrary Python into the file.
// We do not attempt to parse it. Instead, we lex just enough to scan
// beyond it, treating the Python block as an unintepreted blob.

// hasPythonPrefix reports whether p begins with a keyword that would
// introduce an uninterpreted Python block.
func hasPythonPrefix(p []byte) bool {
	if tables.FormatBzlFiles {
		return false
	}

	for _, pre := range prefixes {
		if hasPrefixSpace(p, pre) {
			return true
		}
	}
	return false
}

// These keywords introduce uninterpreted Python blocks.
var prefixes = []string{
	"assert",
	"class",
	"def",
	"del",
	"for",
	"if",
	"try",
	"else",
	"elif",
	"except",
}

// hasPrefixSpace reports whether p begins with pre followed by a space or colon.
func hasPrefixSpace(p []byte, pre string) bool {

	if len(p) <= len(pre) || p[len(pre)] != ' ' && p[len(pre)] != '\t' && p[len(pre)] != ':' {
		return false
	}
	for i := range pre {
		if p[i] != pre[i] {
			return false
		}
	}
	return true
}

// A utility function for the legacy formatter.
// Returns whether a given code starts with a top-level statement (maybe with some preceeding
// comments and blank lines)
func isOutsideBlock(b []byte) bool {
	isBlankLine := true
	isComment := false
	for _, c := range b {
		switch {
		case c == ' ' || c == '\t' || c == '\r':
			isBlankLine = false
		case c == '#':
			isBlankLine = false
			isComment = true
		case c == '\n':
			isBlankLine = true
			isComment = false
		default:
			if !isComment {
				return isBlankLine
			}
		}
	}
	return true
}

// skipStmt returns the data remaining after the statement  beginning at p.
// It does not advance the input position.
// (The only reason for the input receiver is to be able to call in.Error.)
func (in *input) skipStmt(p []byte) []byte {
	quote := byte(0)     // if non-zero, the kind of quote we're in
	tripleQuote := false // if true, the quote is a triple quote
	depth := 0           // nesting depth for ( ) [ ] { }
	var rest []byte      // data after the Python block

	defer func() {
		if quote != 0 {
			in.Error("EOF scanning Python quoted string")
		}
	}()

	// Scan over input one byte at a time until we find
	// an unindented, non-blank, non-comment line
	// outside quoted strings and brackets.
	for i := 0; i < len(p); i++ {
		c := p[i]
		if quote != 0 && c == quote && !tripleQuote {
			quote = 0
			continue
		}
		if quote != 0 && c == quote && tripleQuote && i+2 < len(p) && p[i+1] == quote && p[i+2] == quote {
			i += 2
			quote = 0
			tripleQuote = false
			continue
		}
		if quote != 0 {
			if c == '\\' {
				i++ // skip escaped char
			}
			continue
		}
		if c == '\'' || c == '"' {
			if i+2 < len(p) && p[i+1] == c && p[i+2] == c {
				quote = c
				tripleQuote = true
				i += 2
				continue
			}
			quote = c
			continue
		}

		if depth == 0 && i > 0 && p[i] == '\n' && p[i-1] != '\\' {
			// Possible stopping point. Save the earliest one we find.
			if rest == nil {
				rest = p[i:]
			}

			if tables.FormatBzlFiles {
				// In the bzl files mode we only care about the end of the statement, we've found it.
				return rest
			}
			// In the legacy mode we need to find where the current block ends
			if isOutsideBlock(p[i+1:]) {
				return rest
			}
			// Not a stopping point after all.
			rest = nil

		}

		switch c {
		case '#':
			// Skip comment.
			for i < len(p) && p[i] != '\n' {
				i++
			}
			// Rewind 1 position back because \n should be handled at the next iteration
			i--

		case '(', '[', '{':
			depth++

		case ')', ']', '}':
			depth--
		}
	}
	return rest
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
	case *PythonBlock:
		// nothing
	case *LiteralExpr:
		// nothing
	case *StringExpr:
		// nothing
	case *DotExpr:
		in.order(v.X)
	case *ListExpr:
		for _, x := range v.List {
			in.order(x)
		}
		in.order(&v.End)
	case *ListForExpr:
		in.order(v.X)
		for _, c := range v.For {
			in.order(c)
		}
		in.order(&v.End)
	case *SetExpr:
		for _, x := range v.List {
			in.order(x)
		}
		in.order(&v.End)
	case *ForClauseWithIfClausesOpt:
		in.order(v.For)
		for _, c := range v.Ifs {
			in.order(c)
		}
	case *ForClause:
		for _, name := range v.Var {
			in.order(name)
		}
		in.order(v.Expr)
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
		in.order(&v.End)
	case *UnaryExpr:
		in.order(v.X)
	case *BinaryExpr:
		in.order(v.X)
		in.order(v.Y)
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
		for _, name := range v.Var {
			in.order(name)
		}
		in.order(v.Expr)
	case *ReturnExpr:
		if v.X != nil {
			in.order(v.X)
		}
	case *FuncDef:
		for _, x := range v.Args {
			in.order(x)
		}
		for _, x := range v.Body.Statements {
			in.order(x)
		}
	case *ForLoop:
		for _, x := range v.LoopVars {
			in.order(x)
		}
		in.order(v.Iterable)
		for _, x := range v.Body.Statements {
			in.order(x)
		}
	case *IfElse:
		for _, condition := range v.Conditions {
			in.order(condition.If)
			for _, x := range condition.Then.Statements {
				in.order(x)
			}
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

	// Assign line comments to syntax immediately following.
	line := in.lineComments
	for _, x := range in.pre {
		start, _ := x.Span()
		xcom := x.Comment()
		for len(line) > 0 && start.Byte >= line[0].Start.Byte {
			xcom.Before = append(xcom.Before, line[0])
			line = line[1:]
		}
	}

	// Remaining line comments go at end of file.
	in.file.After = append(in.file.After, line...)

	// Assign suffix comments to syntax immediately before.
	suffix := in.suffixComments
	for i := len(in.post) - 1; i >= 0; i-- {
		x := in.post[i]

		// Do not assign suffix comments to file
		switch x.(type) {
		case *File:
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

// reverseComments reverses the []Comment list.
func reverseComments(list []Comment) {
	for i, j := 0, len(list)-1; i < j; i, j = i+1, j-1 {
		list[i], list[j] = list[j], list[i]
	}
}
