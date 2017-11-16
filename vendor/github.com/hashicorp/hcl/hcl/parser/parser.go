// Package parser implements a parser for HCL (HashiCorp Configuration
// Language)
package parser

import (
	"errors"
	"fmt"
	"strings"

	"github.com/hashicorp/hcl/hcl/ast"
	"github.com/hashicorp/hcl/hcl/scanner"
	"github.com/hashicorp/hcl/hcl/token"
)

type Parser struct {
	sc *scanner.Scanner

	// Last read token
	tok       token.Token
	commaPrev token.Token

	comments    []*ast.CommentGroup
	leadComment *ast.CommentGroup // last lead comment
	lineComment *ast.CommentGroup // last line comment

	enableTrace bool
	indent      int
	n           int // buffer size (max = 1)
}

func newParser(src []byte) *Parser {
	return &Parser{
		sc: scanner.New(src),
	}
}

// Parse returns the fully parsed source and returns the abstract syntax tree.
func Parse(src []byte) (*ast.File, error) {
	p := newParser(src)
	return p.Parse()
}

var errEofToken = errors.New("EOF token found")

// Parse returns the fully parsed source and returns the abstract syntax tree.
func (p *Parser) Parse() (*ast.File, error) {
	f := &ast.File{}
	var err, scerr error
	p.sc.Error = func(pos token.Pos, msg string) {
		scerr = &PosError{Pos: pos, Err: errors.New(msg)}
	}

	f.Node, err = p.objectList()
	if scerr != nil {
		return nil, scerr
	}
	if err != nil {
		return nil, err
	}

	f.Comments = p.comments
	return f, nil
}

func (p *Parser) objectList() (*ast.ObjectList, error) {
	defer un(trace(p, "ParseObjectList"))
	node := &ast.ObjectList{}

	for {
		n, err := p.objectItem()
		if err == errEofToken {
			break // we are finished
		}

		// we don't return a nil node, because might want to use already
		// collected items.
		if err != nil {
			return node, err
		}

		node.Add(n)

		// object lists can be optionally comma-delimited e.g. when a list of maps
		// is being expressed, so a comma is allowed here - it's simply consumed
		tok := p.scan()
		if tok.Type != token.COMMA {
			p.unscan()
		}
	}
	return node, nil
}

func (p *Parser) consumeComment() (comment *ast.Comment, endline int) {
	endline = p.tok.Pos.Line

	// count the endline if it's multiline comment, ie starting with /*
	if len(p.tok.Text) > 1 && p.tok.Text[1] == '*' {
		// don't use range here - no need to decode Unicode code points
		for i := 0; i < len(p.tok.Text); i++ {
			if p.tok.Text[i] == '\n' {
				endline++
			}
		}
	}

	comment = &ast.Comment{Start: p.tok.Pos, Text: p.tok.Text}
	p.tok = p.sc.Scan()
	return
}

func (p *Parser) consumeCommentGroup(n int) (comments *ast.CommentGroup, endline int) {
	var list []*ast.Comment
	endline = p.tok.Pos.Line

	for p.tok.Type == token.COMMENT && p.tok.Pos.Line <= endline+n {
		var comment *ast.Comment
		comment, endline = p.consumeComment()
		list = append(list, comment)
	}

	// add comment group to the comments list
	comments = &ast.CommentGroup{List: list}
	p.comments = append(p.comments, comments)

	return
}

// objectItem parses a single object item
func (p *Parser) objectItem() (*ast.ObjectItem, error) {
	defer un(trace(p, "ParseObjectItem"))

	keys, err := p.objectKey()
	if len(keys) > 0 && err == errEofToken {
		// We ignore eof token here since it is an error if we didn't
		// receive a value (but we did receive a key) for the item.
		err = nil
	}
	if len(keys) > 0 && err != nil && p.tok.Type == token.RBRACE {
		// This is a strange boolean statement, but what it means is:
		// We have keys with no value, and we're likely in an object
		// (since RBrace ends an object). For this, we set err to nil so
		// we continue and get the error below of having the wrong value
		// type.
		err = nil

		// Reset the token type so we don't think it completed fine. See
		// objectType which uses p.tok.Type to check if we're done with
		// the object.
		p.tok.Type = token.EOF
	}
	if err != nil {
		return nil, err
	}

	o := &ast.ObjectItem{
		Keys: keys,
	}

	if p.leadComment != nil {
		o.LeadComment = p.leadComment
		p.leadComment = nil
	}

	switch p.tok.Type {
	case token.ASSIGN:
		o.Assign = p.tok.Pos
		o.Val, err = p.object()
		if err != nil {
			return nil, err
		}
	case token.LBRACE:
		o.Val, err = p.objectType()
		if err != nil {
			return nil, err
		}
	default:
		keyStr := make([]string, 0, len(keys))
		for _, k := range keys {
			keyStr = append(keyStr, k.Token.Text)
		}

		return nil, fmt.Errorf(
			"key '%s' expected start of object ('{') or assignment ('=')",
			strings.Join(keyStr, " "))
	}

	// do a look-ahead for line comment
	p.scan()
	if len(keys) > 0 && o.Val.Pos().Line == keys[0].Pos().Line && p.lineComment != nil {
		o.LineComment = p.lineComment
		p.lineComment = nil
	}
	p.unscan()
	return o, nil
}

// objectKey parses an object key and returns a ObjectKey AST
func (p *Parser) objectKey() ([]*ast.ObjectKey, error) {
	keyCount := 0
	keys := make([]*ast.ObjectKey, 0)

	for {
		tok := p.scan()
		switch tok.Type {
		case token.EOF:
			// It is very important to also return the keys here as well as
			// the error. This is because we need to be able to tell if we
			// did parse keys prior to finding the EOF, or if we just found
			// a bare EOF.
			return keys, errEofToken
		case token.ASSIGN:
			// assignment or object only, but not nested objects. this is not
			// allowed: `foo bar = {}`
			if keyCount > 1 {
				return nil, &PosError{
					Pos: p.tok.Pos,
					Err: fmt.Errorf("nested object expected: LBRACE got: %s", p.tok.Type),
				}
			}

			if keyCount == 0 {
				return nil, &PosError{
					Pos: p.tok.Pos,
					Err: errors.New("no object keys found!"),
				}
			}

			return keys, nil
		case token.LBRACE:
			var err error

			// If we have no keys, then it is a syntax error. i.e. {{}} is not
			// allowed.
			if len(keys) == 0 {
				err = &PosError{
					Pos: p.tok.Pos,
					Err: fmt.Errorf("expected: IDENT | STRING got: %s", p.tok.Type),
				}
			}

			// object
			return keys, err
		case token.IDENT, token.STRING:
			keyCount++
			keys = append(keys, &ast.ObjectKey{Token: p.tok})
		case token.ILLEGAL:
			fmt.Println("illegal")
		default:
			return keys, &PosError{
				Pos: p.tok.Pos,
				Err: fmt.Errorf("expected: IDENT | STRING | ASSIGN | LBRACE got: %s", p.tok.Type),
			}
		}
	}
}

// object parses any type of object, such as number, bool, string, object or
// list.
func (p *Parser) object() (ast.Node, error) {
	defer un(trace(p, "ParseType"))
	tok := p.scan()

	switch tok.Type {
	case token.NUMBER, token.FLOAT, token.BOOL, token.STRING, token.HEREDOC:
		return p.literalType()
	case token.LBRACE:
		return p.objectType()
	case token.LBRACK:
		return p.listType()
	case token.COMMENT:
		// implement comment
	case token.EOF:
		return nil, errEofToken
	}

	return nil, &PosError{
		Pos: tok.Pos,
		Err: fmt.Errorf("Unknown token: %+v", tok),
	}
}

// objectType parses an object type and returns a ObjectType AST
func (p *Parser) objectType() (*ast.ObjectType, error) {
	defer un(trace(p, "ParseObjectType"))

	// we assume that the currently scanned token is a LBRACE
	o := &ast.ObjectType{
		Lbrace: p.tok.Pos,
	}

	l, err := p.objectList()

	// if we hit RBRACE, we are good to go (means we parsed all Items), if it's
	// not a RBRACE, it's an syntax error and we just return it.
	if err != nil && p.tok.Type != token.RBRACE {
		return nil, err
	}

	// If there is no error, we should be at a RBRACE to end the object
	if p.tok.Type != token.RBRACE {
		return nil, fmt.Errorf("object expected closing RBRACE got: %s", p.tok.Type)
	}

	o.List = l
	o.Rbrace = p.tok.Pos // advanced via parseObjectList
	return o, nil
}

// listType parses a list type and returns a ListType AST
func (p *Parser) listType() (*ast.ListType, error) {
	defer un(trace(p, "ParseListType"))

	// we assume that the currently scanned token is a LBRACK
	l := &ast.ListType{
		Lbrack: p.tok.Pos,
	}

	needComma := false
	for {
		tok := p.scan()
		if needComma {
			switch tok.Type {
			case token.COMMA, token.RBRACK:
			default:
				return nil, &PosError{
					Pos: tok.Pos,
					Err: fmt.Errorf(
						"error parsing list, expected comma or list end, got: %s",
						tok.Type),
				}
			}
		}
		switch tok.Type {
		case token.NUMBER, token.FLOAT, token.STRING, token.HEREDOC:
			node, err := p.literalType()
			if err != nil {
				return nil, err
			}

			l.Add(node)
			needComma = true
		case token.COMMA:
			// get next list item or we are at the end
			// do a look-ahead for line comment
			p.scan()
			if p.lineComment != nil && len(l.List) > 0 {
				lit, ok := l.List[len(l.List)-1].(*ast.LiteralType)
				if ok {
					lit.LineComment = p.lineComment
					l.List[len(l.List)-1] = lit
					p.lineComment = nil
				}
			}
			p.unscan()

			needComma = false
			continue
		case token.LBRACE:
			// Looks like a nested object, so parse it out
			node, err := p.objectType()
			if err != nil {
				return nil, &PosError{
					Pos: tok.Pos,
					Err: fmt.Errorf(
						"error while trying to parse object within list: %s", err),
				}
			}
			l.Add(node)
			needComma = true
		case token.BOOL:
			// TODO(arslan) should we support? not supported by HCL yet
		case token.LBRACK:
			// TODO(arslan) should we support nested lists? Even though it's
			// written in README of HCL, it's not a part of the grammar
			// (not defined in parse.y)
		case token.RBRACK:
			// finished
			l.Rbrack = p.tok.Pos
			return l, nil
		default:
			return nil, &PosError{
				Pos: tok.Pos,
				Err: fmt.Errorf("unexpected token while parsing list: %s", tok.Type),
			}
		}
	}
}

// literalType parses a literal type and returns a LiteralType AST
func (p *Parser) literalType() (*ast.LiteralType, error) {
	defer un(trace(p, "ParseLiteral"))

	return &ast.LiteralType{
		Token: p.tok,
	}, nil
}

// scan returns the next token from the underlying scanner. If a token has
// been unscanned then read that instead. In the process, it collects any
// comment groups encountered, and remembers the last lead and line comments.
func (p *Parser) scan() token.Token {
	// If we have a token on the buffer, then return it.
	if p.n != 0 {
		p.n = 0
		return p.tok
	}

	// Otherwise read the next token from the scanner and Save it to the buffer
	// in case we unscan later.
	prev := p.tok
	p.tok = p.sc.Scan()

	if p.tok.Type == token.COMMENT {
		var comment *ast.CommentGroup
		var endline int

		// fmt.Printf("p.tok.Pos.Line = %+v prev: %d endline %d \n",
		// p.tok.Pos.Line, prev.Pos.Line, endline)
		if p.tok.Pos.Line == prev.Pos.Line {
			// The comment is on same line as the previous token; it
			// cannot be a lead comment but may be a line comment.
			comment, endline = p.consumeCommentGroup(0)
			if p.tok.Pos.Line != endline {
				// The next token is on a different line, thus
				// the last comment group is a line comment.
				p.lineComment = comment
			}
		}

		// consume successor comments, if any
		endline = -1
		for p.tok.Type == token.COMMENT {
			comment, endline = p.consumeCommentGroup(1)
		}

		if endline+1 == p.tok.Pos.Line && p.tok.Type != token.RBRACE {
			switch p.tok.Type {
			case token.RBRACE, token.RBRACK:
				// Do not count for these cases
			default:
				// The next token is following on the line immediately after the
				// comment group, thus the last comment group is a lead comment.
				p.leadComment = comment
			}
		}

	}

	return p.tok
}

// unscan pushes the previously read token back onto the buffer.
func (p *Parser) unscan() {
	p.n = 1
}

// ----------------------------------------------------------------------------
// Parsing support

func (p *Parser) printTrace(a ...interface{}) {
	if !p.enableTrace {
		return
	}

	const dots = ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
	const n = len(dots)
	fmt.Printf("%5d:%3d: ", p.tok.Pos.Line, p.tok.Pos.Column)

	i := 2 * p.indent
	for i > n {
		fmt.Print(dots)
		i -= n
	}
	// i <= n
	fmt.Print(dots[0:i])
	fmt.Println(a...)
}

func trace(p *Parser, msg string) *Parser {
	p.printTrace(msg, "(")
	p.indent++
	return p
}

// Usage pattern: defer un(trace(p, "..."))
func un(p *Parser) {
	p.indent--
	p.printTrace(")")
}
