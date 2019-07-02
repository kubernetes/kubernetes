package parser

import (
	"errors"
	"fmt"

	"github.com/hashicorp/hcl/hcl/ast"
	hcltoken "github.com/hashicorp/hcl/hcl/token"
	"github.com/hashicorp/hcl/json/scanner"
	"github.com/hashicorp/hcl/json/token"
)

type Parser struct {
	sc *scanner.Scanner

	// Last read token
	tok       token.Token
	commaPrev token.Token

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
		scerr = fmt.Errorf("%s: %s", pos, msg)
	}

	// The root must be an object in JSON
	object, err := p.object()
	if scerr != nil {
		return nil, scerr
	}
	if err != nil {
		return nil, err
	}

	// We make our final node an object list so it is more HCL compatible
	f.Node = object.List

	// Flatten it, which finds patterns and turns them into more HCL-like
	// AST trees.
	flattenObjects(f.Node)

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

		// Check for a followup comma. If it isn't a comma, then we're done
		if tok := p.scan(); tok.Type != token.COMMA {
			break
		}
	}

	return node, nil
}

// objectItem parses a single object item
func (p *Parser) objectItem() (*ast.ObjectItem, error) {
	defer un(trace(p, "ParseObjectItem"))

	keys, err := p.objectKey()
	if err != nil {
		return nil, err
	}

	o := &ast.ObjectItem{
		Keys: keys,
	}

	switch p.tok.Type {
	case token.COLON:
		pos := p.tok.Pos
		o.Assign = hcltoken.Pos{
			Filename: pos.Filename,
			Offset:   pos.Offset,
			Line:     pos.Line,
			Column:   pos.Column,
		}

		o.Val, err = p.objectValue()
		if err != nil {
			return nil, err
		}
	}

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
			return nil, errEofToken
		case token.STRING:
			keyCount++
			keys = append(keys, &ast.ObjectKey{
				Token: p.tok.HCLToken(),
			})
		case token.COLON:
			// If we have a zero keycount it means that we never got
			// an object key, i.e. `{ :`. This is a syntax error.
			if keyCount == 0 {
				return nil, fmt.Errorf("expected: STRING got: %s", p.tok.Type)
			}

			// Done
			return keys, nil
		case token.ILLEGAL:
			return nil, errors.New("illegal")
		default:
			return nil, fmt.Errorf("expected: STRING got: %s", p.tok.Type)
		}
	}
}

// object parses any type of object, such as number, bool, string, object or
// list.
func (p *Parser) objectValue() (ast.Node, error) {
	defer un(trace(p, "ParseObjectValue"))
	tok := p.scan()

	switch tok.Type {
	case token.NUMBER, token.FLOAT, token.BOOL, token.NULL, token.STRING:
		return p.literalType()
	case token.LBRACE:
		return p.objectType()
	case token.LBRACK:
		return p.listType()
	case token.EOF:
		return nil, errEofToken
	}

	return nil, fmt.Errorf("Expected object value, got unknown token: %+v", tok)
}

// object parses any type of object, such as number, bool, string, object or
// list.
func (p *Parser) object() (*ast.ObjectType, error) {
	defer un(trace(p, "ParseType"))
	tok := p.scan()

	switch tok.Type {
	case token.LBRACE:
		return p.objectType()
	case token.EOF:
		return nil, errEofToken
	}

	return nil, fmt.Errorf("Expected object, got unknown token: %+v", tok)
}

// objectType parses an object type and returns a ObjectType AST
func (p *Parser) objectType() (*ast.ObjectType, error) {
	defer un(trace(p, "ParseObjectType"))

	// we assume that the currently scanned token is a LBRACE
	o := &ast.ObjectType{}

	l, err := p.objectList()

	// if we hit RBRACE, we are good to go (means we parsed all Items), if it's
	// not a RBRACE, it's an syntax error and we just return it.
	if err != nil && p.tok.Type != token.RBRACE {
		return nil, err
	}

	o.List = l
	return o, nil
}

// listType parses a list type and returns a ListType AST
func (p *Parser) listType() (*ast.ListType, error) {
	defer un(trace(p, "ParseListType"))

	// we assume that the currently scanned token is a LBRACK
	l := &ast.ListType{}

	for {
		tok := p.scan()
		switch tok.Type {
		case token.NUMBER, token.FLOAT, token.STRING:
			node, err := p.literalType()
			if err != nil {
				return nil, err
			}

			l.Add(node)
		case token.COMMA:
			continue
		case token.LBRACE:
			node, err := p.objectType()
			if err != nil {
				return nil, err
			}

			l.Add(node)
		case token.BOOL:
			// TODO(arslan) should we support? not supported by HCL yet
		case token.LBRACK:
			// TODO(arslan) should we support nested lists? Even though it's
			// written in README of HCL, it's not a part of the grammar
			// (not defined in parse.y)
		case token.RBRACK:
			// finished
			return l, nil
		default:
			return nil, fmt.Errorf("unexpected token while parsing list: %s", tok.Type)
		}

	}
}

// literalType parses a literal type and returns a LiteralType AST
func (p *Parser) literalType() (*ast.LiteralType, error) {
	defer un(trace(p, "ParseLiteral"))

	return &ast.LiteralType{
		Token: p.tok.HCLToken(),
	}, nil
}

// scan returns the next token from the underlying scanner. If a token has
// been unscanned then read that instead.
func (p *Parser) scan() token.Token {
	// If we have a token on the buffer, then return it.
	if p.n != 0 {
		p.n = 0
		return p.tok
	}

	p.tok = p.sc.Scan()
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
