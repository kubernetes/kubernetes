package syntax

import (
	"errors"
	"fmt"
	"strings"
)

type ParserOptions struct {
	// NoLiterals disables OpChar merging into OpLiteral.
	NoLiterals bool
}

func NewParser(opts *ParserOptions) *Parser {
	return newParser(opts)
}

type Parser struct {
	out      Regexp
	lexer    lexer
	exprPool []Expr

	prefixParselets [256]prefixParselet
	infixParselets  [256]infixParselet

	charClass []Expr
	allocated uint

	opts ParserOptions
}

// ParsePCRE parses PHP-style pattern with delimiters.
// An example of such pattern is `/foo/i`.
func (p *Parser) ParsePCRE(pattern string) (*RegexpPCRE, error) {
	pcre, err := p.newPCRE(pattern)
	if err != nil {
		return nil, err
	}
	if pcre.HasModifier('x') {
		return nil, errors.New("'x' modifier is not supported")
	}
	re, err := p.Parse(pcre.Pattern)
	if re != nil {
		pcre.Expr = re.Expr
	}
	return pcre, err
}

func (p *Parser) Parse(pattern string) (result *Regexp, err error) {
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		if err2, ok := r.(ParseError); ok {
			err = err2
			return
		}
		panic(r)
	}()

	p.lexer.Init(pattern)
	p.allocated = 0
	p.out.Pattern = pattern
	if pattern == "" {
		p.out.Expr = *p.newExpr(OpConcat, Position{})
	} else {
		p.out.Expr = *p.parseExpr(0)
	}

	if !p.opts.NoLiterals {
		p.mergeChars(&p.out.Expr)
	}
	p.setValues(&p.out.Expr)

	return &p.out, nil
}

type prefixParselet func(token) *Expr

type infixParselet func(*Expr, token) *Expr

func newParser(opts *ParserOptions) *Parser {
	var p Parser

	if opts != nil {
		p.opts = *opts
	}
	p.exprPool = make([]Expr, 256)

	for tok, op := range tok2op {
		if op != 0 {
			p.prefixParselets[tokenKind(tok)] = p.parsePrefixElementary
		}
	}

	p.prefixParselets[tokEscapeHexFull] = func(tok token) *Expr {
		return p.newExprForm(OpEscapeHex, FormEscapeHexFull, tok.pos)
	}
	p.prefixParselets[tokEscapeUniFull] = func(tok token) *Expr {
		return p.newExprForm(OpEscapeUni, FormEscapeUniFull, tok.pos)
	}

	p.prefixParselets[tokLparen] = func(tok token) *Expr { return p.parseGroup(OpCapture, tok) }
	p.prefixParselets[tokLparenAtomic] = func(tok token) *Expr { return p.parseGroup(OpAtomicGroup, tok) }
	p.prefixParselets[tokLparenPositiveLookahead] = func(tok token) *Expr { return p.parseGroup(OpPositiveLookahead, tok) }
	p.prefixParselets[tokLparenNegativeLookahead] = func(tok token) *Expr { return p.parseGroup(OpNegativeLookahead, tok) }
	p.prefixParselets[tokLparenPositiveLookbehind] = func(tok token) *Expr { return p.parseGroup(OpPositiveLookbehind, tok) }
	p.prefixParselets[tokLparenNegativeLookbehind] = func(tok token) *Expr { return p.parseGroup(OpNegativeLookbehind, tok) }

	p.prefixParselets[tokLparenName] = func(tok token) *Expr {
		return p.parseNamedCapture(FormDefault, tok)
	}
	p.prefixParselets[tokLparenNameAngle] = func(tok token) *Expr {
		return p.parseNamedCapture(FormNamedCaptureAngle, tok)
	}
	p.prefixParselets[tokLparenNameQuote] = func(tok token) *Expr {
		return p.parseNamedCapture(FormNamedCaptureQuote, tok)
	}

	p.prefixParselets[tokLparenFlags] = p.parseGroupWithFlags

	p.prefixParselets[tokPipe] = func(tok token) *Expr {
		// We need prefix pipe parselet to handle `(|x)` syntax.
		right := p.parseExpr(1)
		return p.newExpr(OpAlt, tok.pos, p.newEmpty(tok.pos), right)
	}
	p.prefixParselets[tokLbracket] = func(tok token) *Expr {
		return p.parseCharClass(OpCharClass, tok)
	}
	p.prefixParselets[tokLbracketCaret] = func(tok token) *Expr {
		return p.parseCharClass(OpNegCharClass, tok)
	}

	p.infixParselets[tokRepeat] = func(left *Expr, tok token) *Expr {
		repeatLit := p.newExpr(OpString, tok.pos)
		return p.newExpr(OpRepeat, combinePos(left.Pos, tok.pos), left, repeatLit)
	}
	p.infixParselets[tokStar] = func(left *Expr, tok token) *Expr {
		return p.newExpr(OpStar, combinePos(left.Pos, tok.pos), left)
	}
	p.infixParselets[tokConcat] = func(left *Expr, tok token) *Expr {
		right := p.parseExpr(2)
		if left.Op == OpConcat {
			left.Args = append(left.Args, *right)
			left.Pos.End = right.End()
			return left
		}
		return p.newExpr(OpConcat, combinePos(left.Pos, right.Pos), left, right)
	}
	p.infixParselets[tokPipe] = p.parseAlt
	p.infixParselets[tokMinus] = p.parseMinus
	p.infixParselets[tokPlus] = p.parsePlus
	p.infixParselets[tokQuestion] = p.parseQuestion

	return &p
}

func (p *Parser) setValues(e *Expr) {
	for i := range e.Args {
		p.setValues(&e.Args[i])
	}
	e.Value = p.exprValue(e)
}

func (p *Parser) exprValue(e *Expr) string {
	return p.out.Pattern[e.Begin():e.End()]
}

func (p *Parser) mergeChars(e *Expr) {
	for i := range e.Args {
		p.mergeChars(&e.Args[i])
	}
	if e.Op != OpConcat || len(e.Args) < 2 {
		return
	}

	args := e.Args[:0]
	i := 0
	for i < len(e.Args) {
		first := i
		chars := 0
		for j := i; j < len(e.Args) && e.Args[j].Op == OpChar; j++ {
			chars++
		}
		if chars > 1 {
			c1 := e.Args[first]
			c2 := e.Args[first+chars-1]
			lit := p.newExpr(OpLiteral, combinePos(c1.Pos, c2.Pos))
			for j := 0; j < chars; j++ {
				lit.Args = append(lit.Args, e.Args[first+j])
			}
			args = append(args, *lit)
			i += chars
		} else {
			args = append(args, e.Args[i])
			i++
		}
	}
	if len(args) == 1 {
		*e = args[0] // Turn OpConcat into OpLiteral
	} else {
		e.Args = args
	}
}

func (p *Parser) newEmpty(pos Position) *Expr {
	return p.newExpr(OpConcat, pos)
}

func (p *Parser) newExprForm(op Operation, form Form, pos Position, args ...*Expr) *Expr {
	e := p.newExpr(op, pos, args...)
	e.Form = form
	return e
}

func (p *Parser) newExpr(op Operation, pos Position, args ...*Expr) *Expr {
	e := p.allocExpr()
	*e = Expr{
		Op:   op,
		Pos:  pos,
		Args: e.Args[:0],
	}
	for _, arg := range args {
		e.Args = append(e.Args, *arg)
	}
	return e
}

func (p *Parser) allocExpr() *Expr {
	i := p.allocated
	if i < uint(len(p.exprPool)) {
		p.allocated++
		return &p.exprPool[i]
	}
	return &Expr{}
}

func (p *Parser) expect(kind tokenKind) Position {
	tok := p.lexer.NextToken()
	if tok.kind != kind {
		throwErrorf(int(tok.pos.Begin), int(tok.pos.End), "expected '%s', found '%s'", kind, tok.kind)
	}
	return tok.pos
}

func (p *Parser) parseExpr(precedence int) *Expr {
	tok := p.lexer.NextToken()
	prefix := p.prefixParselets[tok.kind]
	if prefix == nil {
		throwfPos(tok.pos, "unexpected token: %v", tok)
	}
	left := prefix(tok)

	for precedence < p.precedenceOf(p.lexer.Peek()) {
		tok := p.lexer.NextToken()
		infix := p.infixParselets[tok.kind]
		left = infix(left, tok)
	}

	return left
}

func (p *Parser) parsePrefixElementary(tok token) *Expr {
	return p.newExpr(tok2op[tok.kind], tok.pos)
}

func (p *Parser) parseCharClass(op Operation, tok token) *Expr {
	var endPos Position
	p.charClass = p.charClass[:0]
	for {
		p.charClass = append(p.charClass, *p.parseExpr(0))
		next := p.lexer.Peek()
		if next.kind == tokRbracket {
			endPos = next.pos
			p.lexer.NextToken()
			break
		}
		if next.kind == tokNone {
			throwfPos(tok.pos, "unterminated '['")
		}
	}

	result := p.newExpr(op, combinePos(tok.pos, endPos))
	result.Args = append(result.Args, p.charClass...)
	return result
}

func (p *Parser) parseMinus(left *Expr, tok token) *Expr {
	if p.isValidCharRangeOperand(left) {
		if p.lexer.Peek().kind != tokRbracket {
			right := p.parseExpr(2)
			return p.newExpr(OpCharRange, combinePos(left.Pos, right.Pos), left, right)
		}
	}
	p.charClass = append(p.charClass, *left)
	return p.newExpr(OpChar, tok.pos)
}

func (p *Parser) isValidCharRangeOperand(e *Expr) bool {
	switch e.Op {
	case OpEscapeHex, OpEscapeOctal, OpEscapeMeta, OpChar:
		return true
	case OpEscapeChar:
		switch p.exprValue(e) {
		case `\\`, `\|`, `\*`, `\+`, `\?`, `\.`, `\[`, `\^`, `\$`, `\(`, `\)`:
			return true
		}
	}
	return false
}

func (p *Parser) parsePlus(left *Expr, tok token) *Expr {
	op := OpPlus
	switch left.Op {
	case OpPlus, OpStar, OpQuestion, OpRepeat:
		op = OpPossessive
	}
	return p.newExpr(op, combinePos(left.Pos, tok.pos), left)
}

func (p *Parser) parseQuestion(left *Expr, tok token) *Expr {
	op := OpQuestion
	switch left.Op {
	case OpPlus, OpStar, OpQuestion, OpRepeat:
		op = OpNonGreedy
	}
	return p.newExpr(op, combinePos(left.Pos, tok.pos), left)
}

func (p *Parser) parseAlt(left *Expr, tok token) *Expr {
	var right *Expr
	switch p.lexer.Peek().kind {
	case tokRparen, tokNone:
		// This is needed to handle `(x|)` syntax.
		right = p.newEmpty(tok.pos)
	default:
		right = p.parseExpr(1)
	}
	if left.Op == OpAlt {
		left.Args = append(left.Args, *right)
		left.Pos.End = right.End()
		return left
	}
	return p.newExpr(OpAlt, combinePos(left.Pos, right.Pos), left, right)
}

func (p *Parser) parseGroupItem(tok token) *Expr {
	if p.lexer.Peek().kind == tokRparen {
		// This is needed to handle `() syntax.`
		return p.newEmpty(tok.pos)
	}
	return p.parseExpr(0)
}

func (p *Parser) parseGroup(op Operation, tok token) *Expr {
	x := p.parseGroupItem(tok)
	result := p.newExpr(op, tok.pos, x)
	result.Pos.End = p.expect(tokRparen).End
	return result
}

func (p *Parser) parseNamedCapture(form Form, tok token) *Expr {
	prefixLen := len("(?<")
	if form == FormDefault {
		prefixLen = len("(?P<")
	}
	name := p.newExpr(OpString, Position{
		Begin: tok.pos.Begin + uint16(prefixLen),
		End:   tok.pos.End - uint16(len(">")),
	})
	x := p.parseGroupItem(tok)
	result := p.newExprForm(OpNamedCapture, form, tok.pos, x, name)
	result.Pos.End = p.expect(tokRparen).End
	return result
}

func (p *Parser) parseGroupWithFlags(tok token) *Expr {
	var result *Expr
	val := p.out.Pattern[tok.pos.Begin+1 : tok.pos.End]
	switch {
	case !strings.HasSuffix(val, ":"):
		flags := p.newExpr(OpString, Position{
			Begin: tok.pos.Begin + uint16(len("(?")),
			End:   tok.pos.End,
		})
		result = p.newExpr(OpFlagOnlyGroup, tok.pos, flags)
	case val == "?:":
		x := p.parseGroupItem(tok)
		result = p.newExpr(OpGroup, tok.pos, x)
	default:
		flags := p.newExpr(OpString, Position{
			Begin: tok.pos.Begin + uint16(len("(?")),
			End:   tok.pos.End - uint16(len(":")),
		})
		x := p.parseGroupItem(tok)
		result = p.newExpr(OpGroupWithFlags, tok.pos, x, flags)
	}
	result.Pos.End = p.expect(tokRparen).End
	return result
}

func (p *Parser) precedenceOf(tok token) int {
	switch tok.kind {
	case tokPipe:
		return 1
	case tokConcat, tokMinus:
		return 2
	case tokPlus, tokStar, tokQuestion, tokRepeat:
		return 3
	default:
		return 0
	}
}

func (p *Parser) newPCRE(source string) (*RegexpPCRE, error) {
	if source == "" {
		return nil, errors.New("empty pattern: can't find delimiters")
	}

	delim := source[0]
	endDelim := delim
	switch delim {
	case '(':
		endDelim = ')'
	case '{':
		endDelim = '}'
	case '[':
		endDelim = ']'
	case '<':
		endDelim = '>'
	case '\\':
		return nil, errors.New("'\\' is not a valid delimiter")
	default:
		if isSpace(delim) {
			return nil, errors.New("whitespace is not a valid delimiter")
		}
		if isAlphanumeric(delim) {
			return nil, fmt.Errorf("'%c' is not a valid delimiter", delim)
		}
	}

	j := strings.LastIndexByte(source, endDelim)
	if j == -1 {
		return nil, fmt.Errorf("can't find '%c' ending delimiter", endDelim)
	}

	pcre := &RegexpPCRE{
		Pattern:   source[1:j],
		Source:    source,
		Delim:     [2]byte{delim, endDelim},
		Modifiers: source[j+1:],
	}
	return pcre, nil
}

var tok2op = [256]Operation{
	tokDollar:      OpDollar,
	tokCaret:       OpCaret,
	tokDot:         OpDot,
	tokChar:        OpChar,
	tokMinus:       OpChar,
	tokEscapeChar:  OpEscapeChar,
	tokEscapeMeta:  OpEscapeMeta,
	tokEscapeHex:   OpEscapeHex,
	tokEscapeOctal: OpEscapeOctal,
	tokEscapeUni:   OpEscapeUni,
	tokPosixClass:  OpPosixClass,
	tokQ:           OpQuote,
	tokComment:     OpComment,
}
