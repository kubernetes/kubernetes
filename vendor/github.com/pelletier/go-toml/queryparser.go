/*
  Based on the "jsonpath" spec/concept.

  http://goessner.net/articles/JsonPath/
  https://code.google.com/p/json-path/
*/

package toml

import (
	"fmt"
)

const maxInt = int(^uint(0) >> 1)

type queryParser struct {
	flow         chan token
	tokensBuffer []token
	query        *Query
	union        []pathFn
	err          error
}

type queryParserStateFn func() queryParserStateFn

// Formats and panics an error message based on a token
func (p *queryParser) parseError(tok *token, msg string, args ...interface{}) queryParserStateFn {
	p.err = fmt.Errorf(tok.Position.String()+": "+msg, args...)
	return nil // trigger parse to end
}

func (p *queryParser) run() {
	for state := p.parseStart; state != nil; {
		state = state()
	}
}

func (p *queryParser) backup(tok *token) {
	p.tokensBuffer = append(p.tokensBuffer, *tok)
}

func (p *queryParser) peek() *token {
	if len(p.tokensBuffer) != 0 {
		return &(p.tokensBuffer[0])
	}

	tok, ok := <-p.flow
	if !ok {
		return nil
	}
	p.backup(&tok)
	return &tok
}

func (p *queryParser) lookahead(types ...tokenType) bool {
	result := true
	buffer := []token{}

	for _, typ := range types {
		tok := p.getToken()
		if tok == nil {
			result = false
			break
		}
		buffer = append(buffer, *tok)
		if tok.typ != typ {
			result = false
			break
		}
	}
	// add the tokens back to the buffer, and return
	p.tokensBuffer = append(p.tokensBuffer, buffer...)
	return result
}

func (p *queryParser) getToken() *token {
	if len(p.tokensBuffer) != 0 {
		tok := p.tokensBuffer[0]
		p.tokensBuffer = p.tokensBuffer[1:]
		return &tok
	}
	tok, ok := <-p.flow
	if !ok {
		return nil
	}
	return &tok
}

func (p *queryParser) parseStart() queryParserStateFn {
	tok := p.getToken()

	if tok == nil || tok.typ == tokenEOF {
		return nil
	}

	if tok.typ != tokenDollar {
		return p.parseError(tok, "Expected '$' at start of expression")
	}

	return p.parseMatchExpr
}

// handle '.' prefix, '[]', and '..'
func (p *queryParser) parseMatchExpr() queryParserStateFn {
	tok := p.getToken()
	switch tok.typ {
	case tokenDotDot:
		p.query.appendPath(&matchRecursiveFn{})
		// nested parse for '..'
		tok := p.getToken()
		switch tok.typ {
		case tokenKey:
			p.query.appendPath(newMatchKeyFn(tok.val))
			return p.parseMatchExpr
		case tokenLeftBracket:
			return p.parseBracketExpr
		case tokenStar:
			// do nothing - the recursive predicate is enough
			return p.parseMatchExpr
		}

	case tokenDot:
		// nested parse for '.'
		tok := p.getToken()
		switch tok.typ {
		case tokenKey:
			p.query.appendPath(newMatchKeyFn(tok.val))
			return p.parseMatchExpr
		case tokenStar:
			p.query.appendPath(&matchAnyFn{})
			return p.parseMatchExpr
		}

	case tokenLeftBracket:
		return p.parseBracketExpr

	case tokenEOF:
		return nil // allow EOF at this stage
	}
	return p.parseError(tok, "expected match expression")
}

func (p *queryParser) parseBracketExpr() queryParserStateFn {
	if p.lookahead(tokenInteger, tokenColon) {
		return p.parseSliceExpr
	}
	if p.peek().typ == tokenColon {
		return p.parseSliceExpr
	}
	return p.parseUnionExpr
}

func (p *queryParser) parseUnionExpr() queryParserStateFn {
	var tok *token

	// this state can be traversed after some sub-expressions
	// so be careful when setting up state in the parser
	if p.union == nil {
		p.union = []pathFn{}
	}

loop: // labeled loop for easy breaking
	for {
		if len(p.union) > 0 {
			// parse delimiter or terminator
			tok = p.getToken()
			switch tok.typ {
			case tokenComma:
				// do nothing
			case tokenRightBracket:
				break loop
			default:
				return p.parseError(tok, "expected ',' or ']', not '%s'", tok.val)
			}
		}

		// parse sub expression
		tok = p.getToken()
		switch tok.typ {
		case tokenInteger:
			p.union = append(p.union, newMatchIndexFn(tok.Int()))
		case tokenKey:
			p.union = append(p.union, newMatchKeyFn(tok.val))
		case tokenString:
			p.union = append(p.union, newMatchKeyFn(tok.val))
		case tokenQuestion:
			return p.parseFilterExpr
		default:
			return p.parseError(tok, "expected union sub expression, not '%s', %d", tok.val, len(p.union))
		}
	}

	// if there is only one sub-expression, use that instead
	if len(p.union) == 1 {
		p.query.appendPath(p.union[0])
	} else {
		p.query.appendPath(&matchUnionFn{p.union})
	}

	p.union = nil // clear out state
	return p.parseMatchExpr
}

func (p *queryParser) parseSliceExpr() queryParserStateFn {
	// init slice to grab all elements
	start, end, step := 0, maxInt, 1

	// parse optional start
	tok := p.getToken()
	if tok.typ == tokenInteger {
		start = tok.Int()
		tok = p.getToken()
	}
	if tok.typ != tokenColon {
		return p.parseError(tok, "expected ':'")
	}

	// parse optional end
	tok = p.getToken()
	if tok.typ == tokenInteger {
		end = tok.Int()
		tok = p.getToken()
	}
	if tok.typ == tokenRightBracket {
		p.query.appendPath(newMatchSliceFn(start, end, step))
		return p.parseMatchExpr
	}
	if tok.typ != tokenColon {
		return p.parseError(tok, "expected ']' or ':'")
	}

	// parse optional step
	tok = p.getToken()
	if tok.typ == tokenInteger {
		step = tok.Int()
		if step < 0 {
			return p.parseError(tok, "step must be a positive value")
		}
		tok = p.getToken()
	}
	if tok.typ != tokenRightBracket {
		return p.parseError(tok, "expected ']'")
	}

	p.query.appendPath(newMatchSliceFn(start, end, step))
	return p.parseMatchExpr
}

func (p *queryParser) parseFilterExpr() queryParserStateFn {
	tok := p.getToken()
	if tok.typ != tokenLeftParen {
		return p.parseError(tok, "expected left-parenthesis for filter expression")
	}
	tok = p.getToken()
	if tok.typ != tokenKey && tok.typ != tokenString {
		return p.parseError(tok, "expected key or string for filter funciton name")
	}
	name := tok.val
	tok = p.getToken()
	if tok.typ != tokenRightParen {
		return p.parseError(tok, "expected right-parenthesis for filter expression")
	}
	p.union = append(p.union, newMatchFilterFn(name, tok.Position))
	return p.parseUnionExpr
}

func parseQuery(flow chan token) (*Query, error) {
	parser := &queryParser{
		flow:         flow,
		tokensBuffer: []token{},
		query:        newQuery(),
	}
	parser.run()
	return parser.query, parser.err
}
