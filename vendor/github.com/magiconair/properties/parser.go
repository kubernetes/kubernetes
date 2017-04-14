// Copyright 2017 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
	"fmt"
	"runtime"
)

type parser struct {
	lex *lexer
}

func parse(input string) (properties *Properties, err error) {
	p := &parser{lex: lex(input)}
	defer p.recover(&err)

	properties = NewProperties()
	key := ""
	comments := []string{}

	for {
		token := p.expectOneOf(itemComment, itemKey, itemEOF)
		switch token.typ {
		case itemEOF:
			goto done
		case itemComment:
			comments = append(comments, token.val)
			continue
		case itemKey:
			key = token.val
			if _, ok := properties.m[key]; !ok {
				properties.k = append(properties.k, key)
			}
		}

		token = p.expectOneOf(itemValue, itemEOF)
		if len(comments) > 0 {
			properties.c[key] = comments
			comments = []string{}
		}
		switch token.typ {
		case itemEOF:
			properties.m[key] = ""
			goto done
		case itemValue:
			properties.m[key] = token.val
		}
	}

done:
	return properties, nil
}

func (p *parser) errorf(format string, args ...interface{}) {
	format = fmt.Sprintf("properties: Line %d: %s", p.lex.lineNumber(), format)
	panic(fmt.Errorf(format, args...))
}

func (p *parser) expect(expected itemType) (token item) {
	token = p.lex.nextItem()
	if token.typ != expected {
		p.unexpected(token)
	}
	return token
}

func (p *parser) expectOneOf(expected ...itemType) (token item) {
	token = p.lex.nextItem()
	for _, v := range expected {
		if token.typ == v {
			return token
		}
	}
	p.unexpected(token)
	panic("unexpected token")
}

func (p *parser) unexpected(token item) {
	p.errorf(token.String())
}

// recover is the handler that turns panics into returns from the top level of Parse.
func (p *parser) recover(errp *error) {
	e := recover()
	if e != nil {
		if _, ok := e.(runtime.Error); ok {
			panic(e)
		}
		*errp = e.(error)
	}
	return
}
