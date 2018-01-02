package filters

import (
	"fmt"
	"io"
	"strconv"

	"github.com/containerd/containerd/errdefs"
	"github.com/pkg/errors"
)

/*
Parse the strings into a filter that may be used with an adaptor.

The filter is made up of zero or more selectors.

The format is a comma separated list of expressions, in the form of
`<fieldpath><op><value>`, known as selectors. All selectors must match the
target object for the filter to be true.

We define the operators "==" for equality, "!=" for not equal and "~=" for a
regular expression. If the operator and value are not present, the matcher will
test for the presence of a value, as defined by the target object.

The formal grammar is as follows:

selectors := selector ("," selector)*
selector  := fieldpath (operator value)
fieldpath := field ('.' field)*
field     := quoted | [A-Za-z] [A-Za-z0-9_]+
operator  := "==" | "!=" | "~="
value     := quoted | [^\s,]+
quoted    := <go string syntax>

*/
func Parse(s string) (Filter, error) {
	// special case empty to match all
	if s == "" {
		return Always, nil
	}

	p := parser{input: s}
	return p.parse()
}

// ParseAll parses each filter in ss and returns a filter that will return true
// if any filter matches the expression.
//
// If no filters are provided, the filter will match anything.
func ParseAll(ss ...string) (Filter, error) {
	if len(ss) == 0 {
		return Always, nil
	}

	var fs []Filter
	for _, s := range ss {
		f, err := Parse(s)
		if err != nil {
			return nil, errors.Wrapf(errdefs.ErrInvalidArgument, err.Error())
		}

		fs = append(fs, f)
	}

	return Any(fs), nil
}

type parser struct {
	input   string
	scanner scanner
}

func (p *parser) parse() (Filter, error) {
	p.scanner.init(p.input)

	ss, err := p.selectors()
	if err != nil {
		return nil, errors.Wrap(err, "filters")
	}

	return ss, nil
}

func (p *parser) selectors() (Filter, error) {
	s, err := p.selector()
	if err != nil {
		return nil, err
	}

	ss := All{s}

loop:
	for {
		tok := p.scanner.peek()
		switch tok {
		case ',':
			pos, tok, _ := p.scanner.scan()
			if tok != tokenSeparator {
				return nil, p.mkerr(pos, "expected a separator")
			}

			s, err := p.selector()
			if err != nil {
				return nil, err
			}

			ss = append(ss, s)
		case tokenEOF:
			break loop
		default:
			return nil, p.mkerr(p.scanner.ppos, "unexpected input: %v", string(tok))
		}
	}

	return ss, nil
}

func (p *parser) selector() (selector, error) {
	fieldpath, err := p.fieldpath()
	if err != nil {
		return selector{}, err
	}

	switch p.scanner.peek() {
	case ',', tokenSeparator, tokenEOF:
		return selector{
			fieldpath: fieldpath,
			operator:  operatorPresent,
		}, nil
	}

	op, err := p.operator()
	if err != nil {
		return selector{}, err
	}

	value, err := p.value()
	if err != nil {
		if err == io.EOF {
			return selector{}, io.ErrUnexpectedEOF
		}
		return selector{}, err
	}

	return selector{
		fieldpath: fieldpath,
		value:     value,
		operator:  op,
	}, nil
}

func (p *parser) fieldpath() ([]string, error) {
	f, err := p.field()
	if err != nil {
		return nil, err
	}

	fs := []string{f}
loop:
	for {
		tok := p.scanner.peek() // lookahead to consume field separator

		switch tok {
		case '.':
			pos, tok, _ := p.scanner.scan() // consume separator
			if tok != tokenSeparator {
				return nil, p.mkerr(pos, "expected a field separator (`.`)")
			}

			f, err := p.field()
			if err != nil {
				return nil, err
			}

			fs = append(fs, f)
		default:
			// let the layer above handle the other bad cases.
			break loop
		}
	}

	return fs, nil
}

func (p *parser) field() (string, error) {
	pos, tok, s := p.scanner.scan()
	switch tok {
	case tokenField:
		return s, nil
	case tokenQuoted:
		return p.unquote(pos, s)
	}

	return "", p.mkerr(pos, "expected field or quoted")
}

func (p *parser) operator() (operator, error) {
	pos, tok, s := p.scanner.scan()
	switch tok {
	case tokenOperator:
		switch s {
		case "==":
			return operatorEqual, nil
		case "!=":
			return operatorNotEqual, nil
		case "~=":
			return operatorMatches, nil
		default:
			return 0, p.mkerr(pos, "unsupported operator %q", s)
		}
	}

	return 0, p.mkerr(pos, `expected an operator ("=="|"!="|"~=")`)
}

func (p *parser) value() (string, error) {
	pos, tok, s := p.scanner.scan()

	switch tok {
	case tokenValue, tokenField:
		return s, nil
	case tokenQuoted:
		return p.unquote(pos, s)
	}

	return "", p.mkerr(pos, "expected value or quoted")
}

func (p *parser) unquote(pos int, s string) (string, error) {
	uq, err := strconv.Unquote(s)
	if err != nil {
		return "", p.mkerr(pos, "unquoting failed: %v", err)
	}

	return uq, nil
}

type parseError struct {
	input string
	pos   int
	msg   string
}

func (pe parseError) Error() string {
	if pe.pos < len(pe.input) {
		before := pe.input[:pe.pos]
		location := pe.input[pe.pos : pe.pos+1] // need to handle end
		after := pe.input[pe.pos+1:]

		return fmt.Sprintf("[%s >|%s|< %s]: %v", before, location, after, pe.msg)
	}

	return fmt.Sprintf("[%s]: %v", pe.input, pe.msg)
}

func (p *parser) mkerr(pos int, format string, args ...interface{}) error {
	return errors.Wrap(parseError{
		input: p.input,
		pos:   pos,
		msg:   fmt.Sprintf(format, args...),
	}, "parse error")
}
