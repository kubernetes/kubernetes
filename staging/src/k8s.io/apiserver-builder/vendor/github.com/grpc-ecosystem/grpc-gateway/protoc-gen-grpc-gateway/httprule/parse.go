package httprule

import (
	"fmt"
	"strings"

	"github.com/golang/glog"
)

// InvalidTemplateError indicates that the path template is not valid.
type InvalidTemplateError struct {
	tmpl string
	msg  string
}

func (e InvalidTemplateError) Error() string {
	return fmt.Sprintf("%s: %s", e.msg, e.tmpl)
}

// Parse parses the string representation of path template
func Parse(tmpl string) (Compiler, error) {
	if !strings.HasPrefix(tmpl, "/") {
		return template{}, InvalidTemplateError{tmpl: tmpl, msg: "no leading /"}
	}
	tokens, verb := tokenize(tmpl[1:])

	p := parser{tokens: tokens}
	segs, err := p.topLevelSegments()
	if err != nil {
		return template{}, InvalidTemplateError{tmpl: tmpl, msg: err.Error()}
	}

	return template{
		segments: segs,
		verb:     verb,
		template: tmpl,
	}, nil
}

func tokenize(path string) (tokens []string, verb string) {
	if path == "" {
		return []string{eof}, ""
	}

	const (
		init = iota
		field
		nested
	)
	var (
		st = init
	)
	for path != "" {
		var idx int
		switch st {
		case init:
			idx = strings.IndexAny(path, "/{")
		case field:
			idx = strings.IndexAny(path, ".=}")
		case nested:
			idx = strings.IndexAny(path, "/}")
		}
		if idx < 0 {
			tokens = append(tokens, path)
			break
		}
		switch r := path[idx]; r {
		case '/', '.':
		case '{':
			st = field
		case '=':
			st = nested
		case '}':
			st = init
		}
		if idx == 0 {
			tokens = append(tokens, path[idx:idx+1])
		} else {
			tokens = append(tokens, path[:idx], path[idx:idx+1])
		}
		path = path[idx+1:]
	}

	l := len(tokens)
	t := tokens[l-1]
	if idx := strings.LastIndex(t, ":"); idx == 0 {
		tokens, verb = tokens[:l-1], t[1:]
	} else if idx > 0 {
		tokens[l-1], verb = t[:idx], t[idx+1:]
	}
	tokens = append(tokens, eof)
	return tokens, verb
}

// parser is a parser of the template syntax defined in third_party/googleapis/google/api/httprule.proto.
type parser struct {
	tokens   []string
	accepted []string
}

// topLevelSegments is the target of this parser.
func (p *parser) topLevelSegments() ([]segment, error) {
	glog.V(1).Infof("Parsing %q", p.tokens)
	segs, err := p.segments()
	if err != nil {
		return nil, err
	}
	glog.V(2).Infof("accept segments: %q; %q", p.accepted, p.tokens)
	if _, err := p.accept(typeEOF); err != nil {
		return nil, fmt.Errorf("unexpected token %q after segments %q", p.tokens[0], strings.Join(p.accepted, ""))
	}
	glog.V(2).Infof("accept eof: %q; %q", p.accepted, p.tokens)
	return segs, nil
}

func (p *parser) segments() ([]segment, error) {
	s, err := p.segment()
	if err != nil {
		return nil, err
	}
	glog.V(2).Infof("accept segment: %q; %q", p.accepted, p.tokens)

	segs := []segment{s}
	for {
		if _, err := p.accept("/"); err != nil {
			return segs, nil
		}
		s, err := p.segment()
		if err != nil {
			return segs, err
		}
		segs = append(segs, s)
		glog.V(2).Infof("accept segment: %q; %q", p.accepted, p.tokens)
	}
}

func (p *parser) segment() (segment, error) {
	if _, err := p.accept("*"); err == nil {
		return wildcard{}, nil
	}
	if _, err := p.accept("**"); err == nil {
		return deepWildcard{}, nil
	}
	if l, err := p.literal(); err == nil {
		return l, nil
	}

	v, err := p.variable()
	if err != nil {
		return nil, fmt.Errorf("segment neither wildcards, literal or variable: %v", err)
	}
	return v, err
}

func (p *parser) literal() (segment, error) {
	lit, err := p.accept(typeLiteral)
	if err != nil {
		return nil, err
	}
	return literal(lit), nil
}

func (p *parser) variable() (segment, error) {
	if _, err := p.accept("{"); err != nil {
		return nil, err
	}

	path, err := p.fieldPath()
	if err != nil {
		return nil, err
	}

	var segs []segment
	if _, err := p.accept("="); err == nil {
		segs, err = p.segments()
		if err != nil {
			return nil, fmt.Errorf("invalid segment in variable %q: %v", path, err)
		}
	} else {
		segs = []segment{wildcard{}}
	}

	if _, err := p.accept("}"); err != nil {
		return nil, fmt.Errorf("unterminated variable segment: %s", path)
	}
	return variable{
		path:     path,
		segments: segs,
	}, nil
}

func (p *parser) fieldPath() (string, error) {
	c, err := p.accept(typeIdent)
	if err != nil {
		return "", err
	}
	components := []string{c}
	for {
		if _, err = p.accept("."); err != nil {
			return strings.Join(components, "."), nil
		}
		c, err := p.accept(typeIdent)
		if err != nil {
			return "", fmt.Errorf("invalid field path component: %v", err)
		}
		components = append(components, c)
	}
}

// A termType is a type of terminal symbols.
type termType string

// These constants define some of valid values of termType.
// They improve readability of parse functions.
//
// You can also use "/", "*", "**", "." or "=" as valid values.
const (
	typeIdent   = termType("ident")
	typeLiteral = termType("literal")
	typeEOF     = termType("$")
)

const (
	// eof is the terminal symbol which always appears at the end of token sequence.
	eof = "\u0000"
)

// accept tries to accept a token in "p".
// This function consumes a token and returns it if it matches to the specified "term".
// If it doesn't match, the function does not consume any tokens and return an error.
func (p *parser) accept(term termType) (string, error) {
	t := p.tokens[0]
	switch term {
	case "/", "*", "**", ".", "=", "{", "}":
		if t != string(term) {
			return "", fmt.Errorf("expected %q but got %q", term, t)
		}
	case typeEOF:
		if t != eof {
			return "", fmt.Errorf("expected EOF but got %q", t)
		}
	case typeIdent:
		if err := expectIdent(t); err != nil {
			return "", err
		}
	case typeLiteral:
		if err := expectPChars(t); err != nil {
			return "", err
		}
	default:
		return "", fmt.Errorf("unknown termType %q", term)
	}
	p.tokens = p.tokens[1:]
	p.accepted = append(p.accepted, t)
	return t, nil
}

// expectPChars determines if "t" consists of only pchars defined in RFC3986.
//
// https://www.ietf.org/rfc/rfc3986.txt, P.49
//   pchar         = unreserved / pct-encoded / sub-delims / ":" / "@"
//   unreserved    = ALPHA / DIGIT / "-" / "." / "_" / "~"
//   sub-delims    = "!" / "$" / "&" / "'" / "(" / ")"
//                 / "*" / "+" / "," / ";" / "="
//   pct-encoded   = "%" HEXDIG HEXDIG
func expectPChars(t string) error {
	const (
		init = iota
		pct1
		pct2
	)
	st := init
	for _, r := range t {
		if st != init {
			if !isHexDigit(r) {
				return fmt.Errorf("invalid hexdigit: %c(%U)", r, r)
			}
			switch st {
			case pct1:
				st = pct2
			case pct2:
				st = init
			}
			continue
		}

		// unreserved
		switch {
		case 'A' <= r && r <= 'Z':
			continue
		case 'a' <= r && r <= 'z':
			continue
		case '0' <= r && r <= '9':
			continue
		}
		switch r {
		case '-', '.', '_', '~':
			// unreserved
		case '!', '$', '&', '\'', '(', ')', '*', '+', ',', ';', '=':
			// sub-delims
		case ':', '@':
			// rest of pchar
		case '%':
			// pct-encoded
			st = pct1
		default:
			return fmt.Errorf("invalid character in path segment: %q(%U)", r, r)
		}
	}
	if st != init {
		return fmt.Errorf("invalid percent-encoding in %q", t)
	}
	return nil
}

// expectIdent determines if "ident" is a valid identifier in .proto schema ([[:alpha:]_][[:alphanum:]_]*).
func expectIdent(ident string) error {
	if ident == "" {
		return fmt.Errorf("empty identifier")
	}
	for pos, r := range ident {
		switch {
		case '0' <= r && r <= '9':
			if pos == 0 {
				return fmt.Errorf("identifier starting with digit: %s", ident)
			}
			continue
		case 'A' <= r && r <= 'Z':
			continue
		case 'a' <= r && r <= 'z':
			continue
		case r == '_':
			continue
		default:
			return fmt.Errorf("invalid character %q(%U) in identifier: %s", r, r, ident)
		}
	}
	return nil
}

func isHexDigit(r rune) bool {
	switch {
	case '0' <= r && r <= '9':
		return true
	case 'A' <= r && r <= 'F':
		return true
	case 'a' <= r && r <= 'f':
		return true
	}
	return false
}
