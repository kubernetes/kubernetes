// Package revision extracts git revision from string
// More information about revision : https://www.kernel.org/pub/software/scm/git/docs/gitrevisions.html
package revision

import (
	"bytes"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"time"
)

// ErrInvalidRevision is emitted if string doesn't match valid revision
type ErrInvalidRevision struct {
	s string
}

func (e *ErrInvalidRevision) Error() string {
	return "Revision invalid : " + e.s
}

// Revisioner represents a revision component.
// A revision is made of multiple revision components
// obtained after parsing a revision string,
// for instance revision "master~" will be converted in
// two revision components Ref and TildePath
type Revisioner interface {
}

// Ref represents a reference name : HEAD, master, <hash>
type Ref string

// TildePath represents ~, ~{n}
type TildePath struct {
	Depth int
}

// CaretPath represents ^, ^{n}
type CaretPath struct {
	Depth int
}

// CaretReg represents ^{/foo bar}
type CaretReg struct {
	Regexp *regexp.Regexp
	Negate bool
}

// CaretType represents ^{commit}
type CaretType struct {
	ObjectType string
}

// AtReflog represents @{n}
type AtReflog struct {
	Depth int
}

// AtCheckout represents @{-n}
type AtCheckout struct {
	Depth int
}

// AtUpstream represents @{upstream}, @{u}
type AtUpstream struct {
	BranchName string
}

// AtPush represents @{push}
type AtPush struct {
	BranchName string
}

// AtDate represents @{"2006-01-02T15:04:05Z"}
type AtDate struct {
	Date time.Time
}

// ColonReg represents :/foo bar
type ColonReg struct {
	Regexp *regexp.Regexp
	Negate bool
}

// ColonPath represents :./<path> :<path>
type ColonPath struct {
	Path string
}

// ColonStagePath represents :<n>:/<path>
type ColonStagePath struct {
	Path  string
	Stage int
}

// Parser represents a parser
// use to tokenize and transform to revisioner chunks
// a given string
type Parser struct {
	s                 *scanner
	currentParsedChar struct {
		tok token
		lit string
	}
	unreadLastChar bool
}

// NewParserFromString returns a new instance of parser from a string.
func NewParserFromString(s string) *Parser {
	return NewParser(bytes.NewBufferString(s))
}

// NewParser returns a new instance of parser.
func NewParser(r io.Reader) *Parser {
	return &Parser{s: newScanner(r)}
}

// scan returns the next token from the underlying scanner
// or the last scanned token if an unscan was requested
func (p *Parser) scan() (token, string, error) {
	if p.unreadLastChar {
		p.unreadLastChar = false
		return p.currentParsedChar.tok, p.currentParsedChar.lit, nil
	}

	tok, lit, err := p.s.scan()

	p.currentParsedChar.tok, p.currentParsedChar.lit = tok, lit

	return tok, lit, err
}

// unscan pushes the previously read token back onto the buffer.
func (p *Parser) unscan() { p.unreadLastChar = true }

// Parse explode a revision string into revisioner chunks
func (p *Parser) Parse() ([]Revisioner, error) {
	var rev Revisioner
	var revs []Revisioner
	var tok token
	var err error

	for {
		tok, _, err = p.scan()

		if err != nil {
			return nil, err
		}

		switch tok {
		case at:
			rev, err = p.parseAt()
		case tilde:
			rev, err = p.parseTilde()
		case caret:
			rev, err = p.parseCaret()
		case colon:
			rev, err = p.parseColon()
		case eof:
			err = p.validateFullRevision(&revs)

			if err != nil {
				return []Revisioner{}, err
			}

			return revs, nil
		default:
			p.unscan()
			rev, err = p.parseRef()
		}

		if err != nil {
			return []Revisioner{}, err
		}

		revs = append(revs, rev)
	}
}

// validateFullRevision ensures all revisioner chunks make a valid revision
func (p *Parser) validateFullRevision(chunks *[]Revisioner) error {
	var hasReference bool

	for i, chunk := range *chunks {
		switch chunk.(type) {
		case Ref:
			if i == 0 {
				hasReference = true
			} else {
				return &ErrInvalidRevision{`reference must be defined once at the beginning`}
			}
		case AtDate:
			if len(*chunks) == 1 || hasReference && len(*chunks) == 2 {
				return nil
			}

			return &ErrInvalidRevision{`"@" statement is not valid, could be : <refname>@{<ISO-8601 date>}, @{<ISO-8601 date>}`}
		case AtReflog:
			if len(*chunks) == 1 || hasReference && len(*chunks) == 2 {
				return nil
			}

			return &ErrInvalidRevision{`"@" statement is not valid, could be : <refname>@{<n>}, @{<n>}`}
		case AtCheckout:
			if len(*chunks) == 1 {
				return nil
			}

			return &ErrInvalidRevision{`"@" statement is not valid, could be : @{-<n>}`}
		case AtUpstream:
			if len(*chunks) == 1 || hasReference && len(*chunks) == 2 {
				return nil
			}

			return &ErrInvalidRevision{`"@" statement is not valid, could be : <refname>@{upstream}, @{upstream}, <refname>@{u}, @{u}`}
		case AtPush:
			if len(*chunks) == 1 || hasReference && len(*chunks) == 2 {
				return nil
			}

			return &ErrInvalidRevision{`"@" statement is not valid, could be : <refname>@{push}, @{push}`}
		case TildePath, CaretPath, CaretReg:
			if !hasReference {
				return &ErrInvalidRevision{`"~" or "^" statement must have a reference defined at the beginning`}
			}
		case ColonReg:
			if len(*chunks) == 1 {
				return nil
			}

			return &ErrInvalidRevision{`":" statement is not valid, could be : :/<regexp>`}
		case ColonPath:
			if i == len(*chunks)-1 && hasReference || len(*chunks) == 1 {
				return nil
			}

			return &ErrInvalidRevision{`":" statement is not valid, could be : <revision>:<path>`}
		case ColonStagePath:
			if len(*chunks) == 1 {
				return nil
			}

			return &ErrInvalidRevision{`":" statement is not valid, could be : :<n>:<path>`}
		}
	}

	return nil
}

// parseAt extract @ statements
func (p *Parser) parseAt() (Revisioner, error) {
	var tok, nextTok token
	var lit, nextLit string
	var err error

	tok, _, err = p.scan()

	if err != nil {
		return nil, err
	}

	if tok != obrace {
		p.unscan()

		return Ref("HEAD"), nil
	}

	tok, lit, err = p.scan()

	if err != nil {
		return nil, err
	}

	nextTok, nextLit, err = p.scan()

	if err != nil {
		return nil, err
	}

	switch {
	case tok == word && (lit == "u" || lit == "upstream") && nextTok == cbrace:
		return AtUpstream{}, nil
	case tok == word && lit == "push" && nextTok == cbrace:
		return AtPush{}, nil
	case tok == number && nextTok == cbrace:
		n, _ := strconv.Atoi(lit)

		return AtReflog{n}, nil
	case tok == minus && nextTok == number:
		n, _ := strconv.Atoi(nextLit)

		t, _, err := p.scan()

		if err != nil {
			return nil, err
		}

		if t != cbrace {
			return nil, &ErrInvalidRevision{s: `missing "}" in @{-n} structure`}
		}

		return AtCheckout{n}, nil
	default:
		p.unscan()

		date := lit

		for {
			tok, lit, err = p.scan()

			if err != nil {
				return nil, err
			}

			switch {
			case tok == cbrace:
				t, err := time.Parse("2006-01-02T15:04:05Z", date)

				if err != nil {
					return nil, &ErrInvalidRevision{fmt.Sprintf(`wrong date "%s" must fit ISO-8601 format : 2006-01-02T15:04:05Z`, date)}
				}

				return AtDate{t}, nil
			default:
				date += lit
			}
		}
	}
}

// parseTilde extract ~ statements
func (p *Parser) parseTilde() (Revisioner, error) {
	var tok token
	var lit string
	var err error

	tok, lit, err = p.scan()

	if err != nil {
		return nil, err
	}

	switch {
	case tok == number:
		n, _ := strconv.Atoi(lit)

		return TildePath{n}, nil
	default:
		p.unscan()
		return TildePath{1}, nil
	}
}

// parseCaret extract ^ statements
func (p *Parser) parseCaret() (Revisioner, error) {
	var tok token
	var lit string
	var err error

	tok, lit, err = p.scan()

	if err != nil {
		return nil, err
	}

	switch {
	case tok == obrace:
		r, err := p.parseCaretBraces()

		if err != nil {
			return nil, err
		}

		return r, nil
	case tok == number:
		n, _ := strconv.Atoi(lit)

		if n > 2 {
			return nil, &ErrInvalidRevision{fmt.Sprintf(`"%s" found must be 0, 1 or 2 after "^"`, lit)}
		}

		return CaretPath{n}, nil
	default:
		p.unscan()
		return CaretPath{1}, nil
	}
}

// parseCaretBraces extract ^{<data>} statements
func (p *Parser) parseCaretBraces() (Revisioner, error) {
	var tok, nextTok token
	var lit, _ string
	start := true
	var re string
	var negate bool
	var err error

	for {
		tok, lit, err = p.scan()

		if err != nil {
			return nil, err
		}

		nextTok, _, err = p.scan()

		if err != nil {
			return nil, err
		}

		switch {
		case tok == word && nextTok == cbrace && (lit == "commit" || lit == "tree" || lit == "blob" || lit == "tag" || lit == "object"):
			return CaretType{lit}, nil
		case re == "" && tok == cbrace:
			return CaretType{"tag"}, nil
		case re == "" && tok == emark && nextTok == emark:
			re += lit
		case re == "" && tok == emark && nextTok == minus:
			negate = true
		case re == "" && tok == emark:
			return nil, &ErrInvalidRevision{s: `revision suffix brace component sequences starting with "/!" others than those defined are reserved`}
		case re == "" && tok == slash:
			p.unscan()
		case tok != slash && start:
			return nil, &ErrInvalidRevision{fmt.Sprintf(`"%s" is not a valid revision suffix brace component`, lit)}
		case tok != cbrace:
			p.unscan()
			re += lit
		case tok == cbrace:
			p.unscan()

			reg, err := regexp.Compile(re)

			if err != nil {
				return CaretReg{}, &ErrInvalidRevision{fmt.Sprintf(`revision suffix brace component, %s`, err.Error())}
			}

			return CaretReg{reg, negate}, nil
		}

		start = false
	}
}

// parseColon extract : statements
func (p *Parser) parseColon() (Revisioner, error) {
	var tok token
	var err error

	tok, _, err = p.scan()

	if err != nil {
		return nil, err
	}

	switch tok {
	case slash:
		return p.parseColonSlash()
	default:
		p.unscan()
		return p.parseColonDefault()
	}
}

// parseColonSlash extract :/<data> statements
func (p *Parser) parseColonSlash() (Revisioner, error) {
	var tok, nextTok token
	var lit string
	var re string
	var negate bool
	var err error

	for {
		tok, lit, err = p.scan()

		if err != nil {
			return nil, err
		}

		nextTok, _, err = p.scan()

		if err != nil {
			return nil, err
		}

		switch {
		case tok == emark && nextTok == emark:
			re += lit
		case re == "" && tok == emark && nextTok == minus:
			negate = true
		case re == "" && tok == emark:
			return nil, &ErrInvalidRevision{s: `revision suffix brace component sequences starting with "/!" others than those defined are reserved`}
		case tok == eof:
			p.unscan()
			reg, err := regexp.Compile(re)

			if err != nil {
				return ColonReg{}, &ErrInvalidRevision{fmt.Sprintf(`revision suffix brace component, %s`, err.Error())}
			}

			return ColonReg{reg, negate}, nil
		default:
			p.unscan()
			re += lit
		}
	}
}

// parseColonDefault extract :<data> statements
func (p *Parser) parseColonDefault() (Revisioner, error) {
	var tok token
	var lit string
	var path string
	var stage int
	var err error
	var n = -1

	tok, lit, err = p.scan()

	if err != nil {
		return nil, err
	}

	nextTok, _, err := p.scan()

	if err != nil {
		return nil, err
	}

	if tok == number && nextTok == colon {
		n, _ = strconv.Atoi(lit)
	}

	switch n {
	case 0, 1, 2, 3:
		stage = n
	default:
		path += lit
		p.unscan()
	}

	for {
		tok, lit, err = p.scan()

		if err != nil {
			return nil, err
		}

		switch {
		case tok == eof && n == -1:
			return ColonPath{path}, nil
		case tok == eof:
			return ColonStagePath{path, stage}, nil
		default:
			path += lit
		}
	}
}

// parseRef extract reference name
func (p *Parser) parseRef() (Revisioner, error) {
	var tok, prevTok token
	var lit, buf string
	var endOfRef bool
	var err error

	for {
		tok, lit, err = p.scan()

		if err != nil {
			return nil, err
		}

		switch tok {
		case eof, at, colon, tilde, caret:
			endOfRef = true
		}

		err := p.checkRefFormat(tok, lit, prevTok, buf, endOfRef)

		if err != nil {
			return "", err
		}

		if endOfRef {
			p.unscan()
			return Ref(buf), nil
		}

		buf += lit
		prevTok = tok
	}
}

// checkRefFormat ensure reference name follow rules defined here :
// https://git-scm.com/docs/git-check-ref-format
func (p *Parser) checkRefFormat(token token, literal string, previousToken token, buffer string, endOfRef bool) error {
	switch token {
	case aslash, space, control, qmark, asterisk, obracket:
		return &ErrInvalidRevision{fmt.Sprintf(`must not contains "%s"`, literal)}
	}

	switch {
	case (token == dot || token == slash) && buffer == "":
		return &ErrInvalidRevision{fmt.Sprintf(`must not start with "%s"`, literal)}
	case previousToken == slash && endOfRef:
		return &ErrInvalidRevision{`must not end with "/"`}
	case previousToken == dot && endOfRef:
		return &ErrInvalidRevision{`must not end with "."`}
	case token == dot && previousToken == slash:
		return &ErrInvalidRevision{`must not contains "/."`}
	case previousToken == dot && token == dot:
		return &ErrInvalidRevision{`must not contains ".."`}
	case previousToken == slash && token == slash:
		return &ErrInvalidRevision{`must not contains consecutively "/"`}
	case (token == slash || endOfRef) && len(buffer) > 4 && buffer[len(buffer)-5:] == ".lock":
		return &ErrInvalidRevision{"cannot end with .lock"}
	}

	return nil
}
