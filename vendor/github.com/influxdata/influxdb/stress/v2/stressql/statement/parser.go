package statement

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/influxdata/influxdb/stress/v2/statement"
)

// Token represents a lexical token.
type Token int

// The following tokens represent the different values in the AST that make up stressql
const (
	ILLEGAL Token = iota
	EOF

	WS

	literalBeg
	// IDENT and the following are InfluxQL literal tokens.
	IDENT       // main
	NUMBER      // 12345.67
	DURATIONVAL // 13h
	STRING      // "abc"
	BADSTRING   // "abc
	TEMPLATEVAR // %f
	literalEnd

	COMMA    // ,
	LPAREN   // (
	RPAREN   // )
	LBRACKET // [
	RBRACKET // ]
	PIPE     // |
	PERIOD   // .

	keywordBeg
	SET
	USE
	QUERY
	INSERT
	GO
	DO
	WAIT
	STR
	INT
	FLOAT
	EXEC
	keywordEnd
)

var tokens = [...]string{
	ILLEGAL: "ILLEGAL",
	EOF:     "EOF",
	WS:      "WS",

	IDENT:       "IDENT",
	NUMBER:      "NUMBER",
	DURATIONVAL: "DURATION",
	STRING:      "STRING",
	BADSTRING:   "BADSTRING",
	TEMPLATEVAR: "TEMPLATEVAR",

	COMMA:    ",",
	PERIOD:   ".",
	LPAREN:   "(",
	RPAREN:   ")",
	LBRACKET: "[",
	RBRACKET: "]",
	PIPE:     "|",

	SET:    "SET",
	USE:    "USE",
	QUERY:  "QUERY",
	INSERT: "INSERT",
	EXEC:   "EXEC",
	DO:     "DO",
	GO:     "GO",
	WAIT:   "WAIT",
	INT:    "INT",
	FLOAT:  "FLOAT",
	STR:    "STRING",
}

var eof = rune(1)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func isWhitespace(ch rune) bool { return ch == ' ' || ch == '\t' || ch == '\n' }

func isDigit(r rune) bool {
	return r >= '0' && r <= '9'
}

func isLetter(ch rune) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '@')
}

// Scanner scans over the file and converts the raw text into tokens
type Scanner struct {
	r *bufio.Reader
}

// NewScanner returns a Scanner
func NewScanner(r io.Reader) *Scanner {
	return &Scanner{r: bufio.NewReader(r)}
}

func (s *Scanner) read() rune {
	ch, _, err := s.r.ReadRune()
	if err != nil {
		return eof
	}
	return ch
}

func (s *Scanner) unread() { _ = s.r.UnreadRune() }

func (s *Scanner) peek() rune {
	ch := s.read()
	s.unread()
	return ch
}

// Scan moves to the next character in the file and returns a tokenized version as well as the literal
func (s *Scanner) Scan() (tok Token, lit string) {
	ch := s.read()

	if isWhitespace(ch) {
		s.unread()
		return s.scanWhitespace()
	} else if isLetter(ch) {
		s.unread()
		return s.scanIdent()
	} else if isDigit(ch) {
		s.unread()
		return s.scanNumber()
	}

	switch ch {
	case eof:
		return EOF, ""
	case '"':
		s.unread()
		return s.scanIdent()
	case '%':
		s.unread()
		return s.scanTemplateVar()
	case ',':
		return COMMA, ","
	case '.':
		return PERIOD, "."
	case '(':
		return LPAREN, "("
	case ')':
		return RPAREN, ")"
	case '[':
		return LBRACKET, "["
	case ']':
		return RBRACKET, "]"
	case '|':
		return PIPE, "|"
	}

	return ILLEGAL, string(ch)
}

func (s *Scanner) scanWhitespace() (tok Token, lit string) {
	var buf bytes.Buffer
	buf.WriteRune(s.read())

	for {
		if ch := s.read(); ch == eof {
			break
		} else if !isWhitespace(ch) {
			s.unread()
			break
		} else {
			buf.WriteRune(ch)
		}
	}

	return WS, buf.String()
}

func (s *Scanner) scanIdent() (tok Token, lit string) {
	var buf bytes.Buffer
	buf.WriteRune(s.read())

	for {
		if ch := s.read(); ch == eof {
			break
		} else if !isLetter(ch) && !isDigit(ch) && ch != '_' && ch != ':' && ch != '=' && ch != '-' {
			s.unread()
			break
		} else {
			_, _ = buf.WriteRune(ch)
		}
	}

	switch strings.ToUpper(buf.String()) {
	case "SET":
		return SET, buf.String()
	case "USE":
		return USE, buf.String()
	case "QUERY":
		return QUERY, buf.String()
	case "INSERT":
		return INSERT, buf.String()
	case "EXEC":
		return EXEC, buf.String()
	case "WAIT":
		return WAIT, buf.String()
	case "GO":
		return GO, buf.String()
	case "DO":
		return DO, buf.String()
	case "STR":
		return STR, buf.String()
	case "FLOAT":
		return FLOAT, buf.String()
	case "INT":
		return INT, buf.String()
	}

	return IDENT, buf.String()
}

func (s *Scanner) scanTemplateVar() (tok Token, lit string) {
	var buf bytes.Buffer
	buf.WriteRune(s.read())
	buf.WriteRune(s.read())

	return TEMPLATEVAR, buf.String()
}

func (s *Scanner) scanNumber() (tok Token, lit string) {
	var buf bytes.Buffer
	buf.WriteRune(s.read())

	for {
		if ch := s.read(); ch == eof {
			break
		} else if ch == 'n' || ch == 's' || ch == 'm' {
			_, _ = buf.WriteRune(ch)
			return DURATIONVAL, buf.String()
		} else if !isDigit(ch) {
			s.unread()
			break
		} else {
			_, _ = buf.WriteRune(ch)
		}
	}

	return NUMBER, buf.String()
}

/////////////////////////////////
// PARSER ///////////////////////
/////////////////////////////////

// Parser turns the file from raw text into an AST
type Parser struct {
	s   *Scanner
	buf struct {
		tok Token
		lit string
		n   int
	}
}

// NewParser creates a new Parser
func NewParser(r io.Reader) *Parser {
	return &Parser{s: NewScanner(r)}
}

// Parse returns a Statement
func (p *Parser) Parse() (statement.Statement, error) {
	tok, lit := p.scanIgnoreWhitespace()

	switch tok {
	case QUERY:
		p.unscan()
		return p.ParseQueryStatement()
	case INSERT:
		p.unscan()
		return p.ParseInsertStatement()
	case EXEC:
		p.unscan()
		return p.ParseExecStatement()
	case SET:
		p.unscan()
		return p.ParseSetStatement()
	case GO:
		p.unscan()
		return p.ParseGoStatement()
	case WAIT:
		p.unscan()
		return p.ParseWaitStatement()
	}

	return nil, fmt.Errorf("Improper syntax\n  unknown token found between statements, token: %v\n", lit)
}

// ParseQueryStatement returns a QueryStatement
func (p *Parser) ParseQueryStatement() (*statement.QueryStatement, error) {
	stmt := &statement.QueryStatement{
		StatementID: RandStr(10),
	}
	if tok, lit := p.scanIgnoreWhitespace(); tok != QUERY {
		return nil, fmt.Errorf("Error parsing Query Statement\n  Expected: QUERY\n  Found: %v\n", lit)
	}

	tok, lit := p.scanIgnoreWhitespace()
	if tok != IDENT {
		return nil, fmt.Errorf("Error parsing Query Statement\n  Expected: IDENT\n  Found: %v\n", lit)
	}

	stmt.Name = lit

	for {
		tok, lit := p.scan()
		if tok == TEMPLATEVAR {
			stmt.TemplateString += "%v"
			stmt.Args = append(stmt.Args, lit)
		} else if tok == DO {
			tok, lit := p.scanIgnoreWhitespace()
			if tok != NUMBER {
				return nil, fmt.Errorf("Error parsing Query Statement\n  Expected: NUMBER\n  Found: %v\n", lit)
			}
			// Parse out the integer
			i, err := strconv.ParseInt(lit, 10, 64)
			if err != nil {
				log.Fatalf("Error parsing integer in Query Statement:\n  string: %v\n  error: %v\n", lit, err)
			}
			stmt.Count = int(i)
			break
		} else if tok == WS && lit == "\n" {
			continue
		} else {
			stmt.TemplateString += lit
		}
	}

	return stmt, nil

}

// ParseInsertStatement returns a InsertStatement
func (p *Parser) ParseInsertStatement() (*statement.InsertStatement, error) {

	// Initialize the InsertStatement with a statementId
	stmt := &statement.InsertStatement{
		StatementID: RandStr(10),
	}

	// If the first word is INSERT
	if tok, lit := p.scanIgnoreWhitespace(); tok != INSERT {
		return nil, fmt.Errorf("Error parsing Insert Statement\n  Expected: INSERT\n  Found: %v\n", lit)
	}

	// Next should come the NAME of the statement. It is IDENT type
	tok, lit := p.scanIgnoreWhitespace()
	if tok != IDENT {
		return nil, fmt.Errorf("Error parsing Insert Statement\n  Expected: IDENT\n  Found: %v\n", lit)
	}

	// Set the Name
	stmt.Name = lit

	// Next char should be a newline
	tok, lit = p.scan()
	if tok != WS {
		return nil, fmt.Errorf("Error parsing Insert Statement\n  Expected: WS\n  Found: %v\n", lit)
	}

	// We are now scanning the tags line
	var prev Token
	inTags := true

	for {
		// Start for loop by scanning
		tok, lit = p.scan()

		// If scaned is WS then we are just entering tags or leaving tags or fields
		if tok == WS {

			// If previous is COMMA then we are leaving measurement, continue
			if prev == COMMA {
				continue
			}
			// Otherwise we need to add a space to the template string and we are out of tags
			stmt.TemplateString += " "
			inTags = false
		} else if tok == LBRACKET {
			// If we are still inTags and there is a LBRACKET we are adding another template
			if inTags {
				stmt.TagCount++
			}

			// Add a space to fill template string with template result
			stmt.TemplateString += "%v"

			// parse template should return a template type
			expr, err := p.ParseTemplate()

			// If there is a Template parsing error return it
			if err != nil {
				return nil, err
			}

			// Add template to parsed select statement
			stmt.Templates = append(stmt.Templates, expr)

			// A number signifies that we are in the Timestamp section
		} else if tok == NUMBER {
			// Add a space to fill template string with timestamp
			stmt.TemplateString += "%v"
			p.unscan()

			// Parse out the Timestamp
			ts, err := p.ParseTimestamp()

			// If there is a Timestamp parsing error return it
			if err != nil {
				return nil, err
			}

			// Set the Timestamp
			stmt.Timestamp = ts

			// Break loop as InsertStatement ends
			break
		} else if tok != IDENT && tok != COMMA {
			return nil, fmt.Errorf("Error parsing Insert Statement\n  Expected: IDENT or COMMA\n  Found: %v\n", lit)
		} else {
			prev = tok
			stmt.TemplateString += lit
		}

	}

	return stmt, nil
}

// ParseTemplate returns a Template
func (p *Parser) ParseTemplate() (*statement.Template, error) {

	// Blank template
	tmplt := &statement.Template{}

	for {
		// Scan to start loop
		tok, lit := p.scanIgnoreWhitespace()

		// If the tok == IDENT explicit tags are passed. Add them to the list of tags
		if tok == IDENT {
			tmplt.Tags = append(tmplt.Tags, lit)

			// Different flavors of functions
		} else if tok == INT || tok == FLOAT || tok == STR {
			p.unscan()

			// Parse out the function
			fn, err := p.ParseFunction()

			// If there is a Function parsing error return it
			if err != nil {
				return nil, err
			}

			// Set the Function on the Template
			tmplt.Function = fn

			// End of Function
		} else if tok == RBRACKET {
			break
		}
	}

	return tmplt, nil
}

// ParseExecStatement returns a ExecStatement
func (p *Parser) ParseExecStatement() (*statement.ExecStatement, error) {
	// NEEDS TO PARSE ACTUAL PATH TO SCRIPT CURRENTLY ONLY DOES
	// IDENT SCRIPT NAMES

	stmt := &statement.ExecStatement{
		StatementID: RandStr(10),
	}

	if tok, lit := p.scanIgnoreWhitespace(); tok != EXEC {
		return nil, fmt.Errorf("Error parsing Exec Statement\n  Expected: EXEC\n  Found: %v\n", lit)
	}

	tok, lit := p.scanIgnoreWhitespace()
	if tok != IDENT {
		return nil, fmt.Errorf("Error parsing Exec Statement\n  Expected: IDENT\n  Found: %v\n", lit)
	}

	stmt.Script = lit

	return stmt, nil
}

// ParseSetStatement returns a SetStatement
func (p *Parser) ParseSetStatement() (*statement.SetStatement, error) {

	stmt := &statement.SetStatement{
		StatementID: RandStr(10),
	}

	if tok, lit := p.scanIgnoreWhitespace(); tok != SET {
		return nil, fmt.Errorf("Error parsing Set Statement\n  Expected: SET\n  Found: %v\n", lit)
	}

	tok, lit := p.scanIgnoreWhitespace()
	if tok != IDENT {
		return nil, fmt.Errorf("Error parsing Set Statement\n  Expected: IDENT\n  Found: %v\n", lit)
	}

	stmt.Var = lit

	tok, lit = p.scanIgnoreWhitespace()

	if tok != LBRACKET {
		return nil, fmt.Errorf("Error parsing Set Statement\n  Expected: RBRACKET\n  Found: %v\n", lit)
	}

	for {
		tok, lit = p.scanIgnoreWhitespace()
		if tok == RBRACKET {
			break
		} else if lit != "-" && lit != ":" && tok != IDENT && tok != NUMBER && tok != DURATIONVAL && tok != PERIOD && tok != PIPE {
			return nil, fmt.Errorf("Error parsing Set Statement\n  Expected: IDENT || NUMBER || DURATION\n  Found: %v\n", lit)
		}
		stmt.Value += lit
	}

	return stmt, nil
}

// ParseWaitStatement returns a WaitStatement
func (p *Parser) ParseWaitStatement() (*statement.WaitStatement, error) {

	stmt := &statement.WaitStatement{
		StatementID: RandStr(10),
	}

	if tok, lit := p.scanIgnoreWhitespace(); tok != WAIT {
		return nil, fmt.Errorf("Error parsing Wait Statement\n  Expected: WAIT\n  Found: %v\n", lit)
	}

	return stmt, nil
}

// ParseGoStatement returns a GoStatement
func (p *Parser) ParseGoStatement() (*statement.GoStatement, error) {

	stmt := &statement.GoStatement{}
	stmt.StatementID = RandStr(10)

	if tok, lit := p.scanIgnoreWhitespace(); tok != GO {
		return nil, fmt.Errorf("Error parsing Go Statement\n  Expected: GO\n  Found: %v\n", lit)
	}

	var body statement.Statement
	var err error

	tok, _ := p.scanIgnoreWhitespace()
	switch tok {
	case QUERY:
		p.unscan()
		body, err = p.ParseQueryStatement()
	case INSERT:
		p.unscan()
		body, err = p.ParseInsertStatement()
	case EXEC:
		p.unscan()
		body, err = p.ParseExecStatement()
	}

	if err != nil {
		return nil, err
	}

	stmt.Statement = body

	return stmt, nil

}

// ParseFunction returns a Function
func (p *Parser) ParseFunction() (*statement.Function, error) {

	fn := &statement.Function{}

	tok, lit := p.scanIgnoreWhitespace()
	fn.Type = lit

	tok, lit = p.scanIgnoreWhitespace()
	fn.Fn = lit

	tok, lit = p.scanIgnoreWhitespace()
	if tok != LPAREN {
		return nil, fmt.Errorf("Error parsing Insert template function\n  Expected: LPAREN\n  Found: %v\n", lit)
	}

	tok, lit = p.scanIgnoreWhitespace()
	if tok != NUMBER {
		return nil, fmt.Errorf("Error parsing Insert template function\n  Expected: NUMBER\n  Found: %v\n", lit)
	}

	// Parse out the integer
	i, err := strconv.ParseInt(lit, 10, 64)

	if err != nil {
		log.Fatalf("Error parsing integer in Insert template function:\n  string: %v\n  error: %v\n", lit, err)
	}

	fn.Argument = int(i)

	tok, _ = p.scanIgnoreWhitespace()
	if tok != RPAREN {
		return nil, fmt.Errorf("Error parsing Insert template function\n  Expected: RPAREN\n  Found: %v\n", lit)
	}

	tok, lit = p.scanIgnoreWhitespace()
	if tok != NUMBER {
		return nil, fmt.Errorf("Error parsing Insert template function\n  Expected: NUMBER\n  Found: %v\n", lit)
	}

	// Parse out the integer
	i, err = strconv.ParseInt(lit, 10, 64)

	if err != nil {
		log.Fatalf("Error parsing integer in Insert template function:\n  string: %v\n  error: %v\n", lit, err)
	}

	fn.Count = int(i)

	return fn, nil
}

// ParseTimestamp returns a Timestamp
func (p *Parser) ParseTimestamp() (*statement.Timestamp, error) {

	ts := &statement.Timestamp{}

	tok, lit := p.scanIgnoreWhitespace()
	if tok != NUMBER {
		return nil, fmt.Errorf("Error parsing Insert timestamp\n  Expected: NUMBER\n  Found: %v\n", lit)
	}

	// Parse out the integer
	i, err := strconv.ParseInt(lit, 10, 64)

	if err != nil {
		log.Fatalf("Error parsing integer in Insert timestamp:\n  string: %v\n  error: %v\n", lit, err)
	}

	ts.Count = int(i)

	tok, lit = p.scanIgnoreWhitespace()
	if tok != DURATIONVAL {
		return nil, fmt.Errorf("Error parsing Insert timestamp\n  Expected: DURATION\n  Found: %v\n", lit)
	}

	// Parse out the duration
	dur, err := time.ParseDuration(lit)

	if err != nil {
		log.Fatalf("Error parsing duration in Insert timestamp:\n  string: %v\n  error: %v\n", lit, err)
	}

	ts.Duration = dur

	return ts, nil
}

func (p *Parser) scan() (tok Token, lit string) {
	// If we have a token on the buffer, then return it.
	if p.buf.n != 0 {
		p.buf.n = 0
		return p.buf.tok, p.buf.lit
	}

	// Otherwise read the next token from the scanner.
	tok, lit = p.s.Scan()

	// Save it to the buffer in case we unscan later.
	p.buf.tok, p.buf.lit = tok, lit

	return
}

// scanIgnoreWhitespace scans the next non-whitespace token.
func (p *Parser) scanIgnoreWhitespace() (tok Token, lit string) {
	tok, lit = p.scan()
	if tok == WS {
		tok, lit = p.scan()
	}
	return
}

// unscan pushes the previously read token back onto the buffer.
func (p *Parser) unscan() { p.buf.n = 1 }

// RandStr returns a string of random characters with length n
func RandStr(n int) string {
	b := make([]byte, n/2)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x", b)
}
