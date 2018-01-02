package stressql

import (
	"bufio"
	"bytes"
	"io"
	"log"
	"os"
	"strings"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/stress/v2/statement"
	stressql "github.com/influxdata/influxdb/stress/v2/stressql/statement"
)

// Token represents a lexical token.
type Token int

// These are the lexical tokens used by the file parser
const (
	ILLEGAL Token = iota
	EOF
	STATEMENT
	BREAK
)

var eof = rune(0)

func check(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

func isNewline(r rune) bool {
	return r == '\n'
}

// Scanner scans the file and tokenizes the raw text
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

// Scan moves the Scanner forward one character
func (s *Scanner) Scan() (tok Token, lit string) {
	ch := s.read()

	if isNewline(ch) {
		s.unread()
		return s.scanNewlines()
	} else if ch == eof {
		return EOF, ""
	} else {
		s.unread()
		return s.scanStatements()
	}
	// golint marks as unreachable code
	// return ILLEGAL, string(ch)
}

func (s *Scanner) scanNewlines() (tok Token, lit string) {
	var buf bytes.Buffer
	buf.WriteRune(s.read())

	for {
		if ch := s.read(); ch == eof {
			break
		} else if !isNewline(ch) {
			s.unread()
			break
		} else {
			buf.WriteRune(ch)
		}
	}

	return BREAK, buf.String()
}

func (s *Scanner) scanStatements() (tok Token, lit string) {
	var buf bytes.Buffer
	buf.WriteRune(s.read())

	for {
		if ch := s.read(); ch == eof {
			break
		} else if isNewline(ch) && isNewline(s.peek()) {
			s.unread()
			break
		} else if isNewline(ch) {
			s.unread()
			buf.WriteRune(ch)
		} else {
			buf.WriteRune(ch)
		}
	}

	return STATEMENT, buf.String()
}

// ParseStatements takes a configFile and returns a slice of Statements
func ParseStatements(file string) ([]statement.Statement, error) {
	seq := []statement.Statement{}

	f, err := os.Open(file)
	check(err)

	s := NewScanner(f)

	for {
		t, l := s.Scan()

		if t == EOF {
			break
		}
		_, err := influxql.ParseStatement(l)
		if err == nil {

			seq = append(seq, &statement.InfluxqlStatement{Query: l, StatementID: stressql.RandStr(10)})
		} else if t == BREAK {
			continue
		} else {
			f := strings.NewReader(l)
			p := stressql.NewParser(f)
			s, err := p.Parse()
			if err != nil {
				return nil, err
			}
			seq = append(seq, s)

		}
	}

	f.Close()

	return seq, nil

}
