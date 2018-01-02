package revision

import (
	"bytes"
	"testing"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type ScannerSuite struct{}

var _ = Suite(&ScannerSuite{})

func (s *ScannerSuite) TestReadColon(c *C) {
	scanner := newScanner(bytes.NewBufferString(":"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, ":")
	c.Assert(tok, Equals, colon)
}

func (s *ScannerSuite) TestReadTilde(c *C) {
	scanner := newScanner(bytes.NewBufferString("~"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "~")
	c.Assert(tok, Equals, tilde)
}

func (s *ScannerSuite) TestReadCaret(c *C) {
	scanner := newScanner(bytes.NewBufferString("^"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "^")
	c.Assert(tok, Equals, caret)
}

func (s *ScannerSuite) TestReadDot(c *C) {
	scanner := newScanner(bytes.NewBufferString("."))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, ".")
	c.Assert(tok, Equals, dot)
}

func (s *ScannerSuite) TestReadSlash(c *C) {
	scanner := newScanner(bytes.NewBufferString("/"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "/")
	c.Assert(tok, Equals, slash)
}

func (s *ScannerSuite) TestReadEOF(c *C) {
	scanner := newScanner(bytes.NewBufferString(string(rune(0))))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "")
	c.Assert(tok, Equals, eof)
}

func (s *ScannerSuite) TestReadNumber(c *C) {
	scanner := newScanner(bytes.NewBufferString("1234"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "1234")
	c.Assert(tok, Equals, number)
}

func (s *ScannerSuite) TestReadSpace(c *C) {
	scanner := newScanner(bytes.NewBufferString(" "))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, " ")
	c.Assert(tok, Equals, space)
}

func (s *ScannerSuite) TestReadControl(c *C) {
	scanner := newScanner(bytes.NewBufferString(""))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "\x01")
	c.Assert(tok, Equals, control)
}

func (s *ScannerSuite) TestReadOpenBrace(c *C) {
	scanner := newScanner(bytes.NewBufferString("{"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "{")
	c.Assert(tok, Equals, obrace)
}

func (s *ScannerSuite) TestReadCloseBrace(c *C) {
	scanner := newScanner(bytes.NewBufferString("}"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "}")
	c.Assert(tok, Equals, cbrace)
}

func (s *ScannerSuite) TestReadMinus(c *C) {
	scanner := newScanner(bytes.NewBufferString("-"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "-")
	c.Assert(tok, Equals, minus)
}

func (s *ScannerSuite) TestReadAt(c *C) {
	scanner := newScanner(bytes.NewBufferString("@"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "@")
	c.Assert(tok, Equals, at)
}

func (s *ScannerSuite) TestReadAntislash(c *C) {
	scanner := newScanner(bytes.NewBufferString("\\"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "\\")
	c.Assert(tok, Equals, aslash)
}

func (s *ScannerSuite) TestReadQuestionMark(c *C) {
	scanner := newScanner(bytes.NewBufferString("?"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "?")
	c.Assert(tok, Equals, qmark)
}

func (s *ScannerSuite) TestReadAsterisk(c *C) {
	scanner := newScanner(bytes.NewBufferString("*"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "*")
	c.Assert(tok, Equals, asterisk)
}

func (s *ScannerSuite) TestReadOpenBracket(c *C) {
	scanner := newScanner(bytes.NewBufferString("["))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "[")
	c.Assert(tok, Equals, obracket)
}

func (s *ScannerSuite) TestReadExclamationMark(c *C) {
	scanner := newScanner(bytes.NewBufferString("!"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "!")
	c.Assert(tok, Equals, emark)
}

func (s *ScannerSuite) TestReadWord(c *C) {
	scanner := newScanner(bytes.NewBufferString("abcde"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "abcde")
	c.Assert(tok, Equals, word)
}

func (s *ScannerSuite) TestReadTokenError(c *C) {
	scanner := newScanner(bytes.NewBufferString("`"))
	tok, data, err := scanner.scan()

	c.Assert(err, Equals, nil)
	c.Assert(data, Equals, "`")
	c.Assert(tok, Equals, tokenError)
}
