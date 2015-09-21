package unit

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode"
)

// Deserialize parses a systemd unit file into a list of UnitOption objects.
func Deserialize(f io.Reader) (opts []*UnitOption, err error) {
	lexer, optchan, errchan := newLexer(f)
	go lexer.lex()

	for opt := range optchan {
		opts = append(opts, &(*opt))
	}

	err = <-errchan
	return opts, err
}

func newLexer(f io.Reader) (*lexer, <-chan *UnitOption, <-chan error) {
	optchan := make(chan *UnitOption)
	errchan := make(chan error, 1)
	buf := bufio.NewReader(f)

	return &lexer{buf, optchan, errchan, ""}, optchan, errchan
}

type lexer struct {
	buf     *bufio.Reader
	optchan chan *UnitOption
	errchan chan error
	section string
}

func (l *lexer) lex() {
	var err error
	next := l.lexNextSection
	for next != nil {
		next, err = next()
		if err != nil {
			l.errchan <- err
			break
		}
	}

	close(l.optchan)
	close(l.errchan)
}

type lexStep func() (lexStep, error)

func (l *lexer) lexSectionName() (lexStep, error) {
	sec, err := l.buf.ReadBytes(']')
	if err != nil {
		return nil, errors.New("unable to find end of section")
	}

	return l.lexSectionSuffixFunc(string(sec[:len(sec)-1])), nil
}

func (l *lexer) lexSectionSuffixFunc(section string) lexStep {
	return func() (lexStep, error) {
		garbage, err := l.toEOL()
		if err != nil {
			return nil, err
		}

		garbage = bytes.TrimSpace(garbage)
		if len(garbage) > 0 {
			return nil, fmt.Errorf("found garbage after section name %s: %v", l.section, garbage)
		}

		return l.lexNextSectionOrOptionFunc(section), nil
	}
}

func (l *lexer) ignoreLineFunc(next lexStep) lexStep {
	return func() (lexStep, error) {
		for {
			line, err := l.toEOL()
			if err != nil {
				return nil, err
			}

			line = bytes.TrimSuffix(line, []byte{' '})

			// lack of continuation means this line has been exhausted
			if !bytes.HasSuffix(line, []byte{'\\'}) {
				break
			}
		}

		// reached end of buffer, safe to exit
		return next, nil
	}
}

func (l *lexer) lexNextSection() (lexStep, error) {
	r, _, err := l.buf.ReadRune()
	if err != nil {
		if err == io.EOF {
			err = nil
		}
		return nil, err
	}

	if r == '[' {
		return l.lexSectionName, nil
	} else if isComment(r) {
		return l.ignoreLineFunc(l.lexNextSection), nil
	}

	return l.lexNextSection, nil
}

func (l *lexer) lexNextSectionOrOptionFunc(section string) lexStep {
	return func() (lexStep, error) {
		r, _, err := l.buf.ReadRune()
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return nil, err
		}

		if unicode.IsSpace(r) {
			return l.lexNextSectionOrOptionFunc(section), nil
		} else if r == '[' {
			return l.lexSectionName, nil
		} else if isComment(r) {
			return l.ignoreLineFunc(l.lexNextSectionOrOptionFunc(section)), nil
		}

		l.buf.UnreadRune()
		return l.lexOptionNameFunc(section), nil
	}
}

func (l *lexer) lexOptionNameFunc(section string) lexStep {
	return func() (lexStep, error) {
		var partial bytes.Buffer
		for {
			r, _, err := l.buf.ReadRune()
			if err != nil {
				return nil, err
			}

			if r == '\n' || r == '\r' {
				return nil, errors.New("unexpected newline encountered while parsing option name")
			}

			if r == '=' {
				break
			}

			partial.WriteRune(r)
		}

		name := strings.TrimSpace(partial.String())
		return l.lexOptionValueFunc(section, name), nil
	}
}

func (l *lexer) lexOptionValueFunc(section, name string) lexStep {
	return func() (lexStep, error) {
		var partial bytes.Buffer

		for {
			line, err := l.toEOL()
			if err != nil {
				return nil, err
			}

			// lack of continuation means this value has been exhausted
			idx := bytes.LastIndex(line, []byte{'\\'})
			if idx == -1 || idx != (len(line)-1) {
				partial.Write(line)
				break
			}

			partial.Write(line[0:idx])
			partial.WriteRune(' ')
		}

		val := strings.TrimSpace(partial.String())
		l.optchan <- &UnitOption{Section: section, Name: name, Value: val}

		return l.lexNextSectionOrOptionFunc(section), nil
	}
}

func (l *lexer) toEOL() ([]byte, error) {
	line, err := l.buf.ReadBytes('\n')
	// ignore EOF here since it's roughly equivalent to EOL
	if err != nil && err != io.EOF {
		return nil, err
	}

	line = bytes.TrimSuffix(line, []byte{'\r'})
	line = bytes.TrimSuffix(line, []byte{'\n'})

	return line, nil
}

func isComment(r rune) bool {
	return r == '#' || r == ';'
}
