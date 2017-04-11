// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

const (
	// SYSTEMD_LINE_MAX mimics the maximum line length that systemd can use.
	// On typical systemd platforms (i.e. modern Linux), this will most
	// commonly be 2048, so let's use that as a sanity check.
	// Technically, we should probably pull this at runtime:
	//    SYSTEMD_LINE_MAX = int(C.sysconf(C.__SC_LINE_MAX))
	// but this would introduce an (unfortunate) dependency on cgo
	SYSTEMD_LINE_MAX = 2048

	// characters that systemd considers indicate a newline
	SYSTEMD_NEWLINE = "\r\n"
)

var (
	ErrLineTooLong = fmt.Errorf("line too long (max %d bytes)", SYSTEMD_LINE_MAX)
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
	defer func() {
		close(l.optchan)
		close(l.errchan)
	}()
	next := l.lexNextSection
	for next != nil {
		if l.buf.Buffered() >= SYSTEMD_LINE_MAX {
			// systemd truncates lines longer than LINE_MAX
			// https://bugs.freedesktop.org/show_bug.cgi?id=85308
			// Rather than allowing this to pass silently, let's
			// explicitly gate people from encountering this
			line, err := l.buf.Peek(SYSTEMD_LINE_MAX)
			if err != nil {
				l.errchan <- err
				return
			}
			if bytes.IndexAny(line, SYSTEMD_NEWLINE) == -1 {
				l.errchan <- ErrLineTooLong
				return
			}
		}

		next, err = next()
		if err != nil {
			l.errchan <- err
			return
		}
	}
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
		garbage, _, err := l.toEOL()
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
			line, _, err := l.toEOL()
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
		return l.lexOptionValueFunc(section, name, bytes.Buffer{}), nil
	}
}

func (l *lexer) lexOptionValueFunc(section, name string, partial bytes.Buffer) lexStep {
	return func() (lexStep, error) {
		for {
			line, eof, err := l.toEOL()
			if err != nil {
				return nil, err
			}

			if len(bytes.TrimSpace(line)) == 0 {
				break
			}

			partial.Write(line)

			// lack of continuation means this value has been exhausted
			idx := bytes.LastIndex(line, []byte{'\\'})
			if idx == -1 || idx != (len(line)-1) {
				break
			}

			if !eof {
				partial.WriteRune('\n')
			}

			return l.lexOptionValueFunc(section, name, partial), nil
		}

		val := partial.String()
		if strings.HasSuffix(val, "\n") {
			// A newline was added to the end, so the file didn't end with a backslash.
			// => Keep the newline
			val = strings.TrimSpace(val) + "\n"
		} else {
			val = strings.TrimSpace(val)
		}
		l.optchan <- &UnitOption{Section: section, Name: name, Value: val}

		return l.lexNextSectionOrOptionFunc(section), nil
	}
}

// toEOL reads until the end-of-line or end-of-file.
// Returns (data, EOFfound, error)
func (l *lexer) toEOL() ([]byte, bool, error) {
	line, err := l.buf.ReadBytes('\n')
	// ignore EOF here since it's roughly equivalent to EOL
	if err != nil && err != io.EOF {
		return nil, false, err
	}

	line = bytes.TrimSuffix(line, []byte{'\r'})
	line = bytes.TrimSuffix(line, []byte{'\n'})

	return line, err == io.EOF, nil
}

func isComment(r rune) bool {
	return r == '#' || r == ';'
}
