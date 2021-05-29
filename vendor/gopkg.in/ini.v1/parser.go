// Copyright 2015 Unknwon
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package ini

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

const minReaderBufferSize = 4096

var pythonMultiline = regexp.MustCompile(`^([\t\f ]+)(.*)`)

type parserOptions struct {
	IgnoreContinuation          bool
	IgnoreInlineComment         bool
	AllowPythonMultilineValues  bool
	SpaceBeforeInlineComment    bool
	UnescapeValueDoubleQuotes   bool
	UnescapeValueCommentSymbols bool
	PreserveSurroundedQuote     bool
	DebugFunc                   DebugFunc
	ReaderBufferSize            int
}

type parser struct {
	buf     *bufio.Reader
	options parserOptions

	isEOF   bool
	count   int
	comment *bytes.Buffer
}

func (p *parser) debug(format string, args ...interface{}) {
	if p.options.DebugFunc != nil {
		p.options.DebugFunc(fmt.Sprintf(format, args...))
	}
}

func newParser(r io.Reader, opts parserOptions) *parser {
	size := opts.ReaderBufferSize
	if size < minReaderBufferSize {
		size = minReaderBufferSize
	}

	return &parser{
		buf:     bufio.NewReaderSize(r, size),
		options: opts,
		count:   1,
		comment: &bytes.Buffer{},
	}
}

// BOM handles header of UTF-8, UTF-16 LE and UTF-16 BE's BOM format.
// http://en.wikipedia.org/wiki/Byte_order_mark#Representations_of_byte_order_marks_by_encoding
func (p *parser) BOM() error {
	mask, err := p.buf.Peek(2)
	if err != nil && err != io.EOF {
		return err
	} else if len(mask) < 2 {
		return nil
	}

	switch {
	case mask[0] == 254 && mask[1] == 255:
		fallthrough
	case mask[0] == 255 && mask[1] == 254:
		p.buf.Read(mask)
	case mask[0] == 239 && mask[1] == 187:
		mask, err := p.buf.Peek(3)
		if err != nil && err != io.EOF {
			return err
		} else if len(mask) < 3 {
			return nil
		}
		if mask[2] == 191 {
			p.buf.Read(mask)
		}
	}
	return nil
}

func (p *parser) readUntil(delim byte) ([]byte, error) {
	data, err := p.buf.ReadBytes(delim)
	if err != nil {
		if err == io.EOF {
			p.isEOF = true
		} else {
			return nil, err
		}
	}
	return data, nil
}

func cleanComment(in []byte) ([]byte, bool) {
	i := bytes.IndexAny(in, "#;")
	if i == -1 {
		return nil, false
	}
	return in[i:], true
}

func readKeyName(delimiters string, in []byte) (string, int, error) {
	line := string(in)

	// Check if key name surrounded by quotes.
	var keyQuote string
	if line[0] == '"' {
		if len(line) > 6 && string(line[0:3]) == `"""` {
			keyQuote = `"""`
		} else {
			keyQuote = `"`
		}
	} else if line[0] == '`' {
		keyQuote = "`"
	}

	// Get out key name
	endIdx := -1
	if len(keyQuote) > 0 {
		startIdx := len(keyQuote)
		// FIXME: fail case -> """"""name"""=value
		pos := strings.Index(line[startIdx:], keyQuote)
		if pos == -1 {
			return "", -1, fmt.Errorf("missing closing key quote: %s", line)
		}
		pos += startIdx

		// Find key-value delimiter
		i := strings.IndexAny(line[pos+startIdx:], delimiters)
		if i < 0 {
			return "", -1, ErrDelimiterNotFound{line}
		}
		endIdx = pos + i
		return strings.TrimSpace(line[startIdx:pos]), endIdx + startIdx + 1, nil
	}

	endIdx = strings.IndexAny(line, delimiters)
	if endIdx < 0 {
		return "", -1, ErrDelimiterNotFound{line}
	}
	return strings.TrimSpace(line[0:endIdx]), endIdx + 1, nil
}

func (p *parser) readMultilines(line, val, valQuote string) (string, error) {
	for {
		data, err := p.readUntil('\n')
		if err != nil {
			return "", err
		}
		next := string(data)

		pos := strings.LastIndex(next, valQuote)
		if pos > -1 {
			val += next[:pos]

			comment, has := cleanComment([]byte(next[pos:]))
			if has {
				p.comment.Write(bytes.TrimSpace(comment))
			}
			break
		}
		val += next
		if p.isEOF {
			return "", fmt.Errorf("missing closing key quote from '%s' to '%s'", line, next)
		}
	}
	return val, nil
}

func (p *parser) readContinuationLines(val string) (string, error) {
	for {
		data, err := p.readUntil('\n')
		if err != nil {
			return "", err
		}
		next := strings.TrimSpace(string(data))

		if len(next) == 0 {
			break
		}
		val += next
		if val[len(val)-1] != '\\' {
			break
		}
		val = val[:len(val)-1]
	}
	return val, nil
}

// hasSurroundedQuote check if and only if the first and last characters
// are quotes \" or \'.
// It returns false if any other parts also contain same kind of quotes.
func hasSurroundedQuote(in string, quote byte) bool {
	return len(in) >= 2 && in[0] == quote && in[len(in)-1] == quote &&
		strings.IndexByte(in[1:], quote) == len(in)-2
}

func (p *parser) readValue(in []byte, bufferSize int) (string, error) {

	line := strings.TrimLeftFunc(string(in), unicode.IsSpace)
	if len(line) == 0 {
		if p.options.AllowPythonMultilineValues && len(in) > 0 && in[len(in)-1] == '\n' {
			return p.readPythonMultilines(line, bufferSize)
		}
		return "", nil
	}

	var valQuote string
	if len(line) > 3 && string(line[0:3]) == `"""` {
		valQuote = `"""`
	} else if line[0] == '`' {
		valQuote = "`"
	} else if p.options.UnescapeValueDoubleQuotes && line[0] == '"' {
		valQuote = `"`
	}

	if len(valQuote) > 0 {
		startIdx := len(valQuote)
		pos := strings.LastIndex(line[startIdx:], valQuote)
		// Check for multi-line value
		if pos == -1 {
			return p.readMultilines(line, line[startIdx:], valQuote)
		}

		if p.options.UnescapeValueDoubleQuotes && valQuote == `"` {
			return strings.Replace(line[startIdx:pos+startIdx], `\"`, `"`, -1), nil
		}
		return line[startIdx : pos+startIdx], nil
	}

	lastChar := line[len(line)-1]
	// Won't be able to reach here if value only contains whitespace
	line = strings.TrimSpace(line)
	trimmedLastChar := line[len(line)-1]

	// Check continuation lines when desired
	if !p.options.IgnoreContinuation && trimmedLastChar == '\\' {
		return p.readContinuationLines(line[:len(line)-1])
	}

	// Check if ignore inline comment
	if !p.options.IgnoreInlineComment {
		var i int
		if p.options.SpaceBeforeInlineComment {
			i = strings.Index(line, " #")
			if i == -1 {
				i = strings.Index(line, " ;")
			}

		} else {
			i = strings.IndexAny(line, "#;")
		}

		if i > -1 {
			p.comment.WriteString(line[i:])
			line = strings.TrimSpace(line[:i])
		}

	}

	// Trim single and double quotes
	if (hasSurroundedQuote(line, '\'') ||
		hasSurroundedQuote(line, '"')) && !p.options.PreserveSurroundedQuote {
		line = line[1 : len(line)-1]
	} else if len(valQuote) == 0 && p.options.UnescapeValueCommentSymbols {
		if strings.Contains(line, `\;`) {
			line = strings.Replace(line, `\;`, ";", -1)
		}
		if strings.Contains(line, `\#`) {
			line = strings.Replace(line, `\#`, "#", -1)
		}
	} else if p.options.AllowPythonMultilineValues && lastChar == '\n' {
		return p.readPythonMultilines(line, bufferSize)
	}

	return line, nil
}

func (p *parser) readPythonMultilines(line string, bufferSize int) (string, error) {
	parserBufferPeekResult, _ := p.buf.Peek(bufferSize)
	peekBuffer := bytes.NewBuffer(parserBufferPeekResult)

	indentSize := 0
	for {
		peekData, peekErr := peekBuffer.ReadBytes('\n')
		if peekErr != nil {
			if peekErr == io.EOF {
				p.debug("readPythonMultilines: io.EOF, peekData: %q, line: %q", string(peekData), line)
				return line, nil
			}

			p.debug("readPythonMultilines: failed to peek with error: %v", peekErr)
			return "", peekErr
		}

		p.debug("readPythonMultilines: parsing %q", string(peekData))

		peekMatches := pythonMultiline.FindStringSubmatch(string(peekData))
		p.debug("readPythonMultilines: matched %d parts", len(peekMatches))
		for n, v := range peekMatches {
			p.debug("   %d: %q", n, v)
		}

		// Return if not a Python multiline value.
		if len(peekMatches) != 3 {
			p.debug("readPythonMultilines: end of value, got: %q", line)
			return line, nil
		}

		// Determine indent size and line prefix.
		currentIndentSize := len(peekMatches[1])
		if indentSize < 1 {
			indentSize = currentIndentSize
			p.debug("readPythonMultilines: indent size is %d", indentSize)
		}

		// Make sure each line is indented at least as far as first line.
		if currentIndentSize < indentSize {
			p.debug("readPythonMultilines: end of value, current indent: %d, expected indent: %d, line: %q", currentIndentSize, indentSize, line)
			return line, nil
		}

		// Advance the parser reader (buffer) in-sync with the peek buffer.
		_, err := p.buf.Discard(len(peekData))
		if err != nil {
			p.debug("readPythonMultilines: failed to skip to the end, returning error")
			return "", err
		}

		// Handle indented empty line.
		line += "\n" + peekMatches[1][indentSize:] + peekMatches[2]
	}
}

// parse parses data through an io.Reader.
func (f *File) parse(reader io.Reader) (err error) {
	p := newParser(reader, parserOptions{
		IgnoreContinuation:          f.options.IgnoreContinuation,
		IgnoreInlineComment:         f.options.IgnoreInlineComment,
		AllowPythonMultilineValues:  f.options.AllowPythonMultilineValues,
		SpaceBeforeInlineComment:    f.options.SpaceBeforeInlineComment,
		UnescapeValueDoubleQuotes:   f.options.UnescapeValueDoubleQuotes,
		UnescapeValueCommentSymbols: f.options.UnescapeValueCommentSymbols,
		PreserveSurroundedQuote:     f.options.PreserveSurroundedQuote,
		DebugFunc:                   f.options.DebugFunc,
		ReaderBufferSize:            f.options.ReaderBufferSize,
	})
	if err = p.BOM(); err != nil {
		return fmt.Errorf("BOM: %v", err)
	}

	// Ignore error because default section name is never empty string.
	name := DefaultSection
	if f.options.Insensitive {
		name = strings.ToLower(DefaultSection)
	}
	section, _ := f.NewSection(name)

	// This "last" is not strictly equivalent to "previous one" if current key is not the first nested key
	var isLastValueEmpty bool
	var lastRegularKey *Key

	var line []byte
	var inUnparseableSection bool

	// NOTE: Iterate and increase `currentPeekSize` until
	// the size of the parser buffer is found.
	// TODO(unknwon): When Golang 1.10 is the lowest version supported, replace with `parserBufferSize := p.buf.Size()`.
	parserBufferSize := 0
	// NOTE: Peek 4kb at a time.
	currentPeekSize := minReaderBufferSize

	if f.options.AllowPythonMultilineValues {
		for {
			peekBytes, _ := p.buf.Peek(currentPeekSize)
			peekBytesLength := len(peekBytes)

			if parserBufferSize >= peekBytesLength {
				break
			}

			currentPeekSize *= 2
			parserBufferSize = peekBytesLength
		}
	}

	for !p.isEOF {
		line, err = p.readUntil('\n')
		if err != nil {
			return err
		}

		if f.options.AllowNestedValues &&
			isLastValueEmpty && len(line) > 0 {
			if line[0] == ' ' || line[0] == '\t' {
				lastRegularKey.addNestedValue(string(bytes.TrimSpace(line)))
				continue
			}
		}

		line = bytes.TrimLeftFunc(line, unicode.IsSpace)
		if len(line) == 0 {
			continue
		}

		// Comments
		if line[0] == '#' || line[0] == ';' {
			// Note: we do not care ending line break,
			// it is needed for adding second line,
			// so just clean it once at the end when set to value.
			p.comment.Write(line)
			continue
		}

		// Section
		if line[0] == '[' {
			// Read to the next ']' (TODO: support quoted strings)
			closeIdx := bytes.LastIndexByte(line, ']')
			if closeIdx == -1 {
				return fmt.Errorf("unclosed section: %s", line)
			}

			name := string(line[1:closeIdx])
			section, err = f.NewSection(name)
			if err != nil {
				return err
			}

			comment, has := cleanComment(line[closeIdx+1:])
			if has {
				p.comment.Write(comment)
			}

			section.Comment = strings.TrimSpace(p.comment.String())

			// Reset aotu-counter and comments
			p.comment.Reset()
			p.count = 1

			inUnparseableSection = false
			for i := range f.options.UnparseableSections {
				if f.options.UnparseableSections[i] == name ||
					(f.options.Insensitive && strings.ToLower(f.options.UnparseableSections[i]) == strings.ToLower(name)) {
					inUnparseableSection = true
					continue
				}
			}
			continue
		}

		if inUnparseableSection {
			section.isRawSection = true
			section.rawBody += string(line)
			continue
		}

		kname, offset, err := readKeyName(f.options.KeyValueDelimiters, line)
		if err != nil {
			// Treat as boolean key when desired, and whole line is key name.
			if IsErrDelimiterNotFound(err) {
				switch {
				case f.options.AllowBooleanKeys:
					kname, err := p.readValue(line, parserBufferSize)
					if err != nil {
						return err
					}
					key, err := section.NewBooleanKey(kname)
					if err != nil {
						return err
					}
					key.Comment = strings.TrimSpace(p.comment.String())
					p.comment.Reset()
					continue

				case f.options.SkipUnrecognizableLines:
					continue
				}
			}
			return err
		}

		// Auto increment.
		isAutoIncr := false
		if kname == "-" {
			isAutoIncr = true
			kname = "#" + strconv.Itoa(p.count)
			p.count++
		}

		value, err := p.readValue(line[offset:], parserBufferSize)
		if err != nil {
			return err
		}
		isLastValueEmpty = len(value) == 0

		key, err := section.NewKey(kname, value)
		if err != nil {
			return err
		}
		key.isAutoIncrement = isAutoIncr
		key.Comment = strings.TrimSpace(p.comment.String())
		p.comment.Reset()
		lastRegularKey = key
	}
	return nil
}
