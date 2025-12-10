// Copyright (c) 2017-2023 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package stack

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"strings"
)

const _defaultBufferSize = 64 * 1024 // 64 KiB

// Stack represents a single Goroutine's stack.
type Stack struct {
	id    int
	state string // e.g. 'running', 'chan receive'

	// The first function on the stack.
	firstFunction string

	// A set of all functions in the stack,
	allFunctions map[string]struct{}

	// Full, raw stack trace.
	fullStack string
}

// ID returns the goroutine ID.
func (s Stack) ID() int {
	return s.id
}

// State returns the Goroutine's state.
func (s Stack) State() string {
	return s.state
}

// Full returns the full stack trace for this goroutine.
func (s Stack) Full() string {
	return s.fullStack
}

// FirstFunction returns the name of the first function on the stack.
func (s Stack) FirstFunction() string {
	return s.firstFunction
}

// HasFunction reports whether the stack has the given function
// anywhere in it.
func (s Stack) HasFunction(name string) bool {
	_, ok := s.allFunctions[name]
	return ok
}

func (s Stack) String() string {
	return fmt.Sprintf(
		"Goroutine %v in state %v, with %v on top of the stack:\n%s",
		s.id, s.state, s.firstFunction, s.Full())
}

func getStacks(all bool) []Stack {
	trace := getStackBuffer(all)
	stacks, err := newStackParser(bytes.NewReader(trace)).Parse()
	if err != nil {
		// Well-formed stack traces should never fail to parse.
		// If they do, it's a bug in this package.
		// Panic so we can fix it.
		panic(fmt.Sprintf("Failed to parse stack trace: %v\n%s", err, trace))
	}
	return stacks
}

type stackParser struct {
	scan   *scanner
	stacks []Stack
	errors []error
}

func newStackParser(r io.Reader) *stackParser {
	return &stackParser{
		scan: newScanner(r),
	}
}

func (p *stackParser) Parse() ([]Stack, error) {
	for p.scan.Scan() {
		line := p.scan.Text()

		// If we see the goroutine header, start a new stack.
		if strings.HasPrefix(line, "goroutine ") {
			stack, err := p.parseStack(line)
			if err != nil {
				p.errors = append(p.errors, err)
				continue
			}
			p.stacks = append(p.stacks, stack)
		}
	}

	p.errors = append(p.errors, p.scan.Err())
	return p.stacks, errors.Join(p.errors...)
}

// parseStack parses a single stack trace from the given scanner.
// line is the first line of the stack trace, which should look like:
//
//	goroutine 123 [runnable]:
func (p *stackParser) parseStack(line string) (Stack, error) {
	id, state, err := parseGoStackHeader(line)
	if err != nil {
		return Stack{}, fmt.Errorf("parse header: %w", err)
	}

	// Read the rest of the stack trace.
	var (
		firstFunction string
		fullStack     bytes.Buffer
	)
	funcs := make(map[string]struct{})
	for p.scan.Scan() {
		line := p.scan.Text()
		if strings.HasPrefix(line, "goroutine ") {
			// If we see the goroutine header,
			// it's the end of this stack.
			// Unscan so the next Scan sees the same line.
			p.scan.Unscan()
			break
		}

		fullStack.WriteString(line)
		fullStack.WriteByte('\n') // scanner trims the newline

		if len(line) == 0 {
			// Empty line usually marks the end of the stack
			// but we don't want to have to rely on that.
			// Just skip it.
			continue
		}

		funcName, creator, err := parseFuncName(line)
		if err != nil {
			return Stack{}, fmt.Errorf("parse function: %w", err)
		}
		if !creator {
			// A function is part of a goroutine's stack
			// only if it's not a "created by" function.
			//
			// The creator function is part of a different stack.
			// We don't care about it right now.
			funcs[funcName] = struct{}{}
			if firstFunction == "" {
				firstFunction = funcName
			}

		}

		// The function name followed by a line in the form:
		//
		//	<tab>example.com/path/to/package/file.go:123 +0x123
		//
		// We don't care about the position so we can skip this line.
		if p.scan.Scan() {
			// Be defensive:
			// Skip the line only if it starts with a tab.
			bs := p.scan.Bytes()
			if len(bs) > 0 && bs[0] == '\t' {
				fullStack.Write(bs)
				fullStack.WriteByte('\n')
			} else {
				// Put it back and let the next iteration handle it
				// if it doesn't start with a tab.
				p.scan.Unscan()
			}
		}

		if creator {
			// The "created by" line is the last line of the stack.
			// We can stop parsing now.
			//
			// Note that if tracebackancestors=N is set,
			// there may be more a traceback of the creator function
			// following the "created by" line,
			// but it should not be considered part of this stack.
			// e.g.,
			//
			// created by testing.(*T).Run in goroutine 1
			//         /usr/lib/go/src/testing/testing.go:1648 +0x3ad
			// [originating from goroutine 1]:
			// testing.(*T).Run(...)
			//         /usr/lib/go/src/testing/testing.go:1649 +0x3ad
			//
			break
		}
	}

	return Stack{
		id:            id,
		state:         state,
		firstFunction: firstFunction,
		allFunctions:  funcs,
		fullStack:     fullStack.String(),
	}, nil
}

// All returns the stacks for all running goroutines.
func All() []Stack {
	return getStacks(true)
}

// Current returns the stack for the current goroutine.
func Current() Stack {
	return getStacks(false)[0]
}

func getStackBuffer(all bool) []byte {
	for i := _defaultBufferSize; ; i *= 2 {
		buf := make([]byte, i)
		if n := runtime.Stack(buf, all); n < i {
			return buf[:n]
		}
	}
}

// Parses a single function from the given line.
// The line is in one of these formats:
//
//	example.com/path/to/package.funcName(args...)
//	example.com/path/to/package.(*typeName).funcName(args...)
//	created by example.com/path/to/package.funcName
//	created by example.com/path/to/package.funcName in goroutine [...]
//
// Also reports whether the line was a "created by" line.
func parseFuncName(line string) (name string, creator bool, err error) {
	if after, ok := strings.CutPrefix(line, "created by "); ok {
		// The function name is the part after "created by "
		// and before " in goroutine [...]".
		idx := strings.Index(after, " in goroutine")
		if idx >= 0 {
			after = after[:idx]
		}
		name = after
		creator = true
	} else if idx := strings.LastIndexByte(line, '('); idx >= 0 {
		// The function name is the part before the last '('.
		name = line[:idx]
	}

	if name == "" {
		return "", false, fmt.Errorf("no function found: %q", line)
	}

	return name, creator, nil
}

// parseGoStackHeader parses a stack header that looks like:
// goroutine 643 [runnable]:\n
// And returns the goroutine ID, and the state.
func parseGoStackHeader(line string) (goroutineID int, state string, err error) {
	// The scanner will have already trimmed the "\n",
	// but we'll guard against it just in case.
	//
	// Trimming them separately makes them both optional.
	line = strings.TrimSuffix(strings.TrimSuffix(line, ":"), "\n")
	parts := strings.SplitN(line, " ", 3)
	if len(parts) != 3 {
		return 0, "", fmt.Errorf("unexpected format: %q", line)
	}

	id, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, "", fmt.Errorf("bad goroutine ID %q in line %q", parts[1], line)
	}

	state = strings.TrimSuffix(strings.TrimPrefix(parts[2], "["), "]")
	return id, state, nil
}
