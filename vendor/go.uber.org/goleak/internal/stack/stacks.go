// Copyright (c) 2017 Uber Technologies, Inc.
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
	"bufio"
	"bytes"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"strings"
)

const _defaultBufferSize = 64 * 1024 // 64 KiB

// Stack represents a single Goroutine's stack.
type Stack struct {
	id            int
	state         string
	firstFunction string
	fullStack     *bytes.Buffer
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
	return s.fullStack.String()
}

// FirstFunction returns the name of the first function on the stack.
func (s Stack) FirstFunction() string {
	return s.firstFunction
}

func (s Stack) String() string {
	return fmt.Sprintf(
		"Goroutine %v in state %v, with %v on top of the stack:\n%s",
		s.id, s.state, s.firstFunction, s.Full())
}

func getStacks(all bool) []Stack {
	var stacks []Stack

	var curStack *Stack
	stackReader := bufio.NewReader(bytes.NewReader(getStackBuffer(all)))
	for {
		line, err := stackReader.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			// We're reading using bytes.NewReader which should never fail.
			panic("bufio.NewReader failed on a fixed string")
		}

		// If we see the goroutine header, start a new stack.
		isFirstLine := false
		if strings.HasPrefix(line, "goroutine ") {
			// flush any previous stack
			if curStack != nil {
				stacks = append(stacks, *curStack)
			}
			id, goState := parseGoStackHeader(line)
			curStack = &Stack{
				id:        id,
				state:     goState,
				fullStack: &bytes.Buffer{},
			}
			isFirstLine = true
		}
		curStack.fullStack.WriteString(line)
		if !isFirstLine && curStack.firstFunction == "" {
			curStack.firstFunction = parseFirstFunc(line)
		}
	}

	if curStack != nil {
		stacks = append(stacks, *curStack)
	}
	return stacks
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

func parseFirstFunc(line string) string {
	line = strings.TrimSpace(line)
	if idx := strings.LastIndex(line, "("); idx > 0 {
		return line[:idx]
	}
	panic(fmt.Sprintf("function calls missing parents: %q", line))
}

// parseGoStackHeader parses a stack header that looks like:
// goroutine 643 [runnable]:\n
// And returns the goroutine ID, and the state.
func parseGoStackHeader(line string) (goroutineID int, state string) {
	line = strings.TrimSuffix(line, ":\n")
	parts := strings.SplitN(line, " ", 3)
	if len(parts) != 3 {
		panic(fmt.Sprintf("unexpected stack header format: %q", line))
	}

	id, err := strconv.Atoi(parts[1])
	if err != nil {
		panic(fmt.Sprintf("failed to parse goroutine ID: %v in line %q", parts[1], line))
	}

	state = strings.TrimSuffix(strings.TrimPrefix(parts[2], "["), "]")
	return id, state
}
