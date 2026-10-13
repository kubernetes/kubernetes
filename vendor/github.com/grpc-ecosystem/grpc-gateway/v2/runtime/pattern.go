package runtime

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/grpc-ecosystem/grpc-gateway/v2/utilities"
	"google.golang.org/grpc/grpclog"
)

var (
	// ErrNotMatch indicates that the given HTTP request path does not match to the pattern.
	ErrNotMatch = errors.New("not match to the path pattern")
	// ErrInvalidPattern indicates that the given definition of Pattern is not valid.
	ErrInvalidPattern = errors.New("invalid pattern")
)

type MalformedSequenceError string

func (e MalformedSequenceError) Error() string {
	return "malformed path escape " + strconv.Quote(string(e))
}

type op struct {
	code    utilities.OpCode
	operand int
}

// Pattern is a template pattern of http request paths defined in
// https://github.com/googleapis/googleapis/blob/master/google/api/http.proto
type Pattern struct {
	// ops is a list of operations
	ops []op
	// pool is a constant pool indexed by the operands or vars.
	pool []string
	// vars is a list of variables names to be bound by this pattern
	vars []string
	// stacksize is the max depth of the stack
	stacksize int
	// tailLen is the length of the fixed-size segments after a deep wildcard
	tailLen int
	// verb is the VERB part of the path pattern. It is empty if the pattern does not have VERB part.
	verb string
}

// NewPattern returns a new Pattern from the given definition values.
// "ops" is a sequence of op codes. "pool" is a constant pool.
// "verb" is the verb part of the pattern. It is empty if the pattern does not have the part.
// "version" must be 1 for now.
// It returns an error if the given definition is invalid.
func NewPattern(version int, ops []int, pool []string, verb string) (Pattern, error) {
	if version != 1 {
		grpclog.Errorf("unsupported version: %d", version)
		return Pattern{}, ErrInvalidPattern
	}

	l := len(ops)
	if l%2 != 0 {
		grpclog.Errorf("odd number of ops codes: %d", l)
		return Pattern{}, ErrInvalidPattern
	}

	var (
		typedOps        []op
		stack, maxstack int
		tailLen         int
		pushMSeen       bool
		vars            []string
	)
	for i := 0; i < l; i += 2 {
		op := op{code: utilities.OpCode(ops[i]), operand: ops[i+1]}
		switch op.code {
		case utilities.OpNop:
			continue
		case utilities.OpPush:
			if pushMSeen {
				tailLen++
			}
			stack++
		case utilities.OpPushM:
			if pushMSeen {
				grpclog.Error("pushM appears twice")
				return Pattern{}, ErrInvalidPattern
			}
			pushMSeen = true
			stack++
		case utilities.OpLitPush:
			if op.operand < 0 || len(pool) <= op.operand {
				grpclog.Errorf("negative literal index: %d", op.operand)
				return Pattern{}, ErrInvalidPattern
			}
			if pushMSeen {
				tailLen++
			}
			stack++
		case utilities.OpConcatN:
			if op.operand <= 0 {
				grpclog.Errorf("negative concat size: %d", op.operand)
				return Pattern{}, ErrInvalidPattern
			}
			stack -= op.operand
			if stack < 0 {
				grpclog.Error("stack underflow")
				return Pattern{}, ErrInvalidPattern
			}
			stack++
		case utilities.OpCapture:
			if op.operand < 0 || len(pool) <= op.operand {
				grpclog.Errorf("variable name index out of bound: %d", op.operand)
				return Pattern{}, ErrInvalidPattern
			}
			v := pool[op.operand]
			op.operand = len(vars)
			vars = append(vars, v)
			stack--
			if stack < 0 {
				grpclog.Error("stack underflow")
				return Pattern{}, ErrInvalidPattern
			}
		default:
			grpclog.Errorf("invalid opcode: %d", op.code)
			return Pattern{}, ErrInvalidPattern
		}

		if maxstack < stack {
			maxstack = stack
		}
		typedOps = append(typedOps, op)
	}
	return Pattern{
		ops:       typedOps,
		pool:      pool,
		vars:      vars,
		stacksize: maxstack,
		tailLen:   tailLen,
		verb:      verb,
	}, nil
}

// MustPattern is a helper function which makes it easier to call NewPattern in variable initialization.
func MustPattern(p Pattern, err error) Pattern {
	if err != nil {
		grpclog.Fatalf("Pattern initialization failed: %v", err)
	}
	return p
}

// MatchAndEscape examines components to determine if they match to a Pattern.
// MatchAndEscape will return an error if no Patterns matched or if a pattern
// matched but contained malformed escape sequences. If successful, the function
// returns a mapping from field paths to their captured values.
func (p Pattern) MatchAndEscape(components []string, verb string, unescapingMode UnescapingMode) (map[string]string, error) {
	if p.verb != verb {
		if p.verb != "" {
			return nil, ErrNotMatch
		}
		if len(components) == 0 {
			components = []string{":" + verb}
		} else {
			components = append([]string{}, components...)
			components[len(components)-1] += ":" + verb
		}
	}

	var pos int
	stack := make([]string, 0, p.stacksize)
	captured := make([]string, len(p.vars))
	l := len(components)
	for _, op := range p.ops {
		var err error

		switch op.code {
		case utilities.OpNop:
			continue
		case utilities.OpPush, utilities.OpLitPush:
			if pos >= l {
				return nil, ErrNotMatch
			}
			c := components[pos]
			if op.code == utilities.OpLitPush {
				if lit := p.pool[op.operand]; c != lit {
					return nil, ErrNotMatch
				}
			} else if op.code == utilities.OpPush {
				if c, err = unescape(c, unescapingMode, false); err != nil {
					return nil, err
				}
			}
			stack = append(stack, c)
			pos++
		case utilities.OpPushM:
			end := len(components)
			if end < pos+p.tailLen {
				return nil, ErrNotMatch
			}
			end -= p.tailLen
			c := strings.Join(components[pos:end], "/")
			if c, err = unescape(c, unescapingMode, true); err != nil {
				return nil, err
			}
			stack = append(stack, c)
			pos = end
		case utilities.OpConcatN:
			n := op.operand
			l := len(stack) - n
			stack = append(stack[:l], strings.Join(stack[l:], "/"))
		case utilities.OpCapture:
			n := len(stack) - 1
			captured[op.operand] = stack[n]
			stack = stack[:n]
		}
	}
	if pos < l {
		return nil, ErrNotMatch
	}
	bindings := make(map[string]string)
	for i, val := range captured {
		bindings[p.vars[i]] = val
	}
	return bindings, nil
}

// MatchAndEscape examines components to determine if they match to a Pattern.
// It will never perform per-component unescaping (see: UnescapingModeLegacy).
// MatchAndEscape will return an error if no Patterns matched. If successful,
// the function returns a mapping from field paths to their captured values.
//
// Deprecated: Use MatchAndEscape.
func (p Pattern) Match(components []string, verb string) (map[string]string, error) {
	return p.MatchAndEscape(components, verb, UnescapingModeDefault)
}

// Verb returns the verb part of the Pattern.
func (p Pattern) Verb() string { return p.verb }

func (p Pattern) String() string {
	var stack []string
	for _, op := range p.ops {
		switch op.code {
		case utilities.OpNop:
			continue
		case utilities.OpPush:
			stack = append(stack, "*")
		case utilities.OpLitPush:
			stack = append(stack, p.pool[op.operand])
		case utilities.OpPushM:
			stack = append(stack, "**")
		case utilities.OpConcatN:
			n := op.operand
			l := len(stack) - n
			stack = append(stack[:l], strings.Join(stack[l:], "/"))
		case utilities.OpCapture:
			n := len(stack) - 1
			stack[n] = fmt.Sprintf("{%s=%s}", p.vars[op.operand], stack[n])
		}
	}
	segs := strings.Join(stack, "/")
	if p.verb != "" {
		return fmt.Sprintf("/%s:%s", segs, p.verb)
	}
	return "/" + segs
}

/*
 * The following code is adopted and modified from Go's standard library
 * and carries the attached license.
 *
 *     Copyright 2009 The Go Authors. All rights reserved.
 *     Use of this source code is governed by a BSD-style
 *     license that can be found in the LICENSE file.
 */

// ishex returns whether or not the given byte is a valid hex character
func ishex(c byte) bool {
	switch {
	case '0' <= c && c <= '9':
		return true
	case 'a' <= c && c <= 'f':
		return true
	case 'A' <= c && c <= 'F':
		return true
	}
	return false
}

func isRFC6570Reserved(c byte) bool {
	switch c {
	case '!', '#', '$', '&', '\'', '(', ')', '*',
		'+', ',', '/', ':', ';', '=', '?', '@', '[', ']':
		return true
	default:
		return false
	}
}

// unhex converts a hex point to the bit representation
func unhex(c byte) byte {
	switch {
	case '0' <= c && c <= '9':
		return c - '0'
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10
	}
	return 0
}

// shouldUnescapeWithMode returns true if the character is escapable with the
// given mode
func shouldUnescapeWithMode(c byte, mode UnescapingMode) bool {
	switch mode {
	case UnescapingModeAllExceptReserved:
		if isRFC6570Reserved(c) {
			return false
		}
	case UnescapingModeAllExceptSlash:
		if c == '/' {
			return false
		}
	case UnescapingModeAllCharacters:
		return true
	}
	return true
}

// unescape unescapes a path string using the provided mode
func unescape(s string, mode UnescapingMode, multisegment bool) (string, error) {
	// TODO(v3): remove UnescapingModeLegacy
	if mode == UnescapingModeLegacy {
		return s, nil
	}

	if !multisegment {
		mode = UnescapingModeAllCharacters
	}

	// Count %, check that they're well-formed.
	n := 0
	for i := 0; i < len(s); {
		if s[i] == '%' {
			n++
			if i+2 >= len(s) || !ishex(s[i+1]) || !ishex(s[i+2]) {
				s = s[i:]
				if len(s) > 3 {
					s = s[:3]
				}

				return "", MalformedSequenceError(s)
			}
			i += 3
		} else {
			i++
		}
	}

	if n == 0 {
		return s, nil
	}

	var t strings.Builder
	t.Grow(len(s))
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '%':
			c := unhex(s[i+1])<<4 | unhex(s[i+2])
			if shouldUnescapeWithMode(c, mode) {
				t.WriteByte(c)
				i += 2
				continue
			}
			fallthrough
		default:
			t.WriteByte(s[i])
		}
	}

	return t.String(), nil
}
