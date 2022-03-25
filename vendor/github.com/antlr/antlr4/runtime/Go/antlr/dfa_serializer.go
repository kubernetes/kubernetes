// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"strconv"
	"strings"
)

// DFASerializer is a DFA walker that knows how to dump them to serialized
// strings.
type DFASerializer struct {
	dfa           *DFA
	literalNames  []string
	symbolicNames []string
}

func NewDFASerializer(dfa *DFA, literalNames, symbolicNames []string) *DFASerializer {
	if literalNames == nil {
		literalNames = make([]string, 0)
	}

	if symbolicNames == nil {
		symbolicNames = make([]string, 0)
	}

	return &DFASerializer{
		dfa:           dfa,
		literalNames:  literalNames,
		symbolicNames: symbolicNames,
	}
}

func (d *DFASerializer) String() string {
	if d.dfa.getS0() == nil {
		return ""
	}

	buf := ""
	states := d.dfa.sortedStates()

	for _, s := range states {
		if s.edges != nil {
			n := len(s.edges)

			for j := 0; j < n; j++ {
				t := s.edges[j]

				if t != nil && t.stateNumber != 0x7FFFFFFF {
					buf += d.GetStateString(s)
					buf += "-"
					buf += d.getEdgeLabel(j)
					buf += "->"
					buf += d.GetStateString(t)
					buf += "\n"
				}
			}
		}
	}

	if len(buf) == 0 {
		return ""
	}

	return buf
}

func (d *DFASerializer) getEdgeLabel(i int) string {
	if i == 0 {
		return "EOF"
	} else if d.literalNames != nil && i-1 < len(d.literalNames) {
		return d.literalNames[i-1]
	} else if d.symbolicNames != nil && i-1 < len(d.symbolicNames) {
		return d.symbolicNames[i-1]
	}

	return strconv.Itoa(i - 1)
}

func (d *DFASerializer) GetStateString(s *DFAState) string {
	var a, b string

	if s.isAcceptState {
		a = ":"
	}

	if s.requiresFullContext {
		b = "^"
	}

	baseStateStr := a + "s" + strconv.Itoa(s.stateNumber) + b

	if s.isAcceptState {
		if s.predicates != nil {
			return baseStateStr + "=>" + fmt.Sprint(s.predicates)
		}

		return baseStateStr + "=>" + fmt.Sprint(s.prediction)
	}

	return baseStateStr
}

type LexerDFASerializer struct {
	*DFASerializer
}

func NewLexerDFASerializer(dfa *DFA) *LexerDFASerializer {
	return &LexerDFASerializer{DFASerializer: NewDFASerializer(dfa, nil, nil)}
}

func (l *LexerDFASerializer) getEdgeLabel(i int) string {
	var sb strings.Builder
	sb.Grow(6)
	sb.WriteByte('\'')
	sb.WriteRune(rune(i))
	sb.WriteByte('\'')
	return sb.String()
}

func (l *LexerDFASerializer) String() string {
	if l.dfa.getS0() == nil {
		return ""
	}

	buf := ""
	states := l.dfa.sortedStates()

	for i := 0; i < len(states); i++ {
		s := states[i]

		if s.edges != nil {
			n := len(s.edges)

			for j := 0; j < n; j++ {
				t := s.edges[j]

				if t != nil && t.stateNumber != 0x7FFFFFFF {
					buf += l.GetStateString(s)
					buf += "-"
					buf += l.getEdgeLabel(j)
					buf += "->"
					buf += l.GetStateString(t)
					buf += "\n"
				}
			}
		}
	}

	if len(buf) == 0 {
		return ""
	}

	return buf
}
