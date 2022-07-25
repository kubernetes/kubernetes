/*
HTTP Content-Type Autonegotiation.

The functions in this package implement the behaviour specified in
http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html

Copyright (c) 2011, Open Knowledge Foundation Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

    Neither the name of the Open Knowledge Foundation Ltd. nor the
    names of its contributors may be used to endorse or promote
    products derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

package goautoneg

import (
	"sort"
	"strconv"
	"strings"
)

// Structure to represent a clause in an HTTP Accept Header
type Accept struct {
	Type, SubType string
	Q             float64
	Params        map[string]string
}

// acceptSlice is defined to implement sort interface.
type acceptSlice []Accept

func (slice acceptSlice) Len() int {
	return len(slice)
}

func (slice acceptSlice) Less(i, j int) bool {
	ai, aj := slice[i], slice[j]
	if ai.Q > aj.Q {
		return true
	}
	if ai.Type != "*" && aj.Type == "*" {
		return true
	}
	if ai.SubType != "*" && aj.SubType == "*" {
		return true
	}
	return false
}

func (slice acceptSlice) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

func stringTrimSpaceCutset(r rune) bool {
	return r == ' '
}

func nextSplitElement(s, sep string) (item string, remaining string) {
	if index := strings.Index(s, sep); index != -1 {
		return s[:index], s[index+1:]
	}
	return s, ""
}

// Parse an Accept Header string returning a sorted list
// of clauses
func ParseAccept(header string) acceptSlice {
	partsCount := 0
	remaining := header
	for len(remaining) > 0 {
		partsCount++
		_, remaining = nextSplitElement(remaining, ",")
	}
	accept := make(acceptSlice, 0, partsCount)

	remaining = header
	var part string
	for len(remaining) > 0 {
		part, remaining = nextSplitElement(remaining, ",")
		part = strings.TrimFunc(part, stringTrimSpaceCutset)

		a := Accept{
			Q: 1.0,
		}

		sp, remainingPart := nextSplitElement(part, ";")

		sp0, spRemaining := nextSplitElement(sp, "/")
		a.Type = strings.TrimFunc(sp0, stringTrimSpaceCutset)

		switch {
		case len(spRemaining) == 0:
			if a.Type == "*" {
				a.SubType = "*"
			} else {
				continue
			}
		default:
			var sp1 string
			sp1, spRemaining = nextSplitElement(spRemaining, "/")
			if len(spRemaining) > 0 {
				continue
			}
			a.SubType = strings.TrimFunc(sp1, stringTrimSpaceCutset)
		}

		if len(remainingPart) == 0 {
			accept = append(accept, a)
			continue
		}

		a.Params = make(map[string]string)
		for len(remainingPart) > 0 {
			sp, remainingPart = nextSplitElement(remainingPart, ";")
			sp0, spRemaining = nextSplitElement(sp, "=")
			if len(spRemaining) == 0 {
				continue
			}
			var sp1 string
			sp1, spRemaining = nextSplitElement(spRemaining, "=")
			if len(spRemaining) != 0 {
				continue
			}
			token := strings.TrimFunc(sp0, stringTrimSpaceCutset)
			if token == "q" {
				a.Q, _ = strconv.ParseFloat(sp1, 32)
			} else {
				a.Params[token] = strings.TrimFunc(sp1, stringTrimSpaceCutset)
			}
		}

		accept = append(accept, a)
	}

	sort.Sort(accept)
	return accept
}

// Negotiate the most appropriate content_type given the accept header
// and a list of alternatives.
func Negotiate(header string, alternatives []string) (content_type string) {
	asp := make([][]string, 0, len(alternatives))
	for _, ctype := range alternatives {
		asp = append(asp, strings.SplitN(ctype, "/", 2))
	}
	for _, clause := range ParseAccept(header) {
		for i, ctsp := range asp {
			if clause.Type == ctsp[0] && clause.SubType == ctsp[1] {
				content_type = alternatives[i]
				return
			}
			if clause.Type == ctsp[0] && clause.SubType == "*" {
				content_type = alternatives[i]
				return
			}
			if clause.Type == "*" && clause.SubType == "*" {
				content_type = alternatives[i]
				return
			}
		}
	}
	return
}
