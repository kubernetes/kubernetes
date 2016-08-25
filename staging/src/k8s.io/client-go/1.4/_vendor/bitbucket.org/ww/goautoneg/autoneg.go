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

// For internal use, so that we can use the sort interface
type accept_slice []Accept

func (accept accept_slice) Len() int {
	slice := []Accept(accept)
	return len(slice)
}

func (accept accept_slice) Less(i, j int) bool {
	slice := []Accept(accept)
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

func (accept accept_slice) Swap(i, j int) {
	slice := []Accept(accept)
	slice[i], slice[j] = slice[j], slice[i]
}

// Parse an Accept Header string returning a sorted list
// of clauses
func ParseAccept(header string) (accept []Accept) {
	parts := strings.Split(header, ",")
	accept = make([]Accept, 0, len(parts))
	for _, part := range parts {
		part := strings.Trim(part, " ")

		a := Accept{}
		a.Params = make(map[string]string)
		a.Q = 1.0

		mrp := strings.Split(part, ";")

		media_range := mrp[0]
		sp := strings.Split(media_range, "/")
		a.Type = strings.Trim(sp[0], " ")

		switch {
		case len(sp) == 1 && a.Type == "*":
			a.SubType = "*"
		case len(sp) == 2:
			a.SubType = strings.Trim(sp[1], " ")
		default:
			continue
		}

		if len(mrp) == 1 {
			accept = append(accept, a)
			continue
		}

		for _, param := range mrp[1:] {
			sp := strings.SplitN(param, "=", 2)
			if len(sp) != 2 {
				continue
			}
			token := strings.Trim(sp[0], " ")
			if token == "q" {
				a.Q, _ = strconv.ParseFloat(sp[1], 32)
			} else {
				a.Params[token] = strings.Trim(sp[1], " ")
			}
		}

		accept = append(accept, a)
	}

	slice := accept_slice(accept)
	sort.Sort(slice)

	return
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
