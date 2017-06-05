// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cldr

import (
	"fmt"
	"strings"
	"testing"
)

// A recorder implements the RuleProcessor interface, whereby its methods
// simply record the invocations.
type recorder struct {
	calls []string
}

func (r *recorder) Reset(anchor string, before int) error {
	if before > 5 {
		return fmt.Errorf("before %d > 5", before)
	}
	r.calls = append(r.calls, fmt.Sprintf("R:%s-%d", anchor, before))
	return nil
}

func (r *recorder) Insert(level int, str, context, extend string) error {
	s := fmt.Sprintf("O:%d:%s", level, str)
	if context != "" {
		s += "|" + context
	}
	if extend != "" {
		s += "/" + extend
	}
	r.calls = append(r.calls, s)
	return nil
}

func (r *recorder) Index(id string) {
	r.calls = append(r.calls, fmt.Sprintf("I:%s", id))
}

func (r *recorder) Error(err error) {
	r.calls = append(r.calls, fmt.Sprintf("E:%v", err))
}

func TestRuleProcessor(t *testing.T) {
	for _, tt := range []struct {
		desc string
		in   string
		out  string
	}{
		{desc: "empty"},
		{desc: "whitespace and comments only",
			in: `


		  		# adsfads
# adfadf
		`,
		},
		{
			desc: "reset anchor",
			in: `
			& a
			&b    #
			&  [    before 3  ]  c
			& [before 4] d & ee
			& [first tertiary ignorable]
			&'g'
			& 	'h''h'h'h'
			&'\u0069'  # LATIN SMALL LETTER I
			`,
			out: `
			R:a-0
			R:b-0
			R:c-3
			R:d-4
			R:ee-0
			R:<first tertiary ignorable/>-0
			R:g-0
			R:hhhh-0
			R:i-0
			`,
		},
		{
			desc: "ordering",
			in: `
			& 0
			< 1 <<''2#
<<<			3'3''33'3#
			<<<<4
			= 5 << 6 | s
			<<<< 7 / z
			<< 8'' | s / ch
			`,
			out: `
			R:0-0
			O:1:1
			O:2:'2
			O:3:33333
			O:4:4
			O:5:5
			O:2:6|s
			O:4:7/z
			O:2:8'|s/ch
			`,
		},
		{
			desc: "index",
			in:   "< '\ufdd0'A",
			out:  "I:A",
		},
		{
			desc: "sequence",
			in: `
			& 0
			<<* 1234
			<* a-cde-f
			=* q-q
			`,
			out: `
			R:0-0
			O:2:1
			O:2:2
			O:2:3
			O:2:4
			O:1:a
			O:1:b
			O:1:c
			O:1:d
			O:1:e
			O:1:f
			O:5:q
			`,
		},
		{
			desc: "compact",
			in:   "&B<t<<<T<s<<<S<e<<<E",
			out: `
			R:B-0
			O:1:t
			O:3:T
			O:1:s
			O:3:S
			O:1:e
			O:3:E
			`,
		},
		{
			desc: "err operator",
			in:   "a",
			out:  "E:1: illegal operator 'a'",
		},
		{
			desc: "err line number",
			in: `& a
			<< b
			a`,
			out: `
			R:a-0
			O:2:b
			E:3: illegal operator 'a'`,
		},
		{
			desc: "err empty anchor",
			in: " &			",
			out: "E:1: missing string",
		},
		{
			desc: "err anchor invalid special 1",
			in: " &	[ foo ",
			out: "E:1: unmatched bracket",
		},
		{
			desc: "err anchor invalid special 2",
			in:   "&[",
			out:  "E:1: unmatched bracket",
		},
		{
			desc: "err anchor invalid before 1",
			in:   "&[before a]",
			out:  `E:1: strconv.ParseUint: parsing "a": invalid syntax`,
		},
		{
			desc: "err anchor invalid before 2",
			in:   "&[before 12]",
			out:  `E:1: strconv.ParseUint: parsing "12": value out of range`,
		},
		{
			desc: "err anchor invalid before 3",
			in:   "&[before 2]",
			out:  "E:1: missing string",
		},
		{
			desc: "err anchor invalid before 4",
			in:   "&[before 6] a",
			out:  "E:1: before 6 > 5",
		},
		{
			desc: "err empty order",
			in:   " < ",
			out:  "E:1: missing string",
		},
		{
			desc: "err empty identity",
			in:   " = ",
			out:  "E:1: missing string",
		},
		{
			desc: "err empty context",
			in:   " < a |  ",
			out:  "E:1: missing string after context",
		},
		{
			desc: "err empty extend",
			in:   " < a /  ",
			out:  "E:1: missing string after extension",
		},
		{
			desc: "err empty sequence",
			in:   " <* ",
			out:  "E:1: empty sequence",
		},
		{
			desc: "err sequence 1",
			in:   " <* -a",
			out:  "E:1: range without starter value",
		},
		{
			desc: "err sequence 3",
			in:   " <* a-a-b",
			out: `O:1:a
			E:1: range without starter value
			`,
		},
		{
			desc: "err sequence 3",
			in:   " <* b-a",
			out: `O:1:b
			E:1: invalid range 'b'-'a'
			`,
		},
		{
			desc: "err unmatched quote",
			in:   " < 'b",
			out: ` E:1: unmatched single quote
			`,
		},
	} {
		rec := &recorder{}
		err := Collation{
			Cr: []*Common{
				{hidden: hidden{CharData: tt.in}},
			},
		}.Process(rec)
		if err != nil {
			rec.Error(err)
		}
		got := rec.calls
		want := strings.Split(strings.TrimSpace(tt.out), "\n")
		if tt.out == "" {
			want = nil
		}
		if len(got) != len(want) {
			t.Errorf("%s: nResults: got %d; want %d", tt.desc, len(got), len(want))
			continue
		}
		for i, g := range got {
			if want := strings.TrimSpace(want[i]); g != want {
				t.Errorf("%s:%d: got %q; want %q", tt.desc, i, g, want)
			}
		}
	}
}
