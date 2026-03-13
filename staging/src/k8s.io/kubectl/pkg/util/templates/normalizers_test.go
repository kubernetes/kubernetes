/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package templates

import (
	"testing"
)

func TestLongDescMarkdown(t *testing.T) {
	tests := []struct {
		desc string
		in   string
		out  string
	}{
		{
			desc: "Empty input produces empty output",
			in:   "",
			out:  "",
		},
		{
			desc: "Single line text is preserved as is",
			in:   "Some text",
			out:  "Some text",
		},
		{
			desc: "Consecutive new lines are combined into a single paragraph",
			in:   "Line1\nLine2",
			out:  "Line1 Line2",
		},
		{
			desc: "Two paragraphs",
			in:   "Line1\n\nLine2",
			out:  "Line1\n\n Line2",
		},
		{
			desc: "Leading and trailing spaces are stripped (single line)",
			in:   "\t  \nThe text line  \n  \t",
			out:  "The text line",
		},
		{
			desc: "Leading and trailing spaces are stripped (multi line)",
			in:   "\t  \nLine1\nLine2  \n  \t",
			out:  "Line1 Line2",
		},
		{
			desc: "List Items with order",
			in:   "Title\n\n1. First item\n2. Second item\n\nSome text",
			out:  "Title\n\n  1.  First item\n  2.  Second item\n\n Some text",
		},
		{
			desc: "Multi lines without order",
			in:   "\t\t\t\t\tDescriptions.\n\n * Item.\n * Item2.",
			out:  "Descriptions.\n        \n  *  Item.\n  *  Item2.",
		},
		{
			desc: "With code block",
			in:   "Line1\n\n\t<type>.<fieldName>[.<fieldName>]\n\nLine2",
			out:  "Line1\n\n        <type>.<fieldName>[.<fieldName>]\n        \n Line2",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := LongDesc(test.in)
			if got != test.out {
				t.Errorf("expected(%d):\n%s\n=====\ngot(%d):\n%s\n", len(test.out), test.out, len(got), got)
			}
		})
	}
}

func TestMultiLongDescInvocation(t *testing.T) {
	tests := []struct {
		desc string
		in   string
		out  string
	}{
		{
			desc: "Empty input produces empty output",
			in:   "",
			out:  "",
		},
		{
			desc: "Single line text is preserved as is",
			in:   "Some text",
			out:  "Some text",
		},
		{
			desc: "Consecutive new lines are combined into a single paragraph",
			in:   "Line1\nLine2",
			out:  "Line1 Line2",
		},
		{
			desc: "Two paragraphs",
			in:   "Line1\n\nLine2",
			out:  "Line1\n\n Line2",
		},
		{
			desc: "Leading and trailing spaces are stripped (single line)",
			in:   "\t  \nThe text line  \n  \t",
			out:  "The text line",
		},
		{
			desc: "With code block",
			in:   "Line1\n\n\t<type>.<fieldName>[.<fieldName>]\n\nLine2",
			out:  "Line1\n\n        <type>.<fieldName>[.<fieldName>]\n        \n Line2",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := LongDesc(LongDesc(test.in))
			if got != test.out {
				t.Errorf("expected(%d):\n%s\n=====\ngot(%d):\n%s\n", len(test.out), test.out, len(got), got)
			}
		})
	}
}
