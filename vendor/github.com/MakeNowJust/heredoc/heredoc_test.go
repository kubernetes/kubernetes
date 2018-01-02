// Copyright (c) 2014-2017 TSUYUSATO Kitsune
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php

package heredoc

import (
	"testing"
)

type testCase struct {
	raw, expect string
}

var tests = []testCase{
	{`
		Foo
		Bar
		`,
		"Foo\nBar\n"},
	{`Foo
		Bar`,
		"Foo\nBar"},
	{`Foo
			
		Bar
		`,
		"Foo\n\t\nBar\n"}, // Second line contains two tabs.
	{`
		Foo
			Bar
				Hoge
					`,
		"Foo\n\tBar\n\t\tHoge\n\t\t\t"},
	{`Foo Bar`, "Foo Bar"},
	{
		`
		Foo
		Bar
	`, "Foo\nBar\n"},
}

func TestDoc(t *testing.T) {
	for i, test := range tests {
		result := Doc(test.raw)
		if result != test.expect {
			t.Errorf("tests[%d] failed: expected=> %#v, result=> %#v", i, test.expect, result)
		}
	}
}

func TestDocf(t *testing.T) {
	tc := `
		int: %3d
		string: %s
	`
	i := 42
	s := "Hello"
	expect := "int:  42\nstring: Hello\n"

	result := Docf(tc, i, s)
	if result != expect {
		t.Errorf("test failed: expected=> %#v, result=> %#v", expect, result)
	}
}
