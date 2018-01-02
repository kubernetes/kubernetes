package filters

import (
	"fmt"
	"strconv"
	"testing"
)

type tokenResult struct {
	pos   int
	token token
	text  string
}

func (tr tokenResult) String() string {
	return fmt.Sprintf("{pos: %v, token: %v, text: %q}", tr.pos, tr.token, tr.text)
}

func TestScanner(t *testing.T) {

	for _, testcase := range []struct {
		name     string
		input    string
		expected []tokenResult
	}{
		{
			name:  "Field",
			input: "name",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "name"},
				{pos: 4, token: tokenEOF},
			},
		},
		{
			name:  "SelectorsWithOperators",
			input: "name==value,foo!=bar",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "name"},
				{pos: 4, token: tokenOperator, text: "=="},
				{pos: 6, token: tokenValue, text: "value"},
				{pos: 11, token: tokenSeparator, text: ","},
				{pos: 12, token: tokenField, text: "foo"},
				{pos: 15, token: tokenOperator, text: "!="},
				{pos: 17, token: tokenValue, text: "bar"},
				{pos: 20, token: tokenEOF},
			},
		},
		{
			name:  "SelectorsWithFieldPaths",
			input: "name==value,labels.foo=value,other.bar~=match",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "name"},
				{pos: 4, token: tokenOperator, text: "=="},
				{pos: 6, token: tokenValue, text: "value"},
				{pos: 11, token: tokenSeparator, text: ","},
				{pos: 12, token: tokenField, text: "labels"},
				{pos: 18, token: tokenSeparator, text: "."},
				{pos: 19, token: tokenField, text: "foo"},
				{pos: 22, token: tokenOperator, text: "="},
				{pos: 23, token: tokenValue, text: "value"},
				{pos: 28, token: tokenSeparator, text: ","},
				{pos: 29, token: tokenField, text: "other"},
				{pos: 34, token: tokenSeparator, text: "."},
				{pos: 35, token: tokenField, text: "bar"},
				{pos: 38, token: tokenOperator, text: "~="},
				{pos: 40, token: tokenValue, text: "match"},
				{pos: 45, token: tokenEOF},
			},
		},
		{
			name:  "RegexpValue",
			input: "name~=[abc]+,foo=test",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "name"},
				{pos: 4, token: tokenOperator, text: "~="},
				{pos: 6, token: tokenValue, text: "[abc]+"},
				{pos: 12, token: tokenSeparator, text: ","},
				{pos: 13, token: tokenField, text: "foo"},
				{pos: 16, token: tokenOperator, text: "="},
				{pos: 17, token: tokenValue, text: "test"},
				{pos: 21, token: tokenEOF},
			},
		},
		{
			name:  "RegexpEscapedValue",
			input: `name~=[abc]\+,foo=test`,
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "name"},
				{pos: 4, token: tokenOperator, text: "~="},
				{pos: 6, token: tokenValue, text: "[abc]\\+"},
				{pos: 13, token: tokenSeparator, text: ","},
				{pos: 14, token: tokenField, text: "foo"},
				{pos: 17, token: tokenOperator, text: "="},
				{pos: 18, token: tokenValue, text: "test"},
				{pos: 22, token: tokenEOF},
			},
		},
		{
			name:  "Cowsay",
			input: "name~=牛,labels.moo=true",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "name"},
				{pos: 4, token: tokenOperator, text: "~="},
				{pos: 6, token: tokenValue, text: "牛"},
				{pos: 9, token: tokenSeparator, text: ","},
				{pos: 10, token: tokenField, text: "labels"},
				{pos: 16, token: tokenSeparator, text: "."},
				{pos: 17, token: tokenField, text: "moo"},
				{pos: 20, token: tokenOperator, text: "="},
				{pos: 21, token: tokenValue, text: "true"},
				{pos: 25, token: tokenEOF},
			},
		},
		{
			name:  "Escapes",
			input: `name~="asdf\n\tfooo"`,
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "name"},
				{pos: 4, token: tokenOperator, text: "~="},
				{pos: 6, token: tokenQuoted, text: "\"asdf\\n\\tfooo\""},
				{pos: 20, token: tokenEOF},
			},
		},
		{
			name:  "NullInput",
			input: "foo\x00bar",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "foo"},
				{pos: 3, token: tokenIllegal},
				{pos: 4, token: tokenField, text: "bar"},
				{pos: 7, token: tokenEOF},
			},
		},
		{
			name:  "SpacesChomped",
			input: "foo = bar    ",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "foo"},
				{pos: 4, token: tokenOperator, text: "="},
				{pos: 6, token: tokenValue, text: "bar"},
				{pos: 13, token: tokenEOF},
			},
		},
		{
			name:  "ValuesPunctauted",
			input: "compound.labels==punctuated_value.foo-bar",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "compound"},
				{pos: 8, token: tokenSeparator, text: "."},
				{pos: 9, token: tokenField, text: "labels"},
				{pos: 15, token: tokenOperator, text: "=="},
				{pos: 17, token: tokenValue, text: "punctuated_value.foo-bar"},
				{pos: 41, token: tokenEOF},
			},
		},
		{
			name:  "PartialInput",
			input: "interrupted=",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "interrupted"},
				{pos: 11, token: tokenOperator, text: "="},
				{pos: 12, token: tokenEOF},
			},
		},
		{
			name:  "DoubleValue",
			input: "doublevalue=value value",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "doublevalue"},
				{pos: 11, token: tokenOperator, text: "="},
				{pos: 12, token: tokenValue, text: "value"},
				{pos: 18, token: tokenField, text: "value"},
				{pos: 23, token: tokenEOF},
			},
		},
		{
			name:  "LeadingWithQuoted",
			input: `"leading quote".postquote==value`,
			expected: []tokenResult{
				{pos: 0, token: tokenQuoted, text: "\"leading quote\""},
				{pos: 15, token: tokenSeparator, text: "."},
				{pos: 16, token: tokenField, text: "postquote"},
				{pos: 25, token: tokenOperator, text: "=="},
				{pos: 27, token: tokenValue, text: "value"},
				{pos: 32, token: tokenEOF},
			},
		},
		{
			name:  "MissingValue",
			input: "input==,id?=ff",
			expected: []tokenResult{
				{pos: 0, token: tokenField, text: "input"},
				{pos: 5, token: tokenOperator, text: "=="},
				{pos: 7, token: tokenSeparator, text: ","},
				{pos: 8, token: tokenValue, text: "id?=ff"},
				{pos: 14, token: tokenEOF},
			},
		},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			var sc scanner
			sc.init(testcase.input)
			t.Logf("scan %q", testcase.input)

			// If you leave the expected empty, the test case will just print
			// out the token stream, which you can paste into the testcase when
			// adding new cases.
			if len(testcase.expected) == 0 {
				fmt.Println("Name", testcase.name)
			}

			for i := 0; ; i++ {
				pos, tok, s := sc.scan()
				t.Log("token", pos, tok, strconv.Quote(s))
				if len(testcase.expected) == 0 {
					if len(s) > 0 {
						fmt.Printf("{pos: %v, token: %#v, text: %q},\n", pos, tok, s)
					} else {
						fmt.Printf("{pos: %v, token: %#v},\n", pos, tok)
					}
				} else {
					tokv := tokenResult{pos: pos, token: tok, text: s}
					if i >= len(testcase.expected) {
						t.Fatalf("too many tokens parsed")
					}

					if tokv != testcase.expected[i] {
						t.Fatalf("token unexpected: %v != %v", tokv, testcase.expected[i])
					}
				}

				if tok == tokenEOF {
					break
				}
			}

			// make sure we've eof'd
			_, tok, _ := sc.scan()
			if tok != tokenEOF {
				t.Fatal("must consume all input")
			}

			if len(testcase.expected) == 0 {
				t.Fatal("must define expected tokens")
			}
		})
	}
}
