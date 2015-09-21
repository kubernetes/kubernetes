package text

import (
	"bytes"
	"testing"
)

type T struct {
	inp, exp, pre string
}

var tests = []T{
	{
		"The quick brown fox\njumps over the lazy\ndog.\nBut not quickly.\n",
		"xxxThe quick brown fox\nxxxjumps over the lazy\nxxxdog.\nxxxBut not quickly.\n",
		"xxx",
	},
	{
		"The quick brown fox\njumps over the lazy\ndog.\n\nBut not quickly.",
		"xxxThe quick brown fox\nxxxjumps over the lazy\nxxxdog.\n\nxxxBut not quickly.",
		"xxx",
	},
}

func TestIndent(t *testing.T) {
	for _, test := range tests {
		got := Indent(test.inp, test.pre)
		if got != test.exp {
			t.Errorf("mismatch %q != %q", got, test.exp)
		}
	}
}

type IndentWriterTest struct {
	inp, exp string
	pre      []string
}

var ts = []IndentWriterTest{
	{
		`
The quick brown fox
jumps over the lazy
dog.
But not quickly.
`[1:],
		`
xxxThe quick brown fox
xxxjumps over the lazy
xxxdog.
xxxBut not quickly.
`[1:],
		[]string{"xxx"},
	},
	{
		`
The quick brown fox
jumps over the lazy
dog.
But not quickly.
`[1:],
		`
xxaThe quick brown fox
xxxjumps over the lazy
xxxdog.
xxxBut not quickly.
`[1:],
		[]string{"xxa", "xxx"},
	},
	{
		`
The quick brown fox
jumps over the lazy
dog.
But not quickly.
`[1:],
		`
xxaThe quick brown fox
xxbjumps over the lazy
xxcdog.
xxxBut not quickly.
`[1:],
		[]string{"xxa", "xxb", "xxc", "xxx"},
	},
	{
		`
The quick brown fox
jumps over the lazy
dog.

But not quickly.`[1:],
		`
xxaThe quick brown fox
xxxjumps over the lazy
xxxdog.
xxx
xxxBut not quickly.`[1:],
		[]string{"xxa", "xxx"},
	},
}

func TestIndentWriter(t *testing.T) {
	for _, test := range ts {
		b := new(bytes.Buffer)
		pre := make([][]byte, len(test.pre))
		for i := range test.pre {
			pre[i] = []byte(test.pre[i])
		}
		w := NewIndentWriter(b, pre...)
		if _, err := w.Write([]byte(test.inp)); err != nil {
			t.Error(err)
		}
		if got := b.String(); got != test.exp {
			t.Errorf("mismatch %q != %q", got, test.exp)
			t.Log(got)
			t.Log(test.exp)
		}
	}
}
