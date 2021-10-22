// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package catmsg

import (
	"errors"
	"strings"
	"testing"

	"golang.org/x/text/language"
)

type renderer struct {
	args   []int
	result string
}

func (r *renderer) Arg(i int) interface{} {
	if i >= len(r.args) {
		return nil
	}
	return r.args[i]
}

func (r *renderer) Render(s string) {
	if r.result != "" {
		r.result += "|"
	}
	r.result += s
}

func TestCodec(t *testing.T) {
	type test struct {
		args   []int
		out    string
		decErr string
	}
	single := func(out, err string) []test { return []test{{out: out, decErr: err}} }
	testCases := []struct {
		desc   string
		m      Message
		enc    string
		encErr string
		tests  []test
	}{{
		desc:   "unused variable",
		m:      &Var{"name", String("foo")},
		encErr: errIsVar.Error(),
		tests:  single("", ""),
	}, {
		desc:  "empty",
		m:     empty{},
		tests: single("", ""),
	}, {
		desc:  "sequence with empty",
		m:     seq{empty{}},
		tests: single("", ""),
	}, {
		desc:  "raw string",
		m:     Raw("foo"),
		tests: single("foo", ""),
	}, {
		desc:  "raw string no sub",
		m:     Raw("${foo}"),
		enc:   "\x02${foo}",
		tests: single("${foo}", ""),
	}, {
		desc:  "simple string",
		m:     String("foo"),
		tests: single("foo", ""),
	}, {
		desc:  "affix",
		m:     &Affix{String("foo"), "\t", "\n"},
		tests: single("\t|foo|\n", ""),
	}, {
		desc:   "missing var",
		m:      String("foo${bar}"),
		enc:    "\x03\x03foo\x02\x03bar",
		encErr: `unknown var "bar"`,
		tests:  single("foo|bar", ""),
	}, {
		desc: "empty var",
		m: seq{
			&Var{"bar", seq{}},
			String("foo${bar}"),
		},
		enc: "\x00\x05\x04\x02bar\x03\x03foo\x00\x00",
		// TODO: recognize that it is cheaper to substitute bar.
		tests: single("foo|bar", ""),
	}, {
		desc: "var after value",
		m: seq{
			String("foo${bar}"),
			&Var{"bar", String("baz")},
		},
		encErr: errIsVar.Error(),
		tests:  single("foo|bar", ""),
	}, {
		desc: "substitution",
		m: seq{
			&Var{"bar", String("baz")},
			String("foo${bar}"),
		},
		tests: single("foo|baz", ""),
	}, {
		desc: "affix with substitution",
		m: &Affix{seq{
			&Var{"bar", String("baz")},
			String("foo${bar}"),
		}, "\t", "\n"},
		tests: single("\t|foo|baz|\n", ""),
	}, {
		desc: "shadowed variable",
		m: seq{
			&Var{"bar", String("baz")},
			seq{
				&Var{"bar", String("BAZ")},
				String("foo${bar}"),
			},
		},
		tests: single("foo|BAZ", ""),
	}, {
		desc:  "nested value",
		m:     nestedLang{nestedLang{empty{}}},
		tests: single("nl|nl", ""),
	}, {
		desc: "not shadowed variable",
		m: seq{
			&Var{"bar", String("baz")},
			seq{
				String("foo${bar}"),
				&Var{"bar", String("BAZ")},
			},
		},
		encErr: errIsVar.Error(),
		tests:  single("foo|baz", ""),
	}, {
		desc: "duplicate variable",
		m: seq{
			&Var{"bar", String("baz")},
			&Var{"bar", String("BAZ")},
			String("${bar}"),
		},
		encErr: "catmsg: duplicate variable \"bar\"",
		tests:  single("baz", ""),
	}, {
		desc: "complete incomplete variable",
		m: seq{
			&Var{"bar", incomplete{}},
			String("${bar}"),
		},
		enc: "\x00\t\b\x01\x01\x14\x04\x02bar\x03\x00\x00\x00",
		// TODO: recognize that it is cheaper to substitute bar.
		tests: single("bar", ""),
	}, {
		desc: "incomplete sequence",
		m: seq{
			incomplete{},
			incomplete{},
		},
		encErr: ErrIncomplete.Error(),
		tests:  single("", ErrNoMatch.Error()),
	}, {
		desc: "compile error variable",
		m: seq{
			&Var{"bar", errorCompileMsg{}},
			String("${bar}"),
		},
		encErr: errCompileTest.Error(),
		tests:  single("bar", ""),
	}, {
		desc:   "compile error message",
		m:      errorCompileMsg{},
		encErr: errCompileTest.Error(),
		tests:  single("", ""),
	}, {
		desc: "compile error sequence",
		m: seq{
			errorCompileMsg{},
			errorCompileMsg{},
		},
		encErr: errCompileTest.Error(),
		tests:  single("", ""),
	}, {
		desc:  "macro",
		m:     String("${exists(1)}"),
		tests: single("you betya!", ""),
	}, {
		desc:  "macro incomplete",
		m:     String("${incomplete(1)}"),
		enc:   "\x03\x00\x01\nincomplete\x01",
		tests: single("incomplete", ""),
	}, {
		desc:  "macro undefined at end",
		m:     String("${undefined(1)}"),
		enc:   "\x03\x00\x01\tundefined\x01",
		tests: single("undefined", "catmsg: undefined macro \"undefined\""),
	}, {
		desc:  "macro undefined with more text following",
		m:     String("${undefined(1)}."),
		enc:   "\x03\x00\x01\tundefined\x01\x01.",
		tests: single("undefined|.", "catmsg: undefined macro \"undefined\""),
	}, {
		desc:   "macro missing paren",
		m:      String("${missing(1}"),
		encErr: "catmsg: missing ')'",
		tests:  single("$!(MISSINGPAREN)", ""),
	}, {
		desc:   "macro bad num",
		m:      String("aa${bad(a)}"),
		encErr: "catmsg: invalid number \"a\"",
		tests:  single("aa$!(BADNUM)", ""),
	}, {
		desc:   "var missing brace",
		m:      String("a${missing"),
		encErr: "catmsg: missing '}'",
		tests:  single("a$!(MISSINGBRACE)", ""),
	}}
	r := &renderer{}
	dec := NewDecoder(language.Und, r, macros)
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			// Use a language other than Und so that we can test
			// passing the language to nested values.
			data, err := Compile(language.Dutch, macros, tc.m)
			if failErr(err, tc.encErr) {
				t.Errorf("encoding error: got %+q; want %+q", err, tc.encErr)
			}
			if tc.enc != "" && data != tc.enc {
				t.Errorf("encoding: got %+q; want %+q", data, tc.enc)
			}
			for _, st := range tc.tests {
				t.Run("", func(t *testing.T) {
					*r = renderer{args: st.args}
					if err = dec.Execute(data); failErr(err, st.decErr) {
						t.Errorf("decoding error: got %+q; want %+q", err, st.decErr)
					}
					if r.result != st.out {
						t.Errorf("decode: got %+q; want %+q", r.result, st.out)
					}
				})
			}
		})
	}
}

func failErr(got error, want string) bool {
	if got == nil {
		return want != ""
	}
	return want == "" || !strings.Contains(got.Error(), want)
}

type seq []Message

func (s seq) Compile(e *Encoder) (err error) {
	err = ErrIncomplete
	e.EncodeMessageType(msgFirst)
	for _, m := range s {
		// Pass only the last error, but allow erroneous or complete messages
		// here to allow testing different scenarios.
		err = e.EncodeMessage(m)
	}
	return err
}

type empty struct{}

func (empty) Compile(e *Encoder) (err error) { return nil }

var msgIncomplete = Register(
	"golang.org/x/text/internal/catmsg.incomplete",
	func(d *Decoder) bool { return false })

type incomplete struct{}

func (incomplete) Compile(e *Encoder) (err error) {
	e.EncodeMessageType(msgIncomplete)
	return ErrIncomplete
}

var msgNested = Register(
	"golang.org/x/text/internal/catmsg.nested",
	func(d *Decoder) bool {
		d.Render(d.DecodeString())
		d.ExecuteMessage()
		return true
	})

type nestedLang struct{ Message }

func (n nestedLang) Compile(e *Encoder) (err error) {
	e.EncodeMessageType(msgNested)
	e.EncodeString(e.Language().String())
	e.EncodeMessage(n.Message)
	return nil
}

type errorCompileMsg struct{}

var errCompileTest = errors.New("catmsg: compile error test")

func (errorCompileMsg) Compile(e *Encoder) (err error) {
	return errCompileTest
}

type dictionary struct{}

var (
	macros       = dictionary{}
	dictMessages = map[string]string{
		"exists":     compile(String("you betya!")),
		"incomplete": compile(incomplete{}),
	}
)

func (d dictionary) Lookup(key string) (data string, ok bool) {
	data, ok = dictMessages[key]
	return
}

func compile(m Message) (data string) {
	data, _ = Compile(language.Und, macros, m)
	return data
}
