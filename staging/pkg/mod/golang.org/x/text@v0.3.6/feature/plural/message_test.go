// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plural

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/text/internal/catmsg"
	"golang.org/x/text/language"
	"golang.org/x/text/message/catalog"
)

func TestSelect(t *testing.T) {
	lang := language.English
	type test struct {
		arg    interface{}
		result string
		err    string
	}
	testCases := []struct {
		desc  string
		msg   catalog.Message
		err   string
		tests []test
	}{{
		desc: "basic",
		msg:  Selectf(1, "%d", "one", "foo", "other", "bar"),
		tests: []test{
			{arg: 0, result: "bar"},
			{arg: 1, result: "foo"},
			{arg: 2, result: "bar"},
			{arg: opposite(1), result: "bar"},
			{arg: opposite(2), result: "foo"},
			{arg: "unknown", result: "bar"}, // other
		},
	}, {
		desc: "comparisons",
		msg: Selectf(1, "%d",
			"=0", "zero",
			"=1", "one",
			"one", "cannot match", // never matches
			"<5", "<5", // never matches
			"=5", "=5",
			Other, "other"),
		tests: []test{
			{arg: 0, result: "zero"},
			{arg: 1, result: "one"},
			{arg: 2, result: "<5"},
			{arg: 4, result: "<5"},
			{arg: 5, result: "=5"},
			{arg: 6, result: "other"},
			{arg: "unknown", result: "other"},
		},
	}, {
		desc: "fractions",
		msg:  Selectf(1, "%.2f", "one", "foo", "other", "bar"),
		tests: []test{
			// fractions are always plural in english
			{arg: 0, result: "bar"},
			{arg: 1, result: "bar"},
		},
	}, {
		desc: "decimal without fractions",
		msg:  Selectf(1, "%.0f", "one", "foo", "other", "bar"),
		tests: []test{
			// fractions are always plural in english
			{arg: 0, result: "bar"},
			{arg: 1, result: "foo"},
		},
	}, {
		desc: "scientific",
		msg:  Selectf(1, "%.0e", "one", "foo", "other", "bar"),
		tests: []test{
			{arg: 0, result: "bar"},
			{arg: 1, result: "foo"},
		},
	}, {
		desc: "variable",
		msg:  Selectf(1, "%.1g", "one", "foo", "other", "bar"),
		tests: []test{
			// fractions are always plural in english
			{arg: 0, result: "bar"},
			{arg: 1, result: "foo"},
			{arg: 2, result: "bar"},
		},
	}, {
		desc: "default",
		msg:  Selectf(1, "", "one", "foo", "other", "bar"),
		tests: []test{
			{arg: 0, result: "bar"},
			{arg: 1, result: "foo"},
			{arg: 2, result: "bar"},
			{arg: 1.0, result: "bar"},
		},
	}, {
		desc: "nested",
		msg:  Selectf(1, "", "other", Selectf(2, "", "one", "foo", "other", "bar")),
		tests: []test{
			{arg: 0, result: "bar"},
			{arg: 1, result: "foo"},
			{arg: 2, result: "bar"},
		},
	}, {
		desc:  "arg unavailable",
		msg:   Selectf(100, "%.2f", "one", "foo", "other", "bar"),
		tests: []test{{arg: 1, result: "bar"}},
	}, {
		desc:  "no match",
		msg:   Selectf(1, "%.2f", "one", "foo"),
		tests: []test{{arg: 0, result: "bar", err: catmsg.ErrNoMatch.Error()}},
	}, {
		desc: "error invalid form",
		err:  `invalid plural form "excessive"`,
		msg:  Selectf(1, "%d", "excessive", "foo"),
	}, {
		desc: "error form not used by language",
		err:  `form "many" not supported for language "en"`,
		msg:  Selectf(1, "%d", "many", "foo"),
	}, {
		desc: "error invalid selector",
		err:  `selector of type int; want string or Form`,
		msg:  Selectf(1, "%d", 1, "foo"),
	}, {
		desc: "error missing message",
		err:  `no message defined for selector one`,
		msg:  Selectf(1, "%d", "one"),
	}, {
		desc: "error invalid number",
		err:  `invalid number in selector "<1.00"`,
		msg:  Selectf(1, "%d", "<1.00"),
	}, {
		desc: "error empty selector",
		err:  `empty selector`,
		msg:  Selectf(1, "%d", "", "foo"),
	}, {
		desc: "error invalid message",
		err:  `message of type int; must be string or catalog.Message`,
		msg:  Selectf(1, "%d", "one", 3),
	}, {
		desc: "nested error",
		err:  `empty selector`,
		msg:  Selectf(1, "", "other", Selectf(2, "", "")),
	}}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			data, err := catmsg.Compile(lang, nil, tc.msg)
			chkError(t, err, tc.err)
			for _, tx := range tc.tests {
				t.Run(fmt.Sprint(tx.arg), func(t *testing.T) {
					r := renderer{arg: tx.arg}
					d := catmsg.NewDecoder(lang, &r, nil)
					err := d.Execute(data)
					chkError(t, err, tx.err)
					if r.result != tx.result {
						t.Errorf("got %q; want %q", r.result, tx.result)
					}
				})
			}
		})
	}
}

func chkError(t *testing.T, got error, want string) {
	if (got == nil && want != "") ||
		(got != nil && (want == "" || !strings.Contains(got.Error(), want))) {
		t.Fatalf("got %v; want %v", got, want)
	}
	if got != nil {
		t.SkipNow()
	}
}

type renderer struct {
	arg    interface{}
	result string
}

func (r *renderer) Render(s string) { r.result += s }
func (r *renderer) Arg(i int) interface{} {
	if i > 10 { // Allow testing "arg unavailable" path
		return nil
	}
	return r.arg
}

type opposite int

func (o opposite) PluralForm(lang language.Tag, scale int) (Form, int) {
	if o == 1 {
		return Other, 1
	}
	return One, int(o)
}
