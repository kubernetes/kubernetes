// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"fmt"
	"testing"

	"golang.org/x/text/feature/plural"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func TestWrongVerb(t *testing.T) {
	testCases := []struct {
		f    Formatter
		fmt  string
		want string
	}{{
		f:    Decimal(12),
		fmt:  "%e",
		want: "%!e(int=12)",
	}, {
		f:    Scientific(12),
		fmt:  "%f",
		want: "%!f(int=12)",
	}, {
		f:    Engineering(12),
		fmt:  "%f",
		want: "%!f(int=12)",
	}, {
		f:    Percent(12),
		fmt:  "%e",
		want: "%!e(int=12)",
	}}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			tag := language.Und
			got := message.NewPrinter(tag).Sprintf(tc.fmt, tc.f)
			if got != tc.want {
				t.Errorf("got %q; want %q", got, tc.want)
			}
		})
	}
}

func TestDigits(t *testing.T) {
	testCases := []struct {
		f     Formatter
		scale int
		want  string
	}{{
		f:     Decimal(3),
		scale: 0,
		want:  "digits:[3] exp:1 comma:0 end:1",
	}, {
		f:     Decimal(3.1),
		scale: 0,
		want:  "digits:[3] exp:1 comma:0 end:1",
	}, {
		f:     Scientific(3.1),
		scale: 0,
		want:  "digits:[3] exp:1 comma:1 end:1",
	}, {
		f:     Scientific(3.1),
		scale: 3,
		want:  "digits:[3 1] exp:1 comma:1 end:4",
	}}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			d := tc.f.Digits(nil, language.Croatian, tc.scale)
			got := fmt.Sprintf("digits:%d exp:%d comma:%d end:%d", d.Digits, d.Exp, d.Comma, d.End)
			if got != tc.want {
				t.Errorf("got %v; want %v", got, tc.want)
			}
		})
	}
}

func TestPluralIntegration(t *testing.T) {
	testCases := []struct {
		f    Formatter
		want string
	}{{
		f:    Decimal(1),
		want: "one: 1",
	}, {
		f:    Decimal(5),
		want: "other: 5",
	}}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			message.Set(language.English, "num %f", plural.Selectf(1, "%f",
				"one", "one: %f",
				"other", "other: %f"))

			p := message.NewPrinter(language.English)

			// Indirect the call to p.Sprintf through the variable f
			// to avoid Go tip failing a vet check.
			// TODO: remove once vet check has been fixed. See Issue #22936.
			f := p.Sprintf
			got := f("num %f", tc.f)

			if got != tc.want {
				t.Errorf("got %q; want %q", got, tc.want)
			}
		})
	}
}
