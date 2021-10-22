// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"path"
	"testing"

	"golang.org/x/text/message"
)

func TestCatalog(t *testing.T) {
	args := func(a ...interface{}) []interface{} { return a }
	testCases := []struct {
		lang string
		key  string
		args []interface{}
		want string
	}{{
		lang: "en",
		key:  "Hello world!\n",
		want: "Hello world!\n",
	}, {
		lang: "en",
		key:  "non-existing-key\n",
		want: "non-existing-key\n",
	}, {
		lang: "de",
		key:  "Hello world!\n",
		want: "Hallo Welt!\n",
	}, {
		lang: "en",
		key:  "%d more files remaining!",
		args: args(1),
		want: "One file remaining!",
	}, {
		lang: "en-u-nu-fullwide",
		key:  "%d more files remaining!",
		args: args(5),
		want: "There are ５ more files remaining!",
	}}
	for _, tc := range testCases {
		t.Run(path.Join(tc.lang, tc.key), func(t *testing.T) {
			p := message.NewPrinter(message.MatchLanguage(tc.lang))
			got := p.Sprintf(tc.key, tc.args...)
			if got != tc.want {
				t.Errorf("got %q; want %q", got, tc.want)
			}
		})
	}
}
