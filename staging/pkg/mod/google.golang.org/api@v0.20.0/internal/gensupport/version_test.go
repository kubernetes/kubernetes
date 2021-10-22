// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import "testing"

func TestGoVer(t *testing.T) {
	for _, tst := range []struct {
		in, want string
	}{
		{"go1.8", "1.8.0"},
		{"go1.7.3", "1.7.3"},
		{"go1.8.typealias", "1.8.0-typealias"},
		{"go1.8beta1", "1.8.0-beta1"},
		{"go1.8rc2", "1.8.0-rc2"},
		{"devel +824f981dd4b7 Tue Apr 29 21:41:54 2014 -0400", "824f981dd4b7"},
		{"foo bar zipzap", ""},
	} {
		if got := goVer(tst.in); got != tst.want {
			t.Errorf("goVer(%q) = %q, want %q", tst.in, got, tst.want)
		}
	}
}
