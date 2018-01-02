// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plural

import (
	"testing"
)

func TestFormula(t *testing.T) {
	for i, v := range testData {
		if out := Formula(v.lang)(v.in); out != v.out {
			t.Fatalf("%d/%s: expect = %d, got = %d", i, v.lang, v.out, out)
		}
	}
}

var testData = []struct {
	lang string
	in   int
	out  int
}{
	{"#@", 0, 0},
	{"#@", 1, 0},
	{"#@", 10, 0},
	{"#@", -1, 0},

	{"zh", 0, 0},
	{"zh", 1, 0},
	{"zh", 10, 0},
	{"zh", -1, 0},

	{"zh_CN", 0, 0},
	{"zh_CN", 1, 0},
	{"zh_CN", 10, 0},
	{"zh_CN", -1, 0},

	{"en", 0, 0},
	{"en", 1, 0},
	{"en", 2, 1},
	{"en", 10, 1},
	{"en", -1, 0},

	{"en_US", 0, 0},
	{"en_US", 1, 0},
	{"en_US", 2, 1},
	{"en_US", 10, 1},
	{"en_US", -1, 0},
}
