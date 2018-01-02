// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"reflect"
	"testing"
)

var testDataDir = "../testdata/"

var testPoMoFiles = []struct {
	poFile string
	moFile string
}{
	{"gettext-3-1.po", "gettext-3-1.mo"},
	{"gettext-4.po", "gettext-4.mo"},
	{"gettext-5.po", "gettext-5.mo"},
	{"gettext-6-1.po", "gettext-6-1.mo"},
	{"gettext-6-2.po", "gettext-6-2.mo"},
	{"gettext-7.po", "gettext-7.mo"},
	{"gettextpo-1.de.po", "gettextpo-1.de.mo"},
	{"mm-ko-comp.euc-kr.po", "mm-ko-comp.euc-kr.mo"},
	{"mm-ko.euc-kr.po", "mm-ko.euc-kr.mo"},
	{"mm-viet.comp.po", "mm-viet.comp.mo"},
	{"poedit-1.5.7-zh_CN.po", "poedit-1.5.7-zh_CN.mo"},
	{"qttest2_de.po", "qttest2_de.mo"},
	{"qttest_pl.po", "qttest_pl.mo"},
	{"test.po", "test.mo"},
}

func TestPoMoFiles(t *testing.T) {
	for i := 0; i < len(testPoMoFiles); i++ {
		poName := testPoMoFiles[i].poFile
		moName := testPoMoFiles[i].moFile
		po, err := newPoTranslator(testDataDir+poName, nil)
		if err != nil {
			t.Fatalf("%s: %v", poName, err)
		}
		mo, err := newMoTranslator(testDataDir+moName, nil)
		if err != nil {
			t.Fatalf("%s: %v", poName, err)
		}
		// if no translate, the mo will drop the message.
		// so len(mo) may less than len(po).
		if a, b := len(po.MessageMap), len(mo.MessageMap); a != b {
			t.Logf("%s: %v, %d != %d", poName, "size not equal", a, b)
		}
		for k, v0 := range po.MessageMap {
			v1, ok := mo.MessageMap[k]
			if !ok {
				t.Logf("%s: %q: missing", poName, v0.MsgId)
				continue
			}
			if !reflect.DeepEqual(&v0, &v1) {
				t.Fatalf("%s: %q: expect = %v, got = %v", poName, v0.MsgId, v0, v1)
			}
		}
	}
}
