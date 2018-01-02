// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mo

import (
	"testing"
)

func TestDecodePoString(t *testing.T) {
	if s := decodePoString(poStrEncode); s != poStrDecode {
		t.Fatalf(`expect = %s got = %s`, poStrDecode, s)
	}
}

func TestEncodePoString(t *testing.T) {
	if s := encodePoString(poStrDecode); s != poStrEncodeStd {
		t.Fatalf(`expect = %s; got = %s`, poStrEncodeStd, s)
	}
}

const poStrEncode = `# noise
123456789
"Project-Id-Version: Poedit 1.5\n"
"Report-Msgid-Bugs-To: poedit@googlegroups.com\n"
"POT-Creation-Date: 2012-07-30 10:34+0200\n"
"PO-Revision-Date: 2013-02-24 21:00+0800\n"
"Last-Translator: Christopher Meng <trans@cicku.me>\n"
"Language-Team: \n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"
"Plural-Forms: nplurals=1; plural=0;\n"
"X-Generator: Poedit 1.5.5\n"
"TestPoString: abc"
"123\n"
>>
123456???
`

const poStrEncodeStd = `"Project-Id-Version: Poedit 1.5\n"
"Report-Msgid-Bugs-To: poedit@googlegroups.com\n"
"POT-Creation-Date: 2012-07-30 10:34+0200\n"
"PO-Revision-Date: 2013-02-24 21:00+0800\n"
"Last-Translator: Christopher Meng <trans@cicku.me>\n"
"Language-Team: \n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"
"Plural-Forms: nplurals=1; plural=0;\n"
"X-Generator: Poedit 1.5.5\n"
"TestPoString: abc123\n"
`

const poStrDecode = `Project-Id-Version: Poedit 1.5
Report-Msgid-Bugs-To: poedit@googlegroups.com
POT-Creation-Date: 2012-07-30 10:34+0200
PO-Revision-Date: 2013-02-24 21:00+0800
Last-Translator: Christopher Meng <trans@cicku.me>
Language-Team: 
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=1; plural=0;
X-Generator: Poedit 1.5.5
TestPoString: abc123
`
