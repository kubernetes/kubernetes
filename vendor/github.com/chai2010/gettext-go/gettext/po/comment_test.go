// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package po

import (
	"reflect"
	"testing"
)

func TestPoComment(t *testing.T) {
	var x Comment
	for i := 0; i < len(testPoComments); i++ {
		if i != 2 {
			continue
		}
		err := x.readPoComment(newLineReader(testPoComments[i].Data))
		if err != nil {
			t.Fatalf("%d: %v", i, err)
		}
		x.StartLine = 0 // ingore comment line
		if !reflect.DeepEqual(&x, &testPoComments[i].PoComment) {
			t.Logf("expect(%d):\n", i)
			t.Logf("\n%v\n", &testPoComments[i].PoComment)
			t.Logf("got(%d):\n", i)
			t.Logf("\n%v\n", &x)
			t.FailNow()
		}
		if testPoComments[i].CheckStringer {
			s := testPoComments[i].PoComment.String()
			if s != testPoComments[i].Data {
				t.Logf("expect(%d):\n", i)
				t.Logf("\n%s\n", testPoComments[i].Data)
				t.Logf("got(%d):\n", i)
				t.Logf("\n%s\n", testPoComments[i].PoComment.String())
				t.FailNow()
			}
		}
	}
}

type testPoComment struct {
	CheckStringer bool
	Data          string
	PoComment     Comment
}

var testPoComments = []testPoComment{

	// --------------------------------------------------------------
	// CheckStringer: true
	// --------------------------------------------------------------

	testPoComment{
		CheckStringer: true,
		Data: `# translator comments
`,
		PoComment: Comment{
			TranslatorComment: `translator comments`,
		},
	},
	testPoComment{
		CheckStringer: true,
		Data: `# translator comments
`,
		PoComment: Comment{
			TranslatorComment: `translator comments`,
		},
	},

	testPoComment{
		CheckStringer: true,
		Data: `# translator-comments
# bad comment
#. extracted-comments
#: src/msgcmp.c:338 src/po-lex.c:699 src/msg.c:123
#, fuzzy, c-format, range:0..10
#| msgctxt ""
#| "previous-context1\n"
#| "previous-context2"
#| msgid ""
#| "previous-untranslated-string1\n"
#| "previous-untranslated-string2"
`,
		PoComment: Comment{
			TranslatorComment: "translator-comments\nbad comment",
			ExtractedComment:  "extracted-comments",
			ReferenceFile:     []string{"src/msgcmp.c", "src/po-lex.c", "src/msg.c"},
			ReferenceLine:     []int{338, 699, 123},
			Flags:             []string{"fuzzy", "c-format", "range:0..10"},
			PrevMsgContext:    "previous-context1\nprevious-context2",
			PrevMsgId:         "previous-untranslated-string1\nprevious-untranslated-string2",
		},
	},

	// --------------------------------------------------------------
	// CheckStringer: false
	// --------------------------------------------------------------

	testPoComment{
		CheckStringer: false,
		Data: `
#  translator-comments
#bad comment
#. extracted-comments
#: src/msgcmp.c:338 src/po-lex.c:699
#: src/msg.c:123
#, fuzzy,c-format,range:0..10
#| msgctxt ""
#| "previous-context1\n"
#| "previous-context2"
#| msgid ""
#| "previous-untranslated-string1\n"
#| "previous-untranslated-string2"
`,
		PoComment: Comment{
			TranslatorComment: "translator-comments\nbad comment",
			ExtractedComment:  "extracted-comments",
			ReferenceFile:     []string{"src/msgcmp.c", "src/po-lex.c", "src/msg.c"},
			ReferenceLine:     []int{338, 699, 123},
			Flags:             []string{"fuzzy", "c-format", "range:0..10"},
			PrevMsgContext:    "previous-context1\nprevious-context2",
			PrevMsgId:         "previous-untranslated-string1\nprevious-untranslated-string2",
		},
	},
	testPoComment{
		CheckStringer: false,
		Data: `
# SOME DESCRIPTIVE TITLE.
# Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER
# This file is distributed under the same license as the PACKAGE package.
# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
#
msgid ""
msgstr ""
"Project-Id-Version: Poedit 1.5\n"
"Report-Msgid-Bugs-To: poedit@googlegroups.com\n"
"POT-Creation-Date: 2012-07-30 10:34+0200\n"
"PO-Revision-Date: 2013-12-25 09:32+0800\n"
"Last-Translator: chai2010 <chaishushan@gmail.com>\n"
"Language-Team: \n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"
"Plural-Forms: nplurals=1; plural=0;\n"
"X-Generator: Poedit 1.5.7\n"
`,
		PoComment: Comment{
			TranslatorComment: `SOME DESCRIPTIVE TITLE.
Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER
This file is distributed under the same license as the PACKAGE package.
FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
`,
		},
	},
	testPoComment{
		CheckStringer: false,
		Data: `
#. TRANSLATORS: This is version information in about dialog, it is followed
#. by version number when used (wxWidgets 2.8)
#: ../src/edframe.cpp:2431
#| msgctxt "previous-context asdasd"
"asdad \n asdsad"
msgstr ""
`,
		PoComment: Comment{
			ExtractedComment: `TRANSLATORS: This is version information in about dialog, it is followed
by version number when used (wxWidgets 2.8)`,
			ReferenceFile:  []string{"../src/edframe.cpp"},
			ReferenceLine:  []int{2431},
			PrevMsgContext: "previous-context asdasd",
		},
	},
	testPoComment{
		CheckStringer: false,
		Data: `
#: tst-gettext2.c:33
msgid "First string for testing."
msgstr "Lang1: 1st string"
`,
		PoComment: Comment{
			ReferenceFile: []string{"tst-gettext2.c"},
			ReferenceLine: []int{33},
		},
	},
	testPoComment{
		CheckStringer: false,
		Data: `
#: app/app_procs.c:307
#, fuzzy, c-format
msgid "Can't find output format %s\n"
msgstr ""
"敲矾弊牢 '%s'甫 佬阑荐 绝嚼聪促\n"
"%s"
`,
		PoComment: Comment{
			ReferenceFile: []string{"app/app_procs.c"},
			ReferenceLine: []int{307},
			Flags:         []string{"fuzzy", "c-format"},
		},
	},

	// --------------------------------------------------------------
	// END
	// --------------------------------------------------------------
}
