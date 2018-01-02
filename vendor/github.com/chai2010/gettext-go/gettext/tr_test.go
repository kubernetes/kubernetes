// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"testing"

	"github.com/chai2010/gettext-go/gettext/mo"
	"github.com/chai2010/gettext-go/gettext/po"
)

func TestTranslator_Po(t *testing.T) {
	tr, err := newPoTranslator("test", []byte(testTrPoData))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range testTrData {
		if out := tr.PGettext(v.msgctxt, v.msgid); out != v.msgstr {
			t.Fatalf("%s/%s: expect = %s, got = %s", v.msgctxt, v.msgid, v.msgstr, out)
		}
	}
}

func TestTranslator_Mo(t *testing.T) {
	tr, err := newMoTranslator("test", poToMoData(t, []byte(testTrPoData)))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range testTrData {
		if out := tr.PGettext(v.msgctxt, v.msgid); out != v.msgstr {
			t.Fatalf("%s/%s: expect = %s, got = %s", v.msgctxt, v.msgid, v.msgstr, out)
		}
		break
	}
}

func poToMoData(t *testing.T, data []byte) []byte {
	poFile, err := po.LoadData(data)
	if err != nil {
		t.Fatal(err)
	}
	moFile := &mo.File{
		MimeHeader: mo.Header{
			ProjectIdVersion:        poFile.MimeHeader.ProjectIdVersion,
			ReportMsgidBugsTo:       poFile.MimeHeader.ReportMsgidBugsTo,
			POTCreationDate:         poFile.MimeHeader.POTCreationDate,
			PORevisionDate:          poFile.MimeHeader.PORevisionDate,
			LastTranslator:          poFile.MimeHeader.LastTranslator,
			LanguageTeam:            poFile.MimeHeader.LanguageTeam,
			Language:                poFile.MimeHeader.Language,
			MimeVersion:             poFile.MimeHeader.MimeVersion,
			ContentType:             poFile.MimeHeader.ContentType,
			ContentTransferEncoding: poFile.MimeHeader.ContentTransferEncoding,
			PluralForms:             poFile.MimeHeader.PluralForms,
			XGenerator:              poFile.MimeHeader.XGenerator,
			UnknowFields:            poFile.MimeHeader.UnknowFields,
		},
	}
	for _, v := range poFile.Messages {
		moFile.Messages = append(moFile.Messages, mo.Message{
			MsgContext:   v.MsgContext,
			MsgId:        v.MsgId,
			MsgIdPlural:  v.MsgIdPlural,
			MsgStr:       v.MsgStr,
			MsgStrPlural: v.MsgStrPlural,
		})
	}
	return moFile.Data()
}

var testTrData = []struct {
	msgctxt string
	msgid   string
	msgstr  string
}{
	{"main.init", "Gettext in init.", "Init函数中的Gettext."},
	{"main.main", "Hello, world!", "你好, 世界!"},
	{"main.func", "Gettext in func.", "闭包函数中的Gettext."},
	{"code.google.com/p/gettext-go/examples/hi.SayHi", "pkg hi: Hello, world!", "来自\"Hi\"包的问候: 你好, 世界!"},
}

var testTrPoData = `
msgctxt "main.init"
msgid "Gettext in init."
msgstr "Init函数中的Gettext."

msgctxt "main.main"
msgid "Hello, world!"
msgstr "你好, 世界!"

msgctxt "main.func"
msgid "Gettext in func."
msgstr "闭包函数中的Gettext."

msgctxt "code.google.com/p/gettext-go/examples/hi.SayHi"
msgid "pkg hi: Hello, world!"
msgstr "来自\"Hi\"包的问候: 你好, 世界!"
`
