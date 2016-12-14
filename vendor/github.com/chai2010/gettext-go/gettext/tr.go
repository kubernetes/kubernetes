// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"github.com/chai2010/gettext-go/gettext/mo"
	"github.com/chai2010/gettext-go/gettext/plural"
	"github.com/chai2010/gettext-go/gettext/po"
)

var nilTranslator = &translator{
	MessageMap:    make(map[string]mo.Message),
	PluralFormula: plural.Formula("??"),
}

type translator struct {
	MessageMap    map[string]mo.Message
	PluralFormula func(n int) int
}

func newMoTranslator(name string, data []byte) (*translator, error) {
	var (
		f   *mo.File
		err error
	)
	if len(data) != 0 {
		f, err = mo.LoadData(data)
	} else {
		f, err = mo.Load(name)
	}
	if err != nil {
		return nil, err
	}
	var tr = &translator{
		MessageMap: make(map[string]mo.Message),
	}
	for _, v := range f.Messages {
		tr.MessageMap[tr.makeMapKey(v.MsgContext, v.MsgId)] = v
	}
	if lang := f.MimeHeader.Language; lang != "" {
		tr.PluralFormula = plural.Formula(lang)
	} else {
		tr.PluralFormula = plural.Formula("??")
	}
	return tr, nil
}

func newPoTranslator(name string, data []byte) (*translator, error) {
	var (
		f   *po.File
		err error
	)
	if len(data) != 0 {
		f, err = po.LoadData(data)
	} else {
		f, err = po.Load(name)
	}
	if err != nil {
		return nil, err
	}
	var tr = &translator{
		MessageMap: make(map[string]mo.Message),
	}
	for _, v := range f.Messages {
		tr.MessageMap[tr.makeMapKey(v.MsgContext, v.MsgId)] = mo.Message{
			MsgContext:   v.MsgContext,
			MsgId:        v.MsgId,
			MsgIdPlural:  v.MsgIdPlural,
			MsgStr:       v.MsgStr,
			MsgStrPlural: v.MsgStrPlural,
		}
	}
	if lang := f.MimeHeader.Language; lang != "" {
		tr.PluralFormula = plural.Formula(lang)
	} else {
		tr.PluralFormula = plural.Formula("??")
	}
	return tr, nil
}

func (p *translator) PGettext(msgctxt, msgid string) string {
	return p.PNGettext(msgctxt, msgid, "", 0)
}

func (p *translator) PNGettext(msgctxt, msgid, msgidPlural string, n int) string {
	n = p.PluralFormula(n)
	if ss := p.findMsgStrPlural(msgctxt, msgid, msgidPlural); len(ss) != 0 {
		if n >= len(ss) {
			n = len(ss) - 1
		}
		if ss[n] != "" {
			return ss[n]
		}
	}
	if msgidPlural != "" && n > 0 {
		return msgidPlural
	}
	return msgid
}

func (p *translator) findMsgStrPlural(msgctxt, msgid, msgidPlural string) []string {
	key := p.makeMapKey(msgctxt, msgid)
	if v, ok := p.MessageMap[key]; ok {
		if len(v.MsgIdPlural) != 0 {
			if len(v.MsgStrPlural) != 0 {
				return v.MsgStrPlural
			} else {
				return nil
			}
		} else {
			if len(v.MsgStr) != 0 {
				return []string{v.MsgStr}
			} else {
				return nil
			}
		}
	}
	return nil
}

func (p *translator) makeMapKey(msgctxt, msgid string) string {
	if msgctxt != "" {
		return msgctxt + mo.EotSeparator + msgid
	}
	return msgid
}
