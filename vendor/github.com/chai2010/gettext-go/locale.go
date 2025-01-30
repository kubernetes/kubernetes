// Copyright 2020 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"fmt"
	"sync"
)

type _Locale struct {
	mutex     sync.Mutex
	fs        FileSystem
	lang      string
	domain    string
	trMap     map[string]*translator
	trCurrent *translator
}

var _ Gettexter = (*_Locale)(nil)

func newLocale(domain, path string, data ...interface{}) *_Locale {
	if domain == "" {
		domain = "default"
	}
	p := &_Locale{
		lang:   DefaultLanguage,
		domain: domain,
	}
	if len(data) > 0 {
		p.fs = NewFS(path, data[0])
	} else {
		p.fs = NewFS(path, nil)
	}

	p.syncTrMap()
	return p
}

func (p *_Locale) makeTrMapKey(domain, _Locale string) string {
	return domain + "_$$$_" + _Locale
}

func (p *_Locale) FileSystem() FileSystem {
	return p.fs
}

func (p *_Locale) GetLanguage() string {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	return p.lang
}
func (p *_Locale) SetLanguage(lang string) Gettexter {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	if lang == "" {
		lang = DefaultLanguage
	}
	if lang == p.lang {
		return p
	}

	p.lang = lang
	p.syncTrMap()
	return p
}

func (p *_Locale) GetDomain() string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.domain
}

func (p *_Locale) SetDomain(domain string) Gettexter {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	if domain == "" || domain == p.domain {
		return p
	}

	p.domain = domain
	p.syncTrMap()
	return p
}

func (p *_Locale) syncTrMap() {
	p.trMap = make(map[string]*translator)
	trMapKey := p.makeTrMapKey(p.domain, p.lang)

	if tr, ok := p.trMap[trMapKey]; ok {
		p.trCurrent = tr
		return
	}

	// try load po file
	if data, err := p.fs.LoadMessagesFile(p.domain, p.lang, ".po"); err == nil {
		if tr, err := newPoTranslator(fmt.Sprintf("%s_%s.po", p.domain, p.lang), data); err == nil {
			p.trMap[trMapKey] = tr
			p.trCurrent = tr
			return
		}
	}

	// try load mo file
	if data, err := p.fs.LoadMessagesFile(p.domain, p.lang, ".mo"); err == nil {
		if tr, err := newMoTranslator(fmt.Sprintf("%s_%s.mo", p.domain, p.lang), data); err == nil {
			p.trMap[trMapKey] = tr
			p.trCurrent = tr
			return
		}
	}

	// try load json file
	if data, err := p.fs.LoadMessagesFile(p.domain, p.lang, ".json"); err == nil {
		if tr, err := newJsonTranslator(p.lang, fmt.Sprintf("%s_%s.json", p.domain, p.lang), data); err == nil {
			p.trMap[trMapKey] = tr
			p.trCurrent = tr
			return
		}
	}

	// no po/mo file
	p.trMap[trMapKey] = nilTranslator
	p.trCurrent = nilTranslator
	return
}

func (p *_Locale) Gettext(msgid string) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.trCurrent.PGettext("", msgid)
}

func (p *_Locale) PGettext(msgctxt, msgid string) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.trCurrent.PGettext(msgctxt, msgid)
}

func (p *_Locale) NGettext(msgid, msgidPlural string, n int) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.trCurrent.PNGettext("", msgid, msgidPlural, n)
}

func (p *_Locale) PNGettext(msgctxt, msgid, msgidPlural string, n int) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.trCurrent.PNGettext(msgctxt, msgid, msgidPlural, n)
}

func (p *_Locale) DGettext(domain, msgid string) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.gettext(domain, "", msgid, "", 0)
}

func (p *_Locale) DNGettext(domain, msgid, msgidPlural string, n int) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.gettext(domain, "", msgid, msgidPlural, n)
}

func (p *_Locale) DPGettext(domain, msgctxt, msgid string) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.gettext(domain, msgctxt, msgid, "", 0)
}

func (p *_Locale) DPNGettext(domain, msgctxt, msgid, msgidPlural string, n int) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.gettext(domain, msgctxt, msgid, msgidPlural, n)
}

func (p *_Locale) Getdata(name string) []byte {
	return p.getdata(p.domain, name)
}

func (p *_Locale) DGetdata(domain, name string) []byte {
	return p.getdata(domain, name)
}

func (p *_Locale) gettext(domain, msgctxt, msgid, msgidPlural string, n int) string {
	if f, ok := p.trMap[p.makeTrMapKey(domain, p.lang)]; ok {
		return f.PNGettext(msgctxt, msgid, msgidPlural, n)
	}
	return msgid
}

func (p *_Locale) getdata(domain, name string) []byte {
	if data, err := p.fs.LoadResourceFile(domain, p.lang, name); err == nil {
		return data
	}
	if p.lang != "default" {
		if data, err := p.fs.LoadResourceFile(domain, "default", name); err == nil {
			return data
		}
	}
	return nil
}
