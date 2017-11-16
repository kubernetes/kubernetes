// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"sync"
)

type domainManager struct {
	mutex     sync.Mutex
	locale    string
	domain    string
	domainMap map[string]*fileSystem
	trTextMap map[string]*translator
}

func newDomainManager() *domainManager {
	return &domainManager{
		locale:    DefaultLocale,
		domainMap: make(map[string]*fileSystem),
		trTextMap: make(map[string]*translator),
	}
}

func (p *domainManager) makeTrMapKey(domain, locale string) string {
	return domain + "_$$$_" + locale
}

func (p *domainManager) Bind(domain, path string, data []byte) (domains, paths []string) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	switch {
	case domain != "" && path != "": // bind new domain
		p.bindDomainTranslators(domain, path, data)
	case domain != "" && path == "": // delete domain
		p.deleteDomain(domain)
	}

	// return all bind domain
	for k, fs := range p.domainMap {
		domains = append(domains, k)
		paths = append(paths, fs.FsName)
	}
	return
}

func (p *domainManager) SetLocale(locale string) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if locale != "" {
		p.locale = locale
	}
	return p.locale
}

func (p *domainManager) SetDomain(domain string) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if domain != "" {
		p.domain = domain
	}
	return p.domain
}

func (p *domainManager) Getdata(name string) []byte {
	return p.getdata(p.domain, name)
}

func (p *domainManager) DGetdata(domain, name string) []byte {
	return p.getdata(domain, name)
}

func (p *domainManager) PNGettext(msgctxt, msgid, msgidPlural string, n int) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.gettext(p.domain, msgctxt, msgid, msgidPlural, n)
}

func (p *domainManager) DPNGettext(domain, msgctxt, msgid, msgidPlural string, n int) string {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	return p.gettext(domain, msgctxt, msgid, msgidPlural, n)
}

func (p *domainManager) gettext(domain, msgctxt, msgid, msgidPlural string, n int) string {
	if p.locale == "" || p.domain == "" {
		return msgid
	}
	if _, ok := p.domainMap[domain]; !ok {
		return msgid
	}
	if f, ok := p.trTextMap[p.makeTrMapKey(domain, p.locale)]; ok {
		return f.PNGettext(msgctxt, msgid, msgidPlural, n)
	}
	return msgid
}

func (p *domainManager) getdata(domain, name string) []byte {
	if p.locale == "" || p.domain == "" {
		return nil
	}
	if _, ok := p.domainMap[domain]; !ok {
		return nil
	}
	if fs, ok := p.domainMap[domain]; ok {
		if data, err := fs.LoadResourceFile(domain, p.locale, name); err == nil {
			return data
		}
		if p.locale != "default" {
			if data, err := fs.LoadResourceFile(domain, "default", name); err == nil {
				return data
			}
		}
	}
	return nil
}
