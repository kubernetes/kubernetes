// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"fmt"
	"strings"
)

func (p *domainManager) bindDomainTranslators(domain, path string, data []byte) {
	if _, ok := p.domainMap[domain]; ok {
		p.deleteDomain(domain) // delete old domain
	}
	fs := newFileSystem(path, data)
	for locale, _ := range fs.LocaleMap {
		trMapKey := p.makeTrMapKey(domain, locale)
		if data, err := fs.LoadMessagesFile(domain, locale, ".mo"); err == nil {
			p.trTextMap[trMapKey], _ = newMoTranslator(
				fmt.Sprintf("%s_%s.mo", domain, locale),
				data,
			)
			continue
		}
		if data, err := fs.LoadMessagesFile(domain, locale, ".po"); err == nil {
			p.trTextMap[trMapKey], _ = newPoTranslator(
				fmt.Sprintf("%s_%s.po", domain, locale),
				data,
			)
			continue
		}
		p.trTextMap[p.makeTrMapKey(domain, locale)] = nilTranslator
	}
	p.domainMap[domain] = fs
}

func (p *domainManager) deleteDomain(domain string) {
	if _, ok := p.domainMap[domain]; !ok {
		return
	}
	// delete all mo files
	trMapKeyPrefix := p.makeTrMapKey(domain, "")
	for k, _ := range p.trTextMap {
		if strings.HasPrefix(k, trMapKeyPrefix) {
			delete(p.trTextMap, k)
		}
	}
	delete(p.domainMap, domain)
}
