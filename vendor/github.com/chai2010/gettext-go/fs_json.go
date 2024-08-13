// Copyright 2020 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"encoding/json"
	"fmt"
	"sort"
)

type jsonFS struct {
	name string
	x    map[string]struct {
		LC_MESSAGES map[string][]struct {
			MsgContext  string   `json:"msgctxt"`      // msgctxt context
			MsgId       string   `json:"msgid"`        // msgid untranslated-string
			MsgIdPlural string   `json:"msgid_plural"` // msgid_plural untranslated-string-plural
			MsgStr      []string `json:"msgstr"`       // msgstr translated-string
		}
		LC_RESOURCE map[string]map[string]string
	}
}

func isJsonData() bool {
	return false
}

func newJson(jsonData []byte, name string) (*jsonFS, error) {
	p := &jsonFS{name: name}
	if err := json.Unmarshal(jsonData, &p.x); err != nil {
		return nil, err
	}

	return p, nil
}

func (p *jsonFS) LocaleList() []string {
	var ss []string
	for lang := range p.x {
		ss = append(ss, lang)
	}
	sort.Strings(ss)
	return ss
}

func (p *jsonFS) LoadMessagesFile(domain, lang, ext string) ([]byte, error) {
	if v, ok := p.x[lang]; ok {
		if v, ok := v.LC_MESSAGES[domain+ext]; ok {
			return json.Marshal(v)
		}
	}
	return nil, fmt.Errorf("not found")
}
func (p *jsonFS) LoadResourceFile(domain, lang, name string) ([]byte, error) {
	if v, ok := p.x[lang]; ok {
		if v, ok := v.LC_RESOURCE[domain]; ok {
			return []byte(v[name]), nil
		}
	}
	return nil, fmt.Errorf("not found")
}
func (p *jsonFS) String() string {
	return "gettext.nilfs(" + p.name + ")"
}
