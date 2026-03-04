// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"os"
	"strings"
)

func getDefaultLanguage() string {
	if v := os.Getenv("LC_MESSAGES"); v != "" {
		return simplifiedLanguage(v)
	}
	if v := os.Getenv("LANG"); v != "" {
		return simplifiedLanguage(v)
	}
	return "default"
}

func simplifiedLanguage(lang string) string {
	// en_US/en_US.UTF-8/zh_CN/zh_TW/el_GR@euro/...
	if idx := strings.Index(lang, ":"); idx != -1 {
		lang = lang[:idx]
	}
	if idx := strings.Index(lang, "@"); idx != -1 {
		lang = lang[:idx]
	}
	if idx := strings.Index(lang, "."); idx != -1 {
		lang = lang[:idx]
	}
	return strings.TrimSpace(lang)
}
