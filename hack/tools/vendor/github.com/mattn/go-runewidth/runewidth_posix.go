// +build !windows
// +build !js
// +build !appengine

package runewidth

import (
	"os"
	"regexp"
	"strings"
)

var reLoc = regexp.MustCompile(`^[a-z][a-z][a-z]?(?:_[A-Z][A-Z])?\.(.+)`)

var mblenTable = map[string]int{
	"utf-8":   6,
	"utf8":    6,
	"jis":     8,
	"eucjp":   3,
	"euckr":   2,
	"euccn":   2,
	"sjis":    2,
	"cp932":   2,
	"cp51932": 2,
	"cp936":   2,
	"cp949":   2,
	"cp950":   2,
	"big5":    2,
	"gbk":     2,
	"gb2312":  2,
}

func isEastAsian(locale string) bool {
	charset := strings.ToLower(locale)
	r := reLoc.FindStringSubmatch(locale)
	if len(r) == 2 {
		charset = strings.ToLower(r[1])
	}

	if strings.HasSuffix(charset, "@cjk_narrow") {
		return false
	}

	for pos, b := range []byte(charset) {
		if b == '@' {
			charset = charset[:pos]
			break
		}
	}
	max := 1
	if m, ok := mblenTable[charset]; ok {
		max = m
	}
	if max > 1 && (charset[0] != 'u' ||
		strings.HasPrefix(locale, "ja") ||
		strings.HasPrefix(locale, "ko") ||
		strings.HasPrefix(locale, "zh")) {
		return true
	}
	return false
}

// IsEastAsian return true if the current locale is CJK
func IsEastAsian() bool {
	locale := os.Getenv("LC_ALL")
	if locale == "" {
		locale = os.Getenv("LC_CTYPE")
	}
	if locale == "" {
		locale = os.Getenv("LANG")
	}

	// ignore C locale
	if locale == "POSIX" || locale == "C" {
		return false
	}
	if len(locale) > 1 && locale[0] == 'C' && (locale[1] == '.' || locale[1] == '-') {
		return false
	}

	return isEastAsian(locale)
}
