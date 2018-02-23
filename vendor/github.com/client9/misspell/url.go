package misspell

import (
	"regexp"
)

// Regexp for URL https://mathiasbynens.be/demo/url-regex
//
// original @imme_emosol (54 chars) has trouble with dashes in hostname
// @(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?$@iS
var reURL = regexp.MustCompile(`(?i)(https?|ftp)://(-\.)?([^\s/?\.#]+\.?)+(/[^\s]*)?`)

// StripURL attemps to replace URLs with blank spaces, e.g.
//  "xxx http://foo.com/ yyy -> "xxx          yyyy"
func StripURL(s string) string {
	return reURL.ReplaceAllStringFunc(s, replaceWithBlanks)
}
