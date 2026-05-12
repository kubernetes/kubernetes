// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message

// TODO: some types in this file will need to be made public at some time.
// Documentation and method names will reflect this by using the exported name.

import (
	"golang.org/x/text/language"
	"golang.org/x/text/message/catalog"
)

// MatchLanguage reports the matched tag obtained from language.MatchStrings for
// the Matcher of the DefaultCatalog.
func MatchLanguage(preferred ...string) language.Tag {
	c := DefaultCatalog
	tag, _ := language.MatchStrings(c.Matcher(), preferred...)
	return tag
}

// DefaultCatalog is used by SetString.
var DefaultCatalog catalog.Catalog = defaultCatalog

var defaultCatalog = catalog.NewBuilder()

// SetString calls SetString on the initial default Catalog.
func SetString(tag language.Tag, key string, msg string) error {
	return defaultCatalog.SetString(tag, key, msg)
}

// Set calls Set on the initial default Catalog.
func Set(tag language.Tag, key string, msg ...catalog.Message) error {
	return defaultCatalog.Set(tag, key, msg...)
}
