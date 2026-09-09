// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

var (
	DefaultLanguage string = getDefaultLanguage() // use $(LC_MESSAGES) or $(LANG) or "default"
)

type Gettexter interface {
	FileSystem() FileSystem

	GetDomain() string
	SetDomain(domain string) Gettexter

	GetLanguage() string
	SetLanguage(lang string) Gettexter

	Gettext(msgid string) string
	PGettext(msgctxt, msgid string) string

	NGettext(msgid, msgidPlural string, n int) string
	PNGettext(msgctxt, msgid, msgidPlural string, n int) string

	DGettext(domain, msgid string) string
	DPGettext(domain, msgctxt, msgid string) string
	DNGettext(domain, msgid, msgidPlural string, n int) string
	DPNGettext(domain, msgctxt, msgid, msgidPlural string, n int) string

	Getdata(name string) []byte
	DGetdata(domain, name string) []byte
}

// New create Interface use default language.
func New(domain, path string, data ...interface{}) Gettexter {
	return newLocale(domain, path, data...)
}

var defaultGettexter struct {
	lang   string
	domain string
	Gettexter
}

func init() {
	defaultGettexter.lang = getDefaultLanguage()
	defaultGettexter.domain = "default"
	defaultGettexter.Gettexter = newLocale("", "")
}

// BindLocale sets and queries program's domains.
//
// Examples:
//	BindLocale(New("poedit", "locale")) // bind "poedit" domain
//
// Use zip file:
//	BindLocale(New("poedit", "locale.zip"))          // bind "poedit" domain
//	BindLocale(New("poedit", "locale.zip", zipData)) // bind "poedit" domain
//
// Use FileSystem:
//	BindLocale(New("poedit", "name", OS("path/to/dir"))) // bind "poedit" domain
//	BindLocale(New("poedit", "name", OS("path/to.zip"))) // bind "poedit" domain
//
func BindLocale(g Gettexter) {
	if g != nil {
		defaultGettexter.Gettexter = g
		defaultGettexter.SetLanguage(defaultGettexter.lang)
	} else {
		defaultGettexter.Gettexter = newLocale("", "")
		defaultGettexter.SetLanguage(defaultGettexter.lang)
	}
}

// SetLanguage sets and queries the program's current lang.
//
// If the lang is not empty string, set the new locale.
//
// If the lang is empty string, don't change anything.
//
// Returns is the current locale.
//
// Examples:
//	SetLanguage("")      // get locale: return DefaultLocale
//	SetLanguage("zh_CN") // set locale: return zh_CN
//	SetLanguage("")      // get locale: return zh_CN
func SetLanguage(lang string) string {
	defaultGettexter.SetLanguage(lang)
	return defaultGettexter.GetLanguage()
}

// SetDomain sets and retrieves the current message domain.
//
// If the domain is not empty string, set the new domains.
//
// If the domain is empty string, don't change anything.
//
// Returns is the all used domains.
//
// Examples:
//	SetDomain("poedit") // set domain: poedit
//	SetDomain("")       // get domain: return poedit
func SetDomain(domain string) string {
	defaultGettexter.SetDomain(domain)
	return defaultGettexter.GetDomain()
}

// Gettext attempt to translate a text string into the user's native language,
// by looking up the translation in a message catalog.
//
// It use the caller's function name as the msgctxt.
//
// Examples:
//	func Foo() {
//		msg := gettext.Gettext("Hello") // msgctxt is ""
//	}
func Gettext(msgid string) string {
	return defaultGettexter.Gettext(msgid)
}

// Getdata attempt to translate a resource file into the user's native language,
// by looking up the translation in a message catalog.
//
// Examples:
//	func Foo() {
//		Textdomain("hello")
//		BindLocale("hello", "locale.zip", nilOrZipData)
//		poems := gettext.Getdata("poems.txt")
//	}
func Getdata(name string) []byte {
	return defaultGettexter.Getdata(name)
}

// NGettext attempt to translate a text string into the user's native language,
// by looking up the appropriate plural form of the translation in a message
// catalog.
//
// It use the caller's function name as the msgctxt.
//
// Examples:
//	func Foo() {
//		msg := gettext.NGettext("%d people", "%d peoples", 2)
//	}
func NGettext(msgid, msgidPlural string, n int) string {
	return defaultGettexter.NGettext(msgid, msgidPlural, n)
}

// PGettext attempt to translate a text string into the user's native language,
// by looking up the translation in a message catalog.
//
// Examples:
//	func Foo() {
//		msg := gettext.PGettext("gettext-go.example", "Hello") // msgctxt is "gettext-go.example"
//	}
func PGettext(msgctxt, msgid string) string {
	return defaultGettexter.PGettext(msgctxt, msgid)
}

// PNGettext attempt to translate a text string into the user's native language,
// by looking up the appropriate plural form of the translation in a message
// catalog.
//
// Examples:
//	func Foo() {
//		msg := gettext.PNGettext("gettext-go.example", "%d people", "%d peoples", 2)
//	}
func PNGettext(msgctxt, msgid, msgidPlural string, n int) string {
	return defaultGettexter.PNGettext(msgctxt, msgid, msgidPlural, n)
}

// DGettext like Gettext(), but looking up the message in the specified domain.
//
// Examples:
//	func Foo() {
//		msg := gettext.DGettext("poedit", "Hello")
//	}
func DGettext(domain, msgid string) string {
	return defaultGettexter.DGettext(domain, msgid)
}

// DNGettext like NGettext(), but looking up the message in the specified domain.
//
// Examples:
//	func Foo() {
//		msg := gettext.PNGettext("poedit", "gettext-go.example", "%d people", "%d peoples", 2)
//	}
func DNGettext(domain, msgid, msgidPlural string, n int) string {
	return defaultGettexter.DNGettext(domain, msgid, msgidPlural, n)
}

// DPGettext like PGettext(), but looking up the message in the specified domain.
//
// Examples:
//	func Foo() {
//		msg := gettext.DPGettext("poedit", "gettext-go.example", "Hello")
//	}
func DPGettext(domain, msgctxt, msgid string) string {
	return defaultGettexter.DPGettext(domain, msgctxt, msgid)
}

// DPNGettext like PNGettext(), but looking up the message in the specified domain.
//
// Examples:
//	func Foo() {
//		msg := gettext.DPNGettext("poedit", "gettext-go.example", "%d people", "%d peoples", 2)
//	}
func DPNGettext(domain, msgctxt, msgid, msgidPlural string, n int) string {
	return defaultGettexter.DPNGettext(domain, msgctxt, msgid, msgidPlural, n)
}

// DGetdata like Getdata(), but looking up the resource in the specified domain.
//
// Examples:
//	func Foo() {
//		msg := gettext.DGetdata("hello", "poems.txt")
//	}
func DGetdata(domain, name string) []byte {
	return defaultGettexter.DGetdata(domain, name)
}
