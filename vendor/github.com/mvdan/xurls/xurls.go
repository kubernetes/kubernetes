// Copyright (c) 2015, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

// Package xurls extracts urls from plain text using regular expressions.
package xurls

import "regexp"

//go:generate go run generate/tldsgen/main.go
//go:generate go run generate/regexgen/main.go

const (
	letter    = `\p{L}`
	number    = `\p{N}`
	iriChar   = letter + number
	currency  = `\p{Sc}`
	otherSymb = `\p{So}`
	endChar   = iriChar + `/\-+_&~*%=#` + currency
	midChar   = endChar + `@.,:;'?!|` + otherSymb
	wellParen = `\([` + midChar + `]*(\([` + midChar + `]*\)[` + midChar + `]*)*\)`
	wellBrack = `\[[` + midChar + `]*(\[[` + midChar + `]*\][` + midChar + `]*)*\]`
	wellBrace = `\{[` + midChar + `]*(\{[` + midChar + `]*\}[` + midChar + `]*)*\}`
	wellAll   = wellParen + `|` + wellBrack + `|` + wellBrace
	pathCont  = `([` + midChar + `]*(` + wellAll + `|[` + endChar + `])+)+`
	comScheme = `[a-zA-Z][a-zA-Z.\-+]*://`
	scheme    = `(` + comScheme + `|` + otherScheme + `)`

	iri      = `[` + iriChar + `]([` + iriChar + `\-]*[` + iriChar + `])?`
	domain   = `(` + iri + `\.)+`
	octet    = `(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])`
	ipv4Addr = `\b` + octet + `\.` + octet + `\.` + octet + `\.` + octet + `\b`
	ipv6Addr = `([0-9a-fA-F]{1,4}:([0-9a-fA-F]{1,4}:([0-9a-fA-F]{1,4}:([0-9a-fA-F]{1,4}:([0-9a-fA-F]{1,4}:[0-9a-fA-F]{0,4}|:[0-9a-fA-F]{1,4})?|(:[0-9a-fA-F]{1,4}){0,2})|(:[0-9a-fA-F]{1,4}){0,3})|(:[0-9a-fA-F]{1,4}){0,4})|:(:[0-9a-fA-F]{1,4}){0,5})((:[0-9a-fA-F]{1,4}){2}|:(25[0-5]|(2[0-4]|1[0-9]|[1-9])?[0-9])(\.(25[0-5]|(2[0-4]|1[0-9]|[1-9])?[0-9])){3})|(([0-9a-fA-F]{1,4}:){1,6}|:):[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){7}:`
	ipAddr   = `(` + ipv4Addr + `|` + ipv6Addr + `)`
	site     = domain + gtld
	hostName = `(` + site + `|` + ipAddr + `)`
	port     = `(:[0-9]*)?`
	path     = `(/|/` + pathCont + `?|\b|$)`
	webURL   = hostName + port + path

	strict  = `(\b` + scheme + pathCont + `)`
	relaxed = `(` + strict + `|` + webURL + `)`
)

var (
	// Relaxed matches all the urls it can find.
	Relaxed = regexp.MustCompile(relaxed)
	// Strict only matches urls with a scheme to avoid false positives.
	Strict = regexp.MustCompile(strict)
)

func init() {
	Relaxed.Longest()
	Strict.Longest()
}

// StrictMatchingScheme produces a regexp that matches urls like Strict but
// whose scheme matches the given regular expression.
func StrictMatchingScheme(exp string) (*regexp.Regexp, error) {
	strictMatching := `(\b(?i)(` + exp + `)(?-i)` + pathCont + `)`
	re, err := regexp.Compile(strictMatching)
	if err != nil {
		return nil, err
	}
	re.Longest()
	return re, nil
}
