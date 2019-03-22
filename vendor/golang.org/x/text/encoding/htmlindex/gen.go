// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"golang.org/x/text/internal/gen"
)

type group struct {
	Encodings []struct {
		Labels []string
		Name   string
	}
}

func main() {
	gen.Init()

	r := gen.Open("https://encoding.spec.whatwg.org", "whatwg", "encodings.json")
	var groups []group
	if err := json.NewDecoder(r).Decode(&groups); err != nil {
		log.Fatalf("Error reading encodings.json: %v", err)
	}

	w := &bytes.Buffer{}
	fmt.Fprintln(w, "type htmlEncoding byte")
	fmt.Fprintln(w, "const (")
	for i, g := range groups {
		for _, e := range g.Encodings {
			key := strings.ToLower(e.Name)
			name := consts[key]
			if name == "" {
				log.Fatalf("No const defined for %s.", key)
			}
			if i == 0 {
				fmt.Fprintf(w, "%s htmlEncoding = iota\n", name)
			} else {
				fmt.Fprintf(w, "%s\n", name)
			}
		}
	}
	fmt.Fprintln(w, "numEncodings")
	fmt.Fprint(w, ")\n\n")

	fmt.Fprintln(w, "var canonical = [numEncodings]string{")
	for _, g := range groups {
		for _, e := range g.Encodings {
			fmt.Fprintf(w, "%q,\n", strings.ToLower(e.Name))
		}
	}
	fmt.Fprint(w, "}\n\n")

	fmt.Fprintln(w, "var nameMap = map[string]htmlEncoding{")
	for _, g := range groups {
		for _, e := range g.Encodings {
			for _, l := range e.Labels {
				key := strings.ToLower(e.Name)
				name := consts[key]
				fmt.Fprintf(w, "%q: %s,\n", l, name)
			}
		}
	}
	fmt.Fprint(w, "}\n\n")

	var tags []string
	fmt.Fprintln(w, "var localeMap = []htmlEncoding{")
	for _, loc := range locales {
		tags = append(tags, loc.tag)
		fmt.Fprintf(w, "%s, // %s \n", consts[loc.name], loc.tag)
	}
	fmt.Fprint(w, "}\n\n")

	fmt.Fprintf(w, "const locales = %q\n", strings.Join(tags, " "))

	gen.WriteGoFile("tables.go", "htmlindex", w.Bytes())
}

// consts maps canonical encoding name to internal constant.
var consts = map[string]string{
	"utf-8":          "utf8",
	"ibm866":         "ibm866",
	"iso-8859-2":     "iso8859_2",
	"iso-8859-3":     "iso8859_3",
	"iso-8859-4":     "iso8859_4",
	"iso-8859-5":     "iso8859_5",
	"iso-8859-6":     "iso8859_6",
	"iso-8859-7":     "iso8859_7",
	"iso-8859-8":     "iso8859_8",
	"iso-8859-8-i":   "iso8859_8I",
	"iso-8859-10":    "iso8859_10",
	"iso-8859-13":    "iso8859_13",
	"iso-8859-14":    "iso8859_14",
	"iso-8859-15":    "iso8859_15",
	"iso-8859-16":    "iso8859_16",
	"koi8-r":         "koi8r",
	"koi8-u":         "koi8u",
	"macintosh":      "macintosh",
	"windows-874":    "windows874",
	"windows-1250":   "windows1250",
	"windows-1251":   "windows1251",
	"windows-1252":   "windows1252",
	"windows-1253":   "windows1253",
	"windows-1254":   "windows1254",
	"windows-1255":   "windows1255",
	"windows-1256":   "windows1256",
	"windows-1257":   "windows1257",
	"windows-1258":   "windows1258",
	"x-mac-cyrillic": "macintoshCyrillic",
	"gbk":            "gbk",
	"gb18030":        "gb18030",
	// "hz-gb-2312":     "hzgb2312", // Was removed from WhatWG
	"big5":           "big5",
	"euc-jp":         "eucjp",
	"iso-2022-jp":    "iso2022jp",
	"shift_jis":      "shiftJIS",
	"euc-kr":         "euckr",
	"replacement":    "replacement",
	"utf-16be":       "utf16be",
	"utf-16le":       "utf16le",
	"x-user-defined": "xUserDefined",
}

// locales is taken from
// https://html.spec.whatwg.org/multipage/syntax.html#encoding-sniffing-algorithm.
var locales = []struct{ tag, name string }{
	// The default value. Explicitly state latin to benefit from the exact
	// script option, while still making 1252 the default encoding for languages
	// written in Latin script.
	{"und_Latn", "windows-1252"},
	{"ar", "windows-1256"},
	{"ba", "windows-1251"},
	{"be", "windows-1251"},
	{"bg", "windows-1251"},
	{"cs", "windows-1250"},
	{"el", "iso-8859-7"},
	{"et", "windows-1257"},
	{"fa", "windows-1256"},
	{"he", "windows-1255"},
	{"hr", "windows-1250"},
	{"hu", "iso-8859-2"},
	{"ja", "shift_jis"},
	{"kk", "windows-1251"},
	{"ko", "euc-kr"},
	{"ku", "windows-1254"},
	{"ky", "windows-1251"},
	{"lt", "windows-1257"},
	{"lv", "windows-1257"},
	{"mk", "windows-1251"},
	{"pl", "iso-8859-2"},
	{"ru", "windows-1251"},
	{"sah", "windows-1251"},
	{"sk", "windows-1250"},
	{"sl", "iso-8859-2"},
	{"sr", "windows-1251"},
	{"tg", "windows-1251"},
	{"th", "windows-874"},
	{"tr", "windows-1254"},
	{"tt", "windows-1251"},
	{"uk", "windows-1251"},
	{"vi", "windows-1258"},
	{"zh-hans", "gb18030"},
	{"zh-hant", "big5"},
}
