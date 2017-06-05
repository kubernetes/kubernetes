// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Collation table generator.
// Data read from the web.

package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/collate"
	"golang.org/x/text/collate/build"
	"golang.org/x/text/collate/colltab"
	"golang.org/x/text/internal/gen"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/cldr"
)

var (
	test = flag.Bool("test", false,
		"test existing tables; can be used to compare web data with package data.")
	short = flag.Bool("short", false, `Use "short" alternatives, when available.`)
	draft = flag.Bool("draft", false, `Use draft versions, when available.`)
	tags  = flag.String("tags", "", "build tags to be included after +build directive")
	pkg   = flag.String("package", "collate",
		"the name of the package in which the generated file is to be included")

	tables = flagStringSetAllowAll("tables", "collate", "collate,chars",
		"comma-spearated list of tables to generate.")
	exclude = flagStringSet("exclude", "zh2", "",
		"comma-separated list of languages to exclude.")
	include = flagStringSet("include", "", "",
		"comma-separated list of languages to include. Include trumps exclude.")
	// TODO: Not included: unihan gb2312han zhuyin big5han (for size reasons)
	// TODO: Not included: traditional (buggy for Bengali)
	types = flagStringSetAllowAll("types", "standard,phonebook,phonetic,reformed,pinyin,stroke", "",
		"comma-separated list of types that should be included.")
)

// stringSet implements an ordered set based on a list.  It implements flag.Value
// to allow a set to be specified as a comma-separated list.
type stringSet struct {
	s        []string
	allowed  *stringSet
	dirty    bool // needs compaction if true
	all      bool
	allowAll bool
}

func flagStringSet(name, def, allowed, usage string) *stringSet {
	ss := &stringSet{}
	if allowed != "" {
		usage += fmt.Sprintf(" (allowed values: any of %s)", allowed)
		ss.allowed = &stringSet{}
		failOnError(ss.allowed.Set(allowed))
	}
	ss.Set(def)
	flag.Var(ss, name, usage)
	return ss
}

func flagStringSetAllowAll(name, def, allowed, usage string) *stringSet {
	ss := &stringSet{allowAll: true}
	if allowed == "" {
		flag.Var(ss, name, usage+fmt.Sprintf(` Use "all" to select all.`))
	} else {
		ss.allowed = &stringSet{}
		failOnError(ss.allowed.Set(allowed))
		flag.Var(ss, name, usage+fmt.Sprintf(` (allowed values: "all" or any of %s)`, allowed))
	}
	ss.Set(def)
	return ss
}

func (ss stringSet) Len() int {
	return len(ss.s)
}

func (ss stringSet) String() string {
	return strings.Join(ss.s, ",")
}

func (ss *stringSet) Set(s string) error {
	if ss.allowAll && s == "all" {
		ss.s = nil
		ss.all = true
		return nil
	}
	ss.s = ss.s[:0]
	for _, s := range strings.Split(s, ",") {
		if s := strings.TrimSpace(s); s != "" {
			if ss.allowed != nil && !ss.allowed.contains(s) {
				return fmt.Errorf("unsupported value %q; must be one of %s", s, ss.allowed)
			}
			ss.add(s)
		}
	}
	ss.compact()
	return nil
}

func (ss *stringSet) add(s string) {
	ss.s = append(ss.s, s)
	ss.dirty = true
}

func (ss *stringSet) values() []string {
	ss.compact()
	return ss.s
}

func (ss *stringSet) contains(s string) bool {
	if ss.all {
		return true
	}
	for _, v := range ss.s {
		if v == s {
			return true
		}
	}
	return false
}

func (ss *stringSet) compact() {
	if !ss.dirty {
		return
	}
	a := ss.s
	sort.Strings(a)
	k := 0
	for i := 1; i < len(a); i++ {
		if a[k] != a[i] {
			a[k+1] = a[i]
			k++
		}
	}
	ss.s = a[:k+1]
	ss.dirty = false
}

func skipLang(l string) bool {
	if include.Len() > 0 {
		return !include.contains(l)
	}
	return exclude.contains(l)
}

// altInclude returns a list of alternatives (for the LDML alt attribute)
// in order of preference.  An empty string in this list indicates the
// default entry.
func altInclude() []string {
	l := []string{}
	if *short {
		l = append(l, "short")
	}
	l = append(l, "")
	// TODO: handle draft using cldr.SetDraftLevel
	if *draft {
		l = append(l, "proposed")
	}
	return l
}

func failOnError(e error) {
	if e != nil {
		log.Panic(e)
	}
}

func openArchive() *zip.Reader {
	f := gen.OpenCLDRCoreZip()
	buffer, err := ioutil.ReadAll(f)
	f.Close()
	failOnError(err)
	archive, err := zip.NewReader(bytes.NewReader(buffer), int64(len(buffer)))
	failOnError(err)
	return archive
}

// parseUCA parses a Default Unicode Collation Element Table of the format
// specified in http://www.unicode.org/reports/tr10/#File_Format.
// It returns the variable top.
func parseUCA(builder *build.Builder) {
	var r io.ReadCloser
	var err error
	for _, f := range openArchive().File {
		if strings.HasSuffix(f.Name, "allkeys_CLDR.txt") {
			r, err = f.Open()
		}
	}
	if r == nil {
		log.Fatal("File allkeys_CLDR.txt not found in archive.")
	}
	failOnError(err)
	defer r.Close()
	scanner := bufio.NewScanner(r)
	colelem := regexp.MustCompile(`\[([.*])([0-9A-F.]+)\]`)
	for i := 1; scanner.Scan(); i++ {
		line := scanner.Text()
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		if line[0] == '@' {
			// parse properties
			switch {
			case strings.HasPrefix(line[1:], "version "):
				a := strings.Split(line[1:], " ")
				if a[1] != gen.UnicodeVersion() {
					log.Fatalf("incompatible version %s; want %s", a[1], gen.UnicodeVersion())
				}
			case strings.HasPrefix(line[1:], "backwards "):
				log.Fatalf("%d: unsupported option backwards", i)
			default:
				log.Printf("%d: unknown option %s", i, line[1:])
			}
		} else {
			// parse entries
			part := strings.Split(line, " ; ")
			if len(part) != 2 {
				log.Fatalf("%d: production rule without ';': %v", i, line)
			}
			lhs := []rune{}
			for _, v := range strings.Split(part[0], " ") {
				if v == "" {
					continue
				}
				lhs = append(lhs, rune(convHex(i, v)))
			}
			var n int
			var vars []int
			rhs := [][]int{}
			for i, m := range colelem.FindAllStringSubmatch(part[1], -1) {
				n += len(m[0])
				elem := []int{}
				for _, h := range strings.Split(m[2], ".") {
					elem = append(elem, convHex(i, h))
				}
				if m[1] == "*" {
					vars = append(vars, i)
				}
				rhs = append(rhs, elem)
			}
			if len(part[1]) < n+3 || part[1][n+1] != '#' {
				log.Fatalf("%d: expected comment; found %s", i, part[1][n:])
			}
			if *test {
				testInput.add(string(lhs))
			}
			failOnError(builder.Add(lhs, rhs, vars))
		}
	}
	if scanner.Err() != nil {
		log.Fatal(scanner.Err())
	}
}

func convHex(line int, s string) int {
	r, e := strconv.ParseInt(s, 16, 32)
	if e != nil {
		log.Fatalf("%d: %v", line, e)
	}
	return int(r)
}

var testInput = stringSet{}

var charRe = regexp.MustCompile(`&#x([0-9A-F]*);`)
var tagRe = regexp.MustCompile(`<([a-z_]*)  */>`)

var mainLocales = []string{}

// charsets holds a list of exemplar characters per category.
type charSets map[string][]string

func (p charSets) fprint(w io.Writer) {
	fmt.Fprintln(w, "[exN]string{")
	for i, k := range []string{"", "contractions", "punctuation", "auxiliary", "currencySymbol", "index"} {
		if set := p[k]; len(set) != 0 {
			fmt.Fprintf(w, "\t\t%d: %q,\n", i, strings.Join(set, " "))
		}
	}
	fmt.Fprintln(w, "\t},")
}

var localeChars = make(map[string]charSets)

const exemplarHeader = `
type exemplarType int
const (
	exCharacters exemplarType = iota
	exContractions
	exPunctuation
	exAuxiliary
	exCurrency
	exIndex
	exN
)
`

func printExemplarCharacters(w io.Writer) {
	fmt.Fprintln(w, exemplarHeader)
	fmt.Fprintln(w, "var exemplarCharacters = map[string][exN]string{")
	for _, loc := range mainLocales {
		fmt.Fprintf(w, "\t%q: ", loc)
		localeChars[loc].fprint(w)
	}
	fmt.Fprintln(w, "}")
}

func decodeCLDR(d *cldr.Decoder) *cldr.CLDR {
	r := gen.OpenCLDRCoreZip()
	data, err := d.DecodeZip(r)
	failOnError(err)
	return data
}

// parseMain parses XML files in the main directory of the CLDR core.zip file.
func parseMain() {
	d := &cldr.Decoder{}
	d.SetDirFilter("main")
	d.SetSectionFilter("characters")
	data := decodeCLDR(d)
	for _, loc := range data.Locales() {
		x := data.RawLDML(loc)
		if skipLang(x.Identity.Language.Type) {
			continue
		}
		if x.Characters != nil {
			x, _ = data.LDML(loc)
			loc = language.Make(loc).String()
			for _, ec := range x.Characters.ExemplarCharacters {
				if ec.Draft != "" {
					continue
				}
				if _, ok := localeChars[loc]; !ok {
					mainLocales = append(mainLocales, loc)
					localeChars[loc] = make(charSets)
				}
				localeChars[loc][ec.Type] = parseCharacters(ec.Data())
			}
		}
	}
}

func parseCharacters(chars string) []string {
	parseSingle := func(s string) (r rune, tail string, escaped bool) {
		if s[0] == '\\' {
			return rune(s[1]), s[2:], true
		}
		r, sz := utf8.DecodeRuneInString(s)
		return r, s[sz:], false
	}
	chars = strings.TrimSpace(chars)
	if n := len(chars) - 1; chars[n] == ']' && chars[0] == '[' {
		chars = chars[1:n]
	}
	list := []string{}
	var r, last, end rune
	for len(chars) > 0 {
		if chars[0] == '{' { // character sequence
			buf := []rune{}
			for chars = chars[1:]; len(chars) > 0; {
				r, chars, _ = parseSingle(chars)
				if r == '}' {
					break
				}
				if r == ' ' {
					log.Fatalf("space not supported in sequence %q", chars)
				}
				buf = append(buf, r)
			}
			list = append(list, string(buf))
			last = 0
		} else { // single character
			escaped := false
			r, chars, escaped = parseSingle(chars)
			if r != ' ' {
				if r == '-' && !escaped {
					if last == 0 {
						log.Fatal("'-' should be preceded by a character")
					}
					end, chars, _ = parseSingle(chars)
					for ; last <= end; last++ {
						list = append(list, string(last))
					}
					last = 0
				} else {
					list = append(list, string(r))
					last = r
				}
			}
		}
	}
	return list
}

var fileRe = regexp.MustCompile(`.*/collation/(.*)\.xml`)

// typeMap translates legacy type keys to their BCP47 equivalent.
var typeMap = map[string]string{
	"phonebook":   "phonebk",
	"traditional": "trad",
}

// parseCollation parses XML files in the collation directory of the CLDR core.zip file.
func parseCollation(b *build.Builder) {
	d := &cldr.Decoder{}
	d.SetDirFilter("collation")
	data := decodeCLDR(d)
	for _, loc := range data.Locales() {
		x, err := data.LDML(loc)
		failOnError(err)
		if skipLang(x.Identity.Language.Type) {
			continue
		}
		cs := x.Collations.Collation
		sl := cldr.MakeSlice(&cs)
		if len(types.s) == 0 {
			sl.SelectAnyOf("type", x.Collations.Default())
		} else if !types.all {
			sl.SelectAnyOf("type", types.s...)
		}
		sl.SelectOnePerGroup("alt", altInclude())

		for _, c := range cs {
			id, err := language.Parse(loc)
			if err != nil {
				fmt.Fprintf(os.Stderr, "invalid locale: %q", err)
				continue
			}
			// Support both old- and new-style defaults.
			d := c.Type
			if x.Collations.DefaultCollation == nil {
				d = x.Collations.Default()
			} else {
				d = x.Collations.DefaultCollation.Data()
			}
			// We assume tables are being built either for search or collation,
			// but not both. For search the default is always "search".
			if d != c.Type && c.Type != "search" {
				typ := c.Type
				if len(c.Type) > 8 {
					typ = typeMap[c.Type]
				}
				id, err = id.SetTypeForKey("co", typ)
				failOnError(err)
			}
			t := b.Tailoring(id)
			c.Process(processor{t})
		}
	}
}

type processor struct {
	t *build.Tailoring
}

func (p processor) Reset(anchor string, before int) (err error) {
	if before != 0 {
		err = p.t.SetAnchorBefore(anchor)
	} else {
		err = p.t.SetAnchor(anchor)
	}
	failOnError(err)
	return nil
}

func (p processor) Insert(level int, str, context, extend string) error {
	str = context + str
	if *test {
		testInput.add(str)
	}
	// TODO: mimic bug in old maketables: remove.
	err := p.t.Insert(colltab.Level(level-1), str, context+extend)
	failOnError(err)
	return nil
}

func (p processor) Index(id string) {
}

func testCollator(c *collate.Collator) {
	c0 := collate.New(language.Und)

	// iterator over all characters for all locales and check
	// whether Key is equal.
	buf := collate.Buffer{}

	// Add all common and not too uncommon runes to the test set.
	for i := rune(0); i < 0x30000; i++ {
		testInput.add(string(i))
	}
	for i := rune(0xE0000); i < 0xF0000; i++ {
		testInput.add(string(i))
	}
	for _, str := range testInput.values() {
		k0 := c0.KeyFromString(&buf, str)
		k := c.KeyFromString(&buf, str)
		if !bytes.Equal(k0, k) {
			failOnError(fmt.Errorf("test:%U: keys differ (%x vs %x)", []rune(str), k0, k))
		}
		buf.Reset()
	}
	fmt.Println("PASS")
}

func main() {
	gen.Init()
	b := build.NewBuilder()
	parseUCA(b)
	if tables.contains("chars") {
		parseMain()
	}
	parseCollation(b)

	c, err := b.Build()
	failOnError(err)

	if *test {
		testCollator(collate.NewFromTable(c))
	} else {
		w := &bytes.Buffer{}

		gen.WriteUnicodeVersion(w)
		gen.WriteCLDRVersion(w)

		if tables.contains("collate") {
			_, err = b.Print(w)
			failOnError(err)
		}
		if tables.contains("chars") {
			printExemplarCharacters(w)
		}
		gen.WriteGoFile("tables.go", *pkg, w.Bytes())
	}
}
