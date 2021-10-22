// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// Unicode table generator.
// Data read from the web.

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/ucd"
	"golang.org/x/text/unicode/rangetable"
)

func main() {
	flag.Parse()
	setupOutput()
	loadChars() // always needed
	loadCasefold()
	printCategories()
	printScriptOrProperty(false)
	printScriptOrProperty(true)
	printCases()
	printLatinProperties()
	printCasefold()
	printSizes()
	flushOutput()
}

func defaultVersion() string {
	if v := os.Getenv("UNICODE_VERSION"); v != "" {
		return v
	}
	return unicode.Version
}

var tablelist = flag.String("tables",
	"all",
	"comma-separated list of which tables to generate; can be letter")
var scriptlist = flag.String("scripts",
	"all",
	"comma-separated list of which script tables to generate")
var proplist = flag.String("props",
	"all",
	"comma-separated list of which property tables to generate")
var cases = flag.Bool("cases",
	true,
	"generate case tables")
var test = flag.Bool("test",
	false,
	"test existing tables; can be used to compare web data with package data")

var scriptRe = regexp.MustCompile(`^([0-9A-F]+)(\.\.[0-9A-F]+)? *; ([A-Za-z_]+)$`)
var logger = log.New(os.Stderr, "", log.Lshortfile)

var output *gen.CodeWriter

func setupOutput() {
	output = gen.NewCodeWriter()
}

func flushOutput() {
	output.WriteGoFile("tables.go", "unicode")
}

func printf(format string, args ...interface{}) {
	fmt.Fprintf(output, format, args...)
}

func print(args ...interface{}) {
	fmt.Fprint(output, args...)
}

func println(args ...interface{}) {
	fmt.Fprintln(output, args...)
}

var category = map[string]bool{
	// Nd Lu etc.
	// We use one-character names to identify merged categories
	"L": true, // Lu Ll Lt Lm Lo
	"P": true, // Pc Pd Ps Pe Pu Pf Po
	"M": true, // Mn Mc Me
	"N": true, // Nd Nl No
	"S": true, // Sm Sc Sk So
	"Z": true, // Zs Zl Zp
	"C": true, // Cc Cf Cs Co Cn
}

// This contains only the properties we're interested in.
type Char struct {
	codePoint rune // if zero, this index is not a valid code point.
	category  string
	upperCase rune
	lowerCase rune
	titleCase rune
	foldCase  rune // simple case folding
	caseOrbit rune // next in simple case folding orbit
}

const MaxChar = 0x10FFFF

var chars = make([]Char, MaxChar+1)
var scripts = make(map[string][]rune)
var props = make(map[string][]rune) // a property looks like a script; can share the format

func allCategories() []string {
	a := make([]string, 0, len(category))
	for k := range category {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}

func all(scripts map[string][]rune) []string {
	a := make([]string, 0, len(scripts))
	for k := range scripts {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}

func allCatFold(m map[string]map[rune]bool) []string {
	a := make([]string, 0, len(m))
	for k := range m {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}

func categoryOp(code rune, class uint8) bool {
	category := chars[code].category
	return len(category) > 0 && category[0] == class
}

func loadChars() {
	ucd.Parse(gen.OpenUCDFile("UnicodeData.txt"), func(p *ucd.Parser) {
		c := Char{codePoint: p.Rune(0)}

		getRune := func(field int) rune {
			if p.String(field) == "" {
				return 0
			}
			return p.Rune(field)
		}

		c.category = p.String(ucd.GeneralCategory)
		category[c.category] = true
		switch c.category {
		case "Nd":
			// Decimal digit
			p.Int(ucd.NumericValue)
		case "Lu":
			c.upperCase = getRune(ucd.CodePoint)
			c.lowerCase = getRune(ucd.SimpleLowercaseMapping)
			c.titleCase = getRune(ucd.SimpleTitlecaseMapping)
		case "Ll":
			c.upperCase = getRune(ucd.SimpleUppercaseMapping)
			c.lowerCase = getRune(ucd.CodePoint)
			c.titleCase = getRune(ucd.SimpleTitlecaseMapping)
		case "Lt":
			c.upperCase = getRune(ucd.SimpleUppercaseMapping)
			c.lowerCase = getRune(ucd.SimpleLowercaseMapping)
			c.titleCase = getRune(ucd.CodePoint)
		default:
			c.upperCase = getRune(ucd.SimpleUppercaseMapping)
			c.lowerCase = getRune(ucd.SimpleLowercaseMapping)
			c.titleCase = getRune(ucd.SimpleTitlecaseMapping)
		}

		chars[c.codePoint] = c
	})
}

func loadCasefold() {
	ucd.Parse(gen.OpenUCDFile("CaseFolding.txt"), func(p *ucd.Parser) {
		kind := p.String(1)
		if kind != "C" && kind != "S" {
			// Only care about 'common' and 'simple' foldings.
			return
		}
		p1 := p.Rune(0)
		p2 := p.Rune(2)
		chars[p1].foldCase = rune(p2)
	})
}

var categoryMapping = map[string]string{
	"Lu": "Letter, uppercase",
	"Ll": "Letter, lowercase",
	"Lt": "Letter, titlecase",
	"Lm": "Letter, modifier",
	"Lo": "Letter, other",
	"Mn": "Mark, nonspacing",
	"Mc": "Mark, spacing combining",
	"Me": "Mark, enclosing",
	"Nd": "Number, decimal digit",
	"Nl": "Number, letter",
	"No": "Number, other",
	"Pc": "Punctuation, connector",
	"Pd": "Punctuation, dash",
	"Ps": "Punctuation, open",
	"Pe": "Punctuation, close",
	"Pi": "Punctuation, initial quote",
	"Pf": "Punctuation, final quote",
	"Po": "Punctuation, other",
	"Sm": "Symbol, math",
	"Sc": "Symbol, currency",
	"Sk": "Symbol, modifier",
	"So": "Symbol, other",
	"Zs": "Separator, space",
	"Zl": "Separator, line",
	"Zp": "Separator, paragraph",
	"Cc": "Other, control",
	"Cf": "Other, format",
	"Cs": "Other, surrogate",
	"Co": "Other, private use",
	"Cn": "Other, not assigned",
}

func printCategories() {
	if *tablelist == "" {
		return
	}
	// Find out which categories to dump
	list := strings.Split(*tablelist, ",")
	if *tablelist == "all" {
		list = allCategories()
	}
	if *test {
		fullCategoryTest(list)
		return
	}

	println("// Version is the Unicode edition from which the tables are derived.")
	printf("const Version = %q\n\n", gen.UnicodeVersion())

	if *tablelist == "all" {
		println("// Categories is the set of Unicode category tables.")
		println("var Categories = map[string] *RangeTable {")
		for _, k := range allCategories() {
			printf("\t%q: %s,\n", k, k)
		}
		print("}\n\n")
	}

	decl := make(sort.StringSlice, len(list))
	ndecl := 0
	for _, name := range list {
		if _, ok := category[name]; !ok {
			logger.Fatal("unknown category", name)
		}
		// We generate an UpperCase name to serve as concise documentation and an _UnderScored
		// name to store the data. This stops godoc dumping all the tables but keeps them
		// available to clients.
		// Cases deserving special comments
		varDecl := ""
		switch name {
		case "C":
			varDecl = "\tOther = _C;	// Other/C is the set of Unicode control and special characters, category C.\n"
			varDecl += "\tC = _C\n"
		case "L":
			varDecl = "\tLetter = _L;	// Letter/L is the set of Unicode letters, category L.\n"
			varDecl += "\tL = _L\n"
		case "M":
			varDecl = "\tMark = _M;	// Mark/M is the set of Unicode mark characters, category M.\n"
			varDecl += "\tM = _M\n"
		case "N":
			varDecl = "\tNumber = _N;	// Number/N is the set of Unicode number characters, category N.\n"
			varDecl += "\tN = _N\n"
		case "P":
			varDecl = "\tPunct = _P;	// Punct/P is the set of Unicode punctuation characters, category P.\n"
			varDecl += "\tP = _P\n"
		case "S":
			varDecl = "\tSymbol = _S;	// Symbol/S is the set of Unicode symbol characters, category S.\n"
			varDecl += "\tS = _S\n"
		case "Z":
			varDecl = "\tSpace = _Z;	// Space/Z is the set of Unicode space characters, category Z.\n"
			varDecl += "\tZ = _Z\n"
		case "Nd":
			varDecl = "\tDigit = _Nd;	// Digit is the set of Unicode characters with the \"decimal digit\" property.\n"
		case "Lu":
			varDecl = "\tUpper = _Lu;	// Upper is the set of Unicode upper case letters.\n"
		case "Ll":
			varDecl = "\tLower = _Ll;	// Lower is the set of Unicode lower case letters.\n"
		case "Lt":
			varDecl = "\tTitle = _Lt;	// Title is the set of Unicode title case letters.\n"
		}
		if len(name) > 1 {
			desc, ok := categoryMapping[name]
			if ok {
				varDecl += fmt.Sprintf(
					"\t%s = _%s;	// %s is the set of Unicode characters in category %s (%s).\n",
					name, name, name, name, desc)
			} else {
				varDecl += fmt.Sprintf(
					"\t%s = _%s;	// %s is the set of Unicode characters in category %s.\n",
					name, name, name, name)
			}
		}
		decl[ndecl] = varDecl
		ndecl++
		if len(name) == 1 { // unified categories
			dumpRange(
				"_"+name,
				func(code rune) bool { return categoryOp(code, name[0]) })
			continue
		}
		dumpRange("_"+name,
			func(code rune) bool { return chars[code].category == name })
	}
	decl.Sort()
	println("// These variables have type *RangeTable.")
	println("var (")
	for _, d := range decl {
		print(d)
	}
	print(")\n\n")
}

type Op func(code rune) bool

func dumpRange(name string, inCategory Op) {
	runes := []rune{}
	for i := range chars {
		r := rune(i)
		if inCategory(r) {
			runes = append(runes, r)
		}
	}
	printRangeTable(name, runes)
}

func printRangeTable(name string, runes []rune) {
	rt := rangetable.New(runes...)
	printf("var %s = &RangeTable{\n", name)
	println("\tR16: []Range16{")
	for _, r := range rt.R16 {
		printf("\t\t{%#04x, %#04x, %d},\n", r.Lo, r.Hi, r.Stride)
		range16Count++
	}
	println("\t},")
	if len(rt.R32) > 0 {
		println("\tR32: []Range32{")
		for _, r := range rt.R32 {
			printf("\t\t{%#x, %#x, %d},\n", r.Lo, r.Hi, r.Stride)
			range32Count++
		}
		println("\t},")
	}
	if rt.LatinOffset > 0 {
		printf("\tLatinOffset: %d,\n", rt.LatinOffset)
	}
	printf("}\n\n")
}

func fullCategoryTest(list []string) {
	for _, name := range list {
		if _, ok := category[name]; !ok {
			logger.Fatal("unknown category", name)
		}
		r, ok := unicode.Categories[name]
		if !ok && len(name) > 1 {
			logger.Fatalf("unknown table %q", name)
		}
		if len(name) == 1 {
			verifyRange(name, func(code rune) bool { return categoryOp(code, name[0]) }, r)
		} else {
			verifyRange(
				name,
				func(code rune) bool { return chars[code].category == name },
				r)
		}
	}
}

func verifyRange(name string, inCategory Op, table *unicode.RangeTable) {
	count := 0
	for j := range chars {
		i := rune(j)
		web := inCategory(i)
		pkg := unicode.Is(table, i)
		if web != pkg {
			fmt.Fprintf(os.Stderr, "%s: %U: web=%t pkg=%t\n", name, i, web, pkg)
			count++
			if count > 10 {
				break
			}
		}
	}
}

func fullScriptTest(list []string, installed map[string]*unicode.RangeTable, scripts map[string][]rune) {
	for _, name := range list {
		if _, ok := scripts[name]; !ok {
			logger.Fatal("unknown script", name)
		}
		_, ok := installed[name]
		if !ok {
			logger.Fatal("unknown table", name)
		}
		for _, r := range scripts[name] {
			if !unicode.Is(installed[name], rune(r)) {
				fmt.Fprintf(os.Stderr, "%U: not in script %s\n", r, name)
			}
		}
	}
}

var deprecatedAliases = map[string]string{
	"Sentence_Terminal": "STerm",
}

// PropList.txt has the same format as Scripts.txt so we can share its parser.
func printScriptOrProperty(doProps bool) {
	flaglist := *scriptlist
	file := "Scripts.txt"
	table := scripts
	installed := unicode.Scripts
	if doProps {
		flaglist = *proplist
		file = "PropList.txt"
		table = props
		installed = unicode.Properties
	}
	if flaglist == "" {
		return
	}
	ucd.Parse(gen.OpenUCDFile(file), func(p *ucd.Parser) {
		name := p.String(1)
		table[name] = append(table[name], p.Rune(0))
	})
	// Find out which scripts to dump
	list := strings.Split(flaglist, ",")
	if flaglist == "all" {
		list = all(table)
	}
	if *test {
		fullScriptTest(list, installed, table)
		return
	}

	if flaglist == "all" {
		if doProps {
			println("// Properties is the set of Unicode property tables.")
			println("var Properties = map[string] *RangeTable{")
		} else {
			println("// Scripts is the set of Unicode script tables.")
			println("var Scripts = map[string] *RangeTable{")
		}
		for _, k := range all(table) {
			printf("\t%q: %s,\n", k, k)
			if alias, ok := deprecatedAliases[k]; ok {
				printf("\t%q: %s,\n", alias, k)
			}
		}
		print("}\n\n")
	}

	decl := make(sort.StringSlice, len(list)+len(deprecatedAliases))
	ndecl := 0
	for _, name := range list {
		if doProps {
			decl[ndecl] = fmt.Sprintf(
				"\t%s = _%s;\t// %s is the set of Unicode characters with property %s.\n",
				name, name, name, name)
		} else {
			decl[ndecl] = fmt.Sprintf(
				"\t%s = _%s;\t// %s is the set of Unicode characters in script %s.\n",
				name, name, name, name)
		}
		ndecl++
		if alias, ok := deprecatedAliases[name]; ok {
			decl[ndecl] = fmt.Sprintf(
				"\t%[1]s = _%[2]s;\t// %[1]s is an alias for %[2]s.\n",
				alias, name)
			ndecl++
		}
		printRangeTable("_"+name, table[name])
	}
	decl.Sort()
	println("// These variables have type *RangeTable.")
	println("var (")
	for _, d := range decl {
		print(d)
	}
	print(")\n\n")
}

const (
	CaseUpper = 1 << iota
	CaseLower
	CaseTitle
	CaseNone    = 0  // must be zero
	CaseMissing = -1 // character not present; not a valid case state
)

type caseState struct {
	point        rune
	_case        int
	deltaToUpper rune
	deltaToLower rune
	deltaToTitle rune
}

// Is d a continuation of the state of c?
func (c *caseState) adjacent(d *caseState) bool {
	if d.point < c.point {
		c, d = d, c
	}
	switch {
	case d.point != c.point+1: // code points not adjacent (shouldn't happen)
		return false
	case d._case != c._case: // different cases
		return c.upperLowerAdjacent(d)
	case c._case == CaseNone:
		return false
	case c._case == CaseMissing:
		return false
	case d.deltaToUpper != c.deltaToUpper:
		return false
	case d.deltaToLower != c.deltaToLower:
		return false
	case d.deltaToTitle != c.deltaToTitle:
		return false
	}
	return true
}

// Is d the same as c, but opposite in upper/lower case? this would make it
// an element of an UpperLower sequence.
func (c *caseState) upperLowerAdjacent(d *caseState) bool {
	// check they're a matched case pair.  we know they have adjacent values
	switch {
	case c._case == CaseUpper && d._case != CaseLower:
		return false
	case c._case == CaseLower && d._case != CaseUpper:
		return false
	}
	// matched pair (at least in upper/lower).  make the order Upper Lower
	if c._case == CaseLower {
		c, d = d, c
	}
	// for an Upper Lower sequence the deltas have to be in order
	//	c: 0 1 0
	//	d: -1 0 -1
	switch {
	case c.deltaToUpper != 0:
		return false
	case c.deltaToLower != 1:
		return false
	case c.deltaToTitle != 0:
		return false
	case d.deltaToUpper != -1:
		return false
	case d.deltaToLower != 0:
		return false
	case d.deltaToTitle != -1:
		return false
	}
	return true
}

// Does this character start an UpperLower sequence?
func (c *caseState) isUpperLower() bool {
	// for an Upper Lower sequence the deltas have to be in order
	//	c: 0 1 0
	switch {
	case c.deltaToUpper != 0:
		return false
	case c.deltaToLower != 1:
		return false
	case c.deltaToTitle != 0:
		return false
	}
	return true
}

// Does this character start a LowerUpper sequence?
func (c *caseState) isLowerUpper() bool {
	// for an Upper Lower sequence the deltas have to be in order
	//	c: -1 0 -1
	switch {
	case c.deltaToUpper != -1:
		return false
	case c.deltaToLower != 0:
		return false
	case c.deltaToTitle != -1:
		return false
	}
	return true
}

func getCaseState(i rune) (c *caseState) {
	c = &caseState{point: i, _case: CaseNone}
	ch := &chars[i]
	switch ch.codePoint {
	case 0:
		c._case = CaseMissing // Will get NUL wrong but that doesn't matter
		return
	case ch.upperCase:
		c._case = CaseUpper
	case ch.lowerCase:
		c._case = CaseLower
	case ch.titleCase:
		c._case = CaseTitle
	}
	// Some things such as roman numeral U+2161 don't describe themselves
	// as upper case, but have a lower case. Second-guess them.
	if c._case == CaseNone && ch.lowerCase != 0 {
		c._case = CaseUpper
	}
	// Same in the other direction.
	if c._case == CaseNone && ch.upperCase != 0 {
		c._case = CaseLower
	}

	if ch.upperCase != 0 {
		c.deltaToUpper = ch.upperCase - i
	}
	if ch.lowerCase != 0 {
		c.deltaToLower = ch.lowerCase - i
	}
	if ch.titleCase != 0 {
		c.deltaToTitle = ch.titleCase - i
	}
	return
}

func printCases() {
	if *test {
		fullCaseTest()
		return
	}
	printf(
		"// CaseRanges is the table describing case mappings for all letters with\n" +
			"// non-self mappings.\n" +
			"var CaseRanges = _CaseRanges\n" +
			"var _CaseRanges = []CaseRange {\n")

	var startState *caseState    // the start of a run; nil for not active
	var prevState = &caseState{} // the state of the previous character
	for i := range chars {
		state := getCaseState(rune(i))
		if state.adjacent(prevState) {
			prevState = state
			continue
		}
		// end of run (possibly)
		printCaseRange(startState, prevState)
		startState = nil
		if state._case != CaseMissing && state._case != CaseNone {
			startState = state
		}
		prevState = state
	}
	print("}\n")
}

func printCaseRange(lo, hi *caseState) {
	if lo == nil {
		return
	}
	if lo.deltaToUpper == 0 && lo.deltaToLower == 0 && lo.deltaToTitle == 0 {
		// character represents itself in all cases - no need to mention it
		return
	}
	switch {
	case hi.point > lo.point && lo.isUpperLower():
		printf("\t{0x%04X, 0x%04X, d{UpperLower, UpperLower, UpperLower}},\n",
			lo.point, hi.point)
	case hi.point > lo.point && lo.isLowerUpper():
		logger.Fatalf("LowerUpper sequence: should not happen: %U.  If it's real, need to fix To()", lo.point)
		printf("\t{0x%04X, 0x%04X, d{LowerUpper, LowerUpper, LowerUpper}},\n",
			lo.point, hi.point)
	default:
		printf("\t{0x%04X, 0x%04X, d{%d, %d, %d}},\n",
			lo.point, hi.point,
			lo.deltaToUpper, lo.deltaToLower, lo.deltaToTitle)
	}
}

// If the cased value in the Char is 0, it means use the rune itself.
func caseIt(r, cased rune) rune {
	if cased == 0 {
		return r
	}
	return cased
}

func fullCaseTest() {
	for j, c := range chars {
		i := rune(j)
		lower := unicode.ToLower(i)
		want := caseIt(i, c.lowerCase)
		if lower != want {
			fmt.Fprintf(os.Stderr, "lower %U should be %U is %U\n", i, want, lower)
		}
		upper := unicode.ToUpper(i)
		want = caseIt(i, c.upperCase)
		if upper != want {
			fmt.Fprintf(os.Stderr, "upper %U should be %U is %U\n", i, want, upper)
		}
		title := unicode.ToTitle(i)
		want = caseIt(i, c.titleCase)
		if title != want {
			fmt.Fprintf(os.Stderr, "title %U should be %U is %U\n", i, want, title)
		}
	}
}

func printLatinProperties() {
	if *test {
		return
	}
	println("var properties = [MaxLatin1+1]uint8{")
	for code := 0; code <= unicode.MaxLatin1; code++ {
		var property string
		switch chars[code].category {
		case "Cc", "": // NUL has no category.
			property = "pC"
		case "Cf": // soft hyphen, unique category, not printable.
			property = "0"
		case "Ll":
			property = "pLl | pp"
		case "Lo":
			property = "pLo | pp"
		case "Lu":
			property = "pLu | pp"
		case "Nd", "No":
			property = "pN | pp"
		case "Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps":
			property = "pP | pp"
		case "Sc", "Sk", "Sm", "So":
			property = "pS | pp"
		case "Zs":
			property = "pZ"
		default:
			logger.Fatalf("%U has unknown category %q", code, chars[code].category)
		}
		// Special case
		if code == ' ' {
			property = "pZ | pp"
		}
		printf("\t0x%02X: %s, // %q\n", code, property, code)
	}
	printf("}\n\n")
}

func printCasefold() {
	// Build list of case-folding groups attached to each canonical folded char (typically lower case).
	var caseOrbit = make([][]rune, MaxChar+1)
	for j := range chars {
		i := rune(j)
		c := &chars[i]
		if c.foldCase == 0 {
			continue
		}
		orb := caseOrbit[c.foldCase]
		if orb == nil {
			orb = append(orb, c.foldCase)
		}
		caseOrbit[c.foldCase] = append(orb, i)
	}

	// Insert explicit 1-element groups when assuming [lower, upper] would be wrong.
	for j := range chars {
		i := rune(j)
		c := &chars[i]
		f := c.foldCase
		if f == 0 {
			f = i
		}
		orb := caseOrbit[f]
		if orb == nil && (c.upperCase != 0 && c.upperCase != i || c.lowerCase != 0 && c.lowerCase != i) {
			// Default assumption of [upper, lower] is wrong.
			caseOrbit[i] = []rune{i}
		}
	}

	// Delete the groups for which assuming [lower, upper] or [upper, lower] is right.
	for i, orb := range caseOrbit {
		if len(orb) == 2 && chars[orb[0]].upperCase == orb[1] && chars[orb[1]].lowerCase == orb[0] {
			caseOrbit[i] = nil
		}
		if len(orb) == 2 && chars[orb[1]].upperCase == orb[0] && chars[orb[0]].lowerCase == orb[1] {
			caseOrbit[i] = nil
		}
	}

	// Record orbit information in chars.
	for _, orb := range caseOrbit {
		if orb == nil {
			continue
		}
		sort.Slice(orb, func(i, j int) bool {
			return orb[i] < orb[j]
		})
		c := orb[len(orb)-1]
		for _, d := range orb {
			chars[c].caseOrbit = d
			c = d
		}
	}

	printAsciiFold()
	printCaseOrbit()

	// Tables of category and script folding exceptions: code points
	// that must be added when interpreting a particular category/script
	// in a case-folding context.
	cat := make(map[string]map[rune]bool)
	for name := range category {
		if x := foldExceptions(inCategory(name)); len(x) > 0 {
			cat[name] = x
		}
	}

	scr := make(map[string]map[rune]bool)
	for name := range scripts {
		if x := foldExceptions(scripts[name]); len(x) > 0 {
			scr[name] = x
		}
	}

	printCatFold("FoldCategory", cat)
	printCatFold("FoldScript", scr)
}

// inCategory returns a list of all the runes in the category.
func inCategory(name string) []rune {
	var x []rune
	for j := range chars {
		i := rune(j)
		c := &chars[i]
		if c.category == name || len(name) == 1 && len(c.category) > 1 && c.category[0] == name[0] {
			x = append(x, i)
		}
	}
	return x
}

// foldExceptions returns a list of all the runes fold-equivalent
// to runes in class but not in class themselves.
func foldExceptions(class []rune) map[rune]bool {
	// Create map containing class and all fold-equivalent chars.
	m := make(map[rune]bool)
	for _, r := range class {
		c := &chars[r]
		if c.caseOrbit == 0 {
			// Just upper and lower.
			if u := c.upperCase; u != 0 {
				m[u] = true
			}
			if l := c.lowerCase; l != 0 {
				m[l] = true
			}
			m[r] = true
			continue
		}
		// Otherwise walk orbit.
		r0 := r
		for {
			m[r] = true
			r = chars[r].caseOrbit
			if r == r0 {
				break
			}
		}
	}

	// Remove class itself.
	for _, r := range class {
		delete(m, r)
	}

	// What's left is the exceptions.
	return m
}

var comment = map[string]string{
	"FoldCategory": "// FoldCategory maps a category name to a table of\n" +
		"// code points outside the category that are equivalent under\n" +
		"// simple case folding to code points inside the category.\n" +
		"// If there is no entry for a category name, there are no such points.\n",

	"FoldScript": "// FoldScript maps a script name to a table of\n" +
		"// code points outside the script that are equivalent under\n" +
		"// simple case folding to code points inside the script.\n" +
		"// If there is no entry for a script name, there are no such points.\n",
}

func printAsciiFold() {
	printf("var asciiFold = [MaxASCII + 1]uint16{\n")
	for i := rune(0); i <= unicode.MaxASCII; i++ {
		c := chars[i]
		f := c.caseOrbit
		if f == 0 {
			if c.lowerCase != i && c.lowerCase != 0 {
				f = c.lowerCase
			} else if c.upperCase != i && c.upperCase != 0 {
				f = c.upperCase
			} else {
				f = i
			}
		}
		printf("\t0x%04X,\n", f)
	}
	printf("}\n\n")
}

func printCaseOrbit() {
	if *test {
		for j := range chars {
			i := rune(j)
			c := &chars[i]
			f := c.caseOrbit
			if f == 0 {
				if c.lowerCase != i && c.lowerCase != 0 {
					f = c.lowerCase
				} else if c.upperCase != i && c.upperCase != 0 {
					f = c.upperCase
				} else {
					f = i
				}
			}
			if g := unicode.SimpleFold(i); g != f {
				fmt.Fprintf(os.Stderr, "unicode.SimpleFold(%#U) = %#U, want %#U\n", i, g, f)
			}
		}
		return
	}

	printf("var caseOrbit = []foldPair{\n")
	for i := range chars {
		c := &chars[i]
		if c.caseOrbit != 0 {
			printf("\t{0x%04X, 0x%04X},\n", i, c.caseOrbit)
			foldPairCount++
		}
	}
	printf("}\n\n")
}

func printCatFold(name string, m map[string]map[rune]bool) {
	if *test {
		var pkgMap map[string]*unicode.RangeTable
		if name == "FoldCategory" {
			pkgMap = unicode.FoldCategory
		} else {
			pkgMap = unicode.FoldScript
		}
		if len(pkgMap) != len(m) {
			fmt.Fprintf(os.Stderr, "unicode.%s has %d elements, want %d\n", name, len(pkgMap), len(m))
			return
		}
		for k, v := range m {
			t, ok := pkgMap[k]
			if !ok {
				fmt.Fprintf(os.Stderr, "unicode.%s[%q] missing\n", name, k)
				continue
			}
			n := 0
			for _, r := range t.R16 {
				for c := rune(r.Lo); c <= rune(r.Hi); c += rune(r.Stride) {
					if !v[c] {
						fmt.Fprintf(os.Stderr, "unicode.%s[%q] contains %#U, should not\n", name, k, c)
					}
					n++
				}
			}
			for _, r := range t.R32 {
				for c := rune(r.Lo); c <= rune(r.Hi); c += rune(r.Stride) {
					if !v[c] {
						fmt.Fprintf(os.Stderr, "unicode.%s[%q] contains %#U, should not\n", name, k, c)
					}
					n++
				}
			}
			if n != len(v) {
				fmt.Fprintf(os.Stderr, "unicode.%s[%q] has %d code points, want %d\n", name, k, n, len(v))
			}
		}
		return
	}

	print(comment[name])
	printf("var %s = map[string]*RangeTable{\n", name)
	for _, name := range allCatFold(m) {
		printf("\t%q: fold%s,\n", name, name)
	}
	printf("}\n\n")
	for _, name := range allCatFold(m) {
		class := m[name]
		dumpRange("fold"+name, func(code rune) bool { return class[code] })
	}
}

var range16Count = 0  // Number of entries in the 16-bit range tables.
var range32Count = 0  // Number of entries in the 32-bit range tables.
var foldPairCount = 0 // Number of fold pairs in the exception tables.

func printSizes() {
	if *test {
		return
	}
	println()
	printf("// Range entries: %d 16-bit, %d 32-bit, %d total.\n", range16Count, range32Count, range16Count+range32Count)
	range16Bytes := range16Count * 3 * 2
	range32Bytes := range32Count * 3 * 4
	printf("// Range bytes: %d 16-bit, %d 32-bit, %d total.\n", range16Bytes, range32Bytes, range16Bytes+range32Bytes)
	println()
	printf("// Fold orbit bytes: %d pairs, %d bytes\n", foldPairCount, foldPairCount*2*2)
}
