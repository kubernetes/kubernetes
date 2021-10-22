// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

// This file generates data for the CLDR plural rules, as defined in
//    https://unicode.org/reports/tr35/tr35-numbers.html#Language_Plural_Rules
//
// We assume a slightly simplified grammar:
//
// 		condition     = and_condition ('or' and_condition)* samples
// 		and_condition = relation ('and' relation)*
// 		relation      = expr ('=' | '!=') range_list
// 		expr          = operand ('%' '10' '0'* )?
// 		operand       = 'n' | 'i' | 'f' | 't' | 'v' | 'w'
// 		range_list    = (range | value) (',' range_list)*
// 		range         = value'..'value
// 		value         = digit+
// 		digit         = 0|1|2|3|4|5|6|7|8|9
//
// 		samples       = ('@integer' sampleList)?
// 		                ('@decimal' sampleList)?
// 		sampleList    = sampleRange (',' sampleRange)* (',' ('…'|'...'))?
// 		sampleRange   = decimalValue ('~' decimalValue)?
// 		decimalValue  = value ('.' value)?
//
//		Symbol	Value
//		n	absolute value of the source number (integer and decimals).
//		i	integer digits of n.
//		v	number of visible fraction digits in n, with trailing zeros.
//		w	number of visible fraction digits in n, without trailing zeros.
//		f	visible fractional digits in n, with trailing zeros.
//		t	visible fractional digits in n, without trailing zeros.
//
// The algorithm for which the data is generated is based on the following
// observations
//
//    - the number of different sets of numbers which the plural rules use to
//      test inclusion is limited,
//    - most numbers that are tested on are < 100
//
// This allows us to define a bitmap for each number < 100 where a bit i
// indicates whether this number is included in some defined set i.
// The function matchPlural in plural.go defines how we can subsequently use
// this data to determine inclusion.
//
// There are a few languages for which this doesn't work. For one Italian and
// Azerbaijan, which both test against numbers > 100 for ordinals and Breton,
// which considers whether numbers are multiples of hundreds. The model here
// could be extended to handle Italian and Azerbaijan fairly easily (by
// considering the numbers 100, 200, 300, ..., 800, 900 in addition to the first
// 100), but for now it seems easier to just hard-code these cases.

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"log"
	"strconv"
	"strings"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/language"
	"golang.org/x/text/internal/language/compact"
	"golang.org/x/text/unicode/cldr"
)

var (
	test = flag.Bool("test", false,
		"test existing tables; can be used to compare web data with package data.")
	outputFile     = flag.String("output", "tables.go", "output file")
	outputTestFile = flag.String("testoutput", "data_test.go", "output file")

	draft = flag.String("draft",
		"contributed",
		`Minimal draft requirements (approved, contributed, provisional, unconfirmed).`)
)

func main() {
	gen.Init()

	const pkg = "plural"

	gen.Repackage("gen_common.go", "common.go", pkg)
	// Read the CLDR zip file.
	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	d.SetDirFilter("supplemental", "main")
	d.SetSectionFilter("numbers", "plurals")
	data, err := d.DecodeZip(r)
	if err != nil {
		log.Fatalf("DecodeZip: %v", err)
	}

	w := gen.NewCodeWriter()
	defer w.WriteGoFile(*outputFile, pkg)

	gen.WriteCLDRVersion(w)

	genPlurals(w, data)

	w = gen.NewCodeWriter()
	defer w.WriteGoFile(*outputTestFile, pkg)

	genPluralsTests(w, data)
}

type pluralTest struct {
	locales string   // space-separated list of locales for this test
	form    int      // Use int instead of Form to simplify generation.
	integer []string // Entries of the form \d+ or \d+~\d+
	decimal []string // Entries of the form \f+ or \f+ +~\f+, where f is \d+\.\d+
}

func genPluralsTests(w *gen.CodeWriter, data *cldr.CLDR) {
	w.WriteType(pluralTest{})

	for _, plurals := range data.Supplemental().Plurals {
		if plurals.Type == "" {
			// The empty type is reserved for plural ranges.
			continue
		}
		tests := []pluralTest{}

		for _, pRules := range plurals.PluralRules {
			for _, rule := range pRules.PluralRule {
				test := pluralTest{
					locales: pRules.Locales,
					form:    int(countMap[rule.Count]),
				}
				scan := bufio.NewScanner(strings.NewReader(rule.Data()))
				scan.Split(splitTokens)
				var p *[]string
				for scan.Scan() {
					switch t := scan.Text(); t {
					case "@integer":
						p = &test.integer
					case "@decimal":
						p = &test.decimal
					case ",", "…":
					default:
						if p != nil {
							*p = append(*p, t)
						}
					}
				}
				tests = append(tests, test)
			}
		}
		w.WriteVar(plurals.Type+"Tests", tests)
	}
}

func genPlurals(w *gen.CodeWriter, data *cldr.CLDR) {
	for _, plurals := range data.Supplemental().Plurals {
		if plurals.Type == "" {
			continue
		}
		// Initialize setMap and inclusionMasks. They are already populated with
		// a few entries to serve as an example and to assign nice numbers to
		// common cases.

		// setMap contains sets of numbers represented by boolean arrays where
		// a true value for element i means that the number i is included.
		setMap := map[[numN]bool]int{
			// The above init func adds an entry for including all numbers.
			[numN]bool{1: true}: 1, // fix {1} to a nice value
			[numN]bool{2: true}: 2, // fix {2} to a nice value
			[numN]bool{0: true}: 3, // fix {0} to a nice value
		}

		// inclusionMasks contains bit masks for every number under numN to
		// indicate in which set the number is included. Bit 1 << x will be set
		// if it is included in set x.
		inclusionMasks := [numN]uint64{
			// Note: these entries are not complete: more bits will be set along the way.
			0: 1 << 3,
			1: 1 << 1,
			2: 1 << 2,
		}

		// Create set {0..99}. We will assign this set the identifier 0.
		var all [numN]bool
		for i := range all {
			// Mark number i as being included in the set (which has identifier 0).
			inclusionMasks[i] |= 1 << 0
			// Mark number i as included in the set.
			all[i] = true
		}
		// Register the identifier for the set.
		setMap[all] = 0

		rules := []pluralCheck{}
		index := []byte{0}
		langMap := map[compact.ID]byte{0: 0}

		for _, pRules := range plurals.PluralRules {
			// Parse the rules.
			var conds []orCondition
			for _, rule := range pRules.PluralRule {
				form := countMap[rule.Count]
				conds = parsePluralCondition(conds, rule.Data(), form)
			}
			// Encode the rules.
			for _, c := range conds {
				// If an or condition only has filters, we create an entry for
				// this filter and the set that contains all values.
				empty := true
				for _, b := range c.used {
					empty = empty && !b
				}
				if empty {
					rules = append(rules, pluralCheck{
						cat:   byte(opMod<<opShift) | byte(c.form),
						setID: 0, // all values
					})
					continue
				}
				// We have some entries with values.
				for i, set := range c.set {
					if !c.used[i] {
						continue
					}
					index, ok := setMap[set]
					if !ok {
						index = len(setMap)
						setMap[set] = index
						for i := range inclusionMasks {
							if set[i] {
								inclusionMasks[i] |= 1 << uint64(index)
							}
						}
					}
					rules = append(rules, pluralCheck{
						cat:   byte(i<<opShift | andNext),
						setID: byte(index),
					})
				}
				// Now set the last entry to the plural form the rule matches.
				rules[len(rules)-1].cat &^= formMask
				rules[len(rules)-1].cat |= byte(c.form)
			}
			// Point the relevant locales to the created entries.
			for _, loc := range strings.Split(pRules.Locales, " ") {
				if strings.TrimSpace(loc) == "" {
					continue
				}
				lang, ok := compact.FromTag(language.MustParse(loc))
				if !ok {
					log.Printf("No compact index for locale %q", loc)
				}
				langMap[lang] = byte(len(index) - 1)
			}
			index = append(index, byte(len(rules)))
		}
		w.WriteVar(plurals.Type+"Rules", rules)
		w.WriteVar(plurals.Type+"Index", index)
		// Expand the values: first by using the parent relationship.
		langToIndex := make([]byte, compact.NumCompactTags)
		for i := range langToIndex {
			for p := compact.ID(i); ; p = p.Parent() {
				if x, ok := langMap[p]; ok {
					langToIndex[i] = x
					break
				}
			}
		}
		// Now expand by including entries with identical languages for which
		// one isn't set.
		for i, v := range langToIndex {
			if v == 0 {
				id, _ := compact.FromTag(language.Tag{
					LangID: compact.ID(i).Tag().LangID,
				})
				if p := langToIndex[id]; p != 0 {
					langToIndex[i] = p
				}
			}
		}
		w.WriteVar(plurals.Type+"LangToIndex", langToIndex)
		// Need to convert array to slice because of golang.org/issue/7651.
		// This will allow tables to be dropped when unused. This is especially
		// relevant for the ordinal data, which I suspect won't be used as much.
		w.WriteVar(plurals.Type+"InclusionMasks", inclusionMasks[:])

		if len(rules) > 0xFF {
			log.Fatalf("Too many entries for rules: %#x", len(rules))
		}
		if len(index) > 0xFF {
			log.Fatalf("Too many entries for index: %#x", len(index))
		}
		if len(setMap) > 64 { // maximum number of bits.
			log.Fatalf("Too many entries for setMap: %d", len(setMap))
		}
		w.WriteComment(
			"Slots used for %s: %X of 0xFF rules; %X of 0xFF indexes; %d of 64 sets",
			plurals.Type, len(rules), len(index), len(setMap))
		// Prevent comment from attaching to the next entry.
		fmt.Fprint(w, "\n\n")
	}
}

type orCondition struct {
	original string // for debugging

	form Form
	used [32]bool
	set  [32][numN]bool
}

func (o *orCondition) add(op opID, mod int, v []int) (ok bool) {
	ok = true
	for _, x := range v {
		if x >= maxMod {
			ok = false
			break
		}
	}
	for i := 0; i < numN; i++ {
		m := i
		if mod != 0 {
			m = i % mod
		}
		if !intIn(m, v) {
			o.set[op][i] = false
		}
	}
	if ok {
		o.used[op] = true
	}
	return ok
}

func intIn(x int, a []int) bool {
	for _, y := range a {
		if x == y {
			return true
		}
	}
	return false
}

var operandIndex = map[string]opID{
	"i": opI,
	"n": opN,
	"f": opF,
	"v": opV,
	"w": opW,
}

// parsePluralCondition parses the condition of a single pluralRule and appends
// the resulting or conditions to conds.
//
// Example rules:
//   // Category "one" in English: only allow 1 with no visible fraction
//   i = 1 and v = 0 @integer 1
//
//   // Category "few" in Czech: all numbers with visible fractions
//   v != 0   @decimal ...
//
//   // Category "zero" in Latvian: all multiples of 10 or the numbers 11-19 or
//   // numbers with a fraction 11..19 and no trailing zeros.
//   n % 10 = 0 or n % 100 = 11..19 or v = 2 and f % 100 = 11..19 @integer ...
//
// @integer and @decimal are followed by examples and are not relevant for the
// rule itself. The are used here to signal the termination of the rule.
func parsePluralCondition(conds []orCondition, s string, f Form) []orCondition {
	scan := bufio.NewScanner(strings.NewReader(s))
	scan.Split(splitTokens)
	for {
		cond := orCondition{original: s, form: f}
		// Set all numbers to be allowed for all number classes and restrict
		// from here on.
		for i := range cond.set {
			for j := range cond.set[i] {
				cond.set[i][j] = true
			}
		}
	andLoop:
		for {
			var token string
			scan.Scan() // Must exist.
			switch class := scan.Text(); class {
			case "t":
				class = "w" // equal to w for t == 0
				fallthrough
			case "n", "i", "f", "v", "w":
				op := scanToken(scan)
				opCode := operandIndex[class]
				mod := 0
				if op == "%" {
					opCode |= opMod

					switch v := scanUint(scan); v {
					case 10, 100:
						mod = v
					case 1000:
						// A more general solution would be to allow checking
						// against multiples of 100 and include entries for the
						// numbers 100..900 in the inclusion masks. At the
						// moment this would only help Azerbaijan and Italian.

						// Italian doesn't use '%', so this must be Azerbaijan.
						cond.used[opAzerbaijan00s] = true
						return append(conds, cond)

					case 1000000:
						cond.used[opBretonM] = true
						return append(conds, cond)

					default:
						log.Fatalf("Modulo value not supported %d", v)
					}
					op = scanToken(scan)
				}
				if op != "=" && op != "!=" {
					log.Fatalf("Unexpected op %q", op)
				}
				if op == "!=" {
					opCode |= opNotEqual
				}
				a := []int{}
				v := scanUint(scan)
				if class == "w" && v != 0 {
					log.Fatalf("Must compare against zero for operand type %q", class)
				}
				token = scanToken(scan)
				for {
					switch token {
					case "..":
						end := scanUint(scan)
						for ; v <= end; v++ {
							a = append(a, v)
						}
						token = scanToken(scan)
					default: // ",", "or", "and", "@..."
						a = append(a, v)
					}
					if token != "," {
						break
					}
					v = scanUint(scan)
					token = scanToken(scan)
				}
				if !cond.add(opCode, mod, a) {
					// Detected large numbers. As we ruled out Azerbaijan, this
					// must be the many rule for Italian ordinals.
					cond.set[opItalian800] = cond.set[opN]
					cond.used[opItalian800] = true
				}

			case "@integer", "@decimal": // "other" entry: tests only.
				return conds
			default:
				log.Fatalf("Unexpected operand class %q (%s)", class, s)
			}
			switch token {
			case "or":
				conds = append(conds, cond)
				break andLoop
			case "@integer", "@decimal": // examples
				// There is always an example in practice, so we always terminate here.
				if err := scan.Err(); err != nil {
					log.Fatal(err)
				}
				return append(conds, cond)
			case "and":
				// keep accumulating
			default:
				log.Fatalf("Unexpected token %q", token)
			}
		}
	}
}

func scanToken(scan *bufio.Scanner) string {
	scan.Scan()
	return scan.Text()
}

func scanUint(scan *bufio.Scanner) int {
	scan.Scan()
	val, err := strconv.ParseUint(scan.Text(), 10, 32)
	if err != nil {
		log.Fatal(err)
	}
	return int(val)
}

// splitTokens can be used with bufio.Scanner to tokenize CLDR plural rules.
func splitTokens(data []byte, atEOF bool) (advance int, token []byte, err error) {
	condTokens := [][]byte{
		[]byte(".."),
		[]byte(","),
		[]byte("!="),
		[]byte("="),
	}
	advance, token, err = bufio.ScanWords(data, atEOF)
	for _, t := range condTokens {
		if len(t) >= len(token) {
			continue
		}
		switch p := bytes.Index(token, t); {
		case p == -1:
		case p == 0:
			advance = len(t)
			token = token[:len(t)]
			return advance - len(token) + len(t), token[:len(t)], err
		case p < advance:
			// Don't split when "=" overlaps "!=".
			if t[0] == '=' && token[p-1] == '!' {
				continue
			}
			advance = p
			token = token[:p]
		}
	}
	return advance, token, err
}
