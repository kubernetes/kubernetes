// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Language tag table generator.
// Data read from the web.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/tag"
	"golang.org/x/text/unicode/cldr"
)

var (
	test = flag.Bool("test",
		false,
		"test existing tables; can be used to compare web data with package data.")
	outputFile = flag.String("output",
		"tables.go",
		"output file for generated tables")
)

var comment = []string{
	`
lang holds an alphabetically sorted list of ISO-639 language identifiers.
All entries are 4 bytes. The index of the identifier (divided by 4) is the language tag.
For 2-byte language identifiers, the two successive bytes have the following meaning:
    - if the first letter of the 2- and 3-letter ISO codes are the same:
      the second and third letter of the 3-letter ISO code.
    - otherwise: a 0 and a by 2 bits right-shifted index into altLangISO3.
For 3-byte language identifiers the 4th byte is 0.`,
	`
langNoIndex is a bit vector of all 3-letter language codes that are not used as an index
in lookup tables. The language ids for these language codes are derived directly
from the letters and are not consecutive.`,
	`
altLangISO3 holds an alphabetically sorted list of 3-letter language code alternatives
to 2-letter language codes that cannot be derived using the method described above.
Each 3-letter code is followed by its 1-byte langID.`,
	`
altLangIndex is used to convert indexes in altLangISO3 to langIDs.`,
	`
langAliasMap maps langIDs to their suggested replacements.`,
	`
script is an alphabetically sorted list of ISO 15924 codes. The index
of the script in the string, divided by 4, is the internal scriptID.`,
	`
isoRegionOffset needs to be added to the index of regionISO to obtain the regionID
for 2-letter ISO codes. (The first isoRegionOffset regionIDs are reserved for
the UN.M49 codes used for groups.)`,
	`
regionISO holds a list of alphabetically sorted 2-letter ISO region codes.
Each 2-letter codes is followed by two bytes with the following meaning:
    - [A-Z}{2}: the first letter of the 2-letter code plus these two 
                letters form the 3-letter ISO code.
    - 0, n:     index into altRegionISO3.`,
	`
regionTypes defines the status of a region for various standards.`,
	`
m49 maps regionIDs to UN.M49 codes. The first isoRegionOffset entries are
codes indicating collections of regions.`,
	`
m49Index gives indexes into fromM49 based on the three most significant bits
of a 10-bit UN.M49 code. To search an UN.M49 code in fromM49, search in
   fromM49[m49Index[msb39(code)]:m49Index[msb3(code)+1]]
for an entry where the first 7 bits match the 7 lsb of the UN.M49 code.
The region code is stored in the 9 lsb of the indexed value.`,
	`
fromM49 contains entries to map UN.M49 codes to regions. See m49Index for details.`,
	`
altRegionISO3 holds a list of 3-letter region codes that cannot be
mapped to 2-letter codes using the default algorithm. This is a short list.`,
	`
altRegionIDs holds a list of regionIDs the positions of which match those
of the 3-letter ISO codes in altRegionISO3.`,
	`
variantNumSpecialized is the number of specialized variants in variants.`,
	`
suppressScript is an index from langID to the dominant script for that language,
if it exists.  If a script is given, it should be suppressed from the language tag.`,
	`
likelyLang is a lookup table, indexed by langID, for the most likely
scripts and regions given incomplete information. If more entries exist for a
given language, region and script are the index and size respectively
of the list in likelyLangList.`,
	`
likelyLangList holds lists info associated with likelyLang.`,
	`
likelyRegion is a lookup table, indexed by regionID, for the most likely
languages and scripts given incomplete information. If more entries exist
for a given regionID, lang and script are the index and size respectively
of the list in likelyRegionList.
TODO: exclude containers and user-definable regions from the list.`,
	`
likelyRegionList holds lists info associated with likelyRegion.`,
	`
likelyScript is a lookup table, indexed by scriptID, for the most likely
languages and regions given a script.`,
	`
matchLang holds pairs of langIDs of base languages that are typically
mutually intelligible. Each pair is associated with a confidence and
whether the intelligibility goes one or both ways.`,
	`
matchScript holds pairs of scriptIDs where readers of one script
can typically also read the other. Each is associated with a confidence.`,
	`
nRegionGroups is the number of region groups.`,
	`
regionInclusion maps region identifiers to sets of regions in regionInclusionBits,
where each set holds all groupings that are directly connected in a region
containment graph.`,
	`
regionInclusionBits is an array of bit vectors where every vector represents
a set of region groupings.  These sets are used to compute the distance
between two regions for the purpose of language matching.`,
	`
regionInclusionNext marks, for each entry in regionInclusionBits, the set of
all groups that are reachable from the groups set in the respective entry.`,
}

// TODO: consider changing some of these structures to tries. This can reduce
// memory, but may increase the need for memory allocations. This could be
// mitigated if we can piggyback on language tags for common cases.

func failOnError(e error) {
	if e != nil {
		log.Panic(e)
	}
}

type setType int

const (
	Indexed setType = 1 + iota // all elements must be of same size
	Linear
)

type stringSet struct {
	s              []string
	sorted, frozen bool

	// We often need to update values after the creation of an index is completed.
	// We include a convenience map for keeping track of this.
	update map[string]string
	typ    setType // used for checking.
}

func (ss *stringSet) clone() stringSet {
	c := *ss
	c.s = append([]string(nil), c.s...)
	return c
}

func (ss *stringSet) setType(t setType) {
	if ss.typ != t && ss.typ != 0 {
		log.Panicf("type %d cannot be assigned as it was already %d", t, ss.typ)
	}
}

// parse parses a whitespace-separated string and initializes ss with its
// components.
func (ss *stringSet) parse(s string) {
	scan := bufio.NewScanner(strings.NewReader(s))
	scan.Split(bufio.ScanWords)
	for scan.Scan() {
		ss.add(scan.Text())
	}
}

func (ss *stringSet) assertChangeable() {
	if ss.frozen {
		log.Panic("attempt to modify a frozen stringSet")
	}
}

func (ss *stringSet) add(s string) {
	ss.assertChangeable()
	ss.s = append(ss.s, s)
	ss.sorted = ss.frozen
}

func (ss *stringSet) freeze() {
	ss.compact()
	ss.frozen = true
}

func (ss *stringSet) compact() {
	if ss.sorted {
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
	ss.sorted = ss.frozen
}

type funcSorter struct {
	fn func(a, b string) bool
	sort.StringSlice
}

func (s funcSorter) Less(i, j int) bool {
	return s.fn(s.StringSlice[i], s.StringSlice[j])
}

func (ss *stringSet) sortFunc(f func(a, b string) bool) {
	ss.compact()
	sort.Sort(funcSorter{f, sort.StringSlice(ss.s)})
}

func (ss *stringSet) remove(s string) {
	ss.assertChangeable()
	if i, ok := ss.find(s); ok {
		copy(ss.s[i:], ss.s[i+1:])
		ss.s = ss.s[:len(ss.s)-1]
	}
}

func (ss *stringSet) replace(ol, nu string) {
	ss.s[ss.index(ol)] = nu
	ss.sorted = ss.frozen
}

func (ss *stringSet) index(s string) int {
	ss.setType(Indexed)
	i, ok := ss.find(s)
	if !ok {
		if i < len(ss.s) {
			log.Panicf("find: item %q is not in list. Closest match is %q.", s, ss.s[i])
		}
		log.Panicf("find: item %q is not in list", s)

	}
	return i
}

func (ss *stringSet) find(s string) (int, bool) {
	ss.compact()
	i := sort.SearchStrings(ss.s, s)
	return i, i != len(ss.s) && ss.s[i] == s
}

func (ss *stringSet) slice() []string {
	ss.compact()
	return ss.s
}

func (ss *stringSet) updateLater(v, key string) {
	if ss.update == nil {
		ss.update = map[string]string{}
	}
	ss.update[v] = key
}

// join joins the string and ensures that all entries are of the same length.
func (ss *stringSet) join() string {
	ss.setType(Indexed)
	n := len(ss.s[0])
	for _, s := range ss.s {
		if len(s) != n {
			log.Panicf("join: not all entries are of the same length: %q", s)
		}
	}
	ss.s = append(ss.s, strings.Repeat("\xff", n))
	return strings.Join(ss.s, "")
}

// ianaEntry holds information for an entry in the IANA Language Subtag Repository.
// All types use the same entry.
// See http://tools.ietf.org/html/bcp47#section-5.1 for a description of the various
// fields.
type ianaEntry struct {
	typ            string
	description    []string
	scope          string
	added          string
	preferred      string
	deprecated     string
	suppressScript string
	macro          string
	prefix         []string
}

type builder struct {
	w    *gen.CodeWriter
	hw   io.Writer // MultiWriter for w and w.Hash
	data *cldr.CLDR
	supp *cldr.SupplementalData

	// indices
	locale      stringSet // common locales
	lang        stringSet // canonical language ids (2 or 3 letter ISO codes) with data
	langNoIndex stringSet // 3-letter ISO codes with no associated data
	script      stringSet // 4-letter ISO codes
	region      stringSet // 2-letter ISO or 3-digit UN M49 codes
	variant     stringSet // 4-8-alphanumeric variant code.

	// Region codes that are groups with their corresponding group IDs.
	groups map[int]index

	// langInfo
	registry map[string]*ianaEntry
}

type index uint

func newBuilder(w *gen.CodeWriter) *builder {
	r := gen.OpenCLDRCoreZip()
	defer r.Close()
	d := &cldr.Decoder{}
	data, err := d.DecodeZip(r)
	failOnError(err)
	b := builder{
		w:    w,
		hw:   io.MultiWriter(w, w.Hash),
		data: data,
		supp: data.Supplemental(),
	}
	b.parseRegistry()
	return &b
}

func (b *builder) parseRegistry() {
	r := gen.OpenIANAFile("assignments/language-subtag-registry")
	defer r.Close()
	b.registry = make(map[string]*ianaEntry)

	scan := bufio.NewScanner(r)
	scan.Split(bufio.ScanWords)
	var record *ianaEntry
	for more := scan.Scan(); more; {
		key := scan.Text()
		more = scan.Scan()
		value := scan.Text()
		switch key {
		case "Type:":
			record = &ianaEntry{typ: value}
		case "Subtag:", "Tag:":
			if s := strings.SplitN(value, "..", 2); len(s) > 1 {
				for a := s[0]; a <= s[1]; a = inc(a) {
					b.addToRegistry(a, record)
				}
			} else {
				b.addToRegistry(value, record)
			}
		case "Suppress-Script:":
			record.suppressScript = value
		case "Added:":
			record.added = value
		case "Deprecated:":
			record.deprecated = value
		case "Macrolanguage:":
			record.macro = value
		case "Preferred-Value:":
			record.preferred = value
		case "Prefix:":
			record.prefix = append(record.prefix, value)
		case "Scope:":
			record.scope = value
		case "Description:":
			buf := []byte(value)
			for more = scan.Scan(); more; more = scan.Scan() {
				b := scan.Bytes()
				if b[0] == '%' || b[len(b)-1] == ':' {
					break
				}
				buf = append(buf, ' ')
				buf = append(buf, b...)
			}
			record.description = append(record.description, string(buf))
			continue
		default:
			continue
		}
		more = scan.Scan()
	}
	if scan.Err() != nil {
		log.Panic(scan.Err())
	}
}

func (b *builder) addToRegistry(key string, entry *ianaEntry) {
	if info, ok := b.registry[key]; ok {
		if info.typ != "language" || entry.typ != "extlang" {
			log.Fatalf("parseRegistry: tag %q already exists", key)
		}
	} else {
		b.registry[key] = entry
	}
}

var commentIndex = make(map[string]string)

func init() {
	for _, s := range comment {
		key := strings.TrimSpace(strings.SplitN(s, " ", 2)[0])
		commentIndex[key] = s
	}
}

func (b *builder) comment(name string) {
	if s := commentIndex[name]; len(s) > 0 {
		b.w.WriteComment(s)
	} else {
		fmt.Fprintln(b.w)
	}
}

func (b *builder) pf(f string, x ...interface{}) {
	fmt.Fprintf(b.hw, f, x...)
	fmt.Fprint(b.hw, "\n")
}

func (b *builder) p(x ...interface{}) {
	fmt.Fprintln(b.hw, x...)
}

func (b *builder) addSize(s int) {
	b.w.Size += s
	b.pf("// Size: %d bytes", s)
}

func (b *builder) writeConst(name string, x interface{}) {
	b.comment(name)
	b.w.WriteConst(name, x)
}

// writeConsts computes f(v) for all v in values and writes the results
// as constants named _v to a single constant block.
func (b *builder) writeConsts(f func(string) int, values ...string) {
	b.pf("const (")
	for _, v := range values {
		b.pf("\t_%s = %v", v, f(v))
	}
	b.pf(")")
}

// writeType writes the type of the given value, which must be a struct.
func (b *builder) writeType(value interface{}) {
	b.comment(reflect.TypeOf(value).Name())
	b.w.WriteType(value)
}

func (b *builder) writeSlice(name string, ss interface{}) {
	b.writeSliceAddSize(name, 0, ss)
}

func (b *builder) writeSliceAddSize(name string, extraSize int, ss interface{}) {
	b.comment(name)
	b.w.Size += extraSize
	v := reflect.ValueOf(ss)
	t := v.Type().Elem()
	b.pf("// Size: %d bytes, %d elements", v.Len()*int(t.Size())+extraSize, v.Len())

	fmt.Fprintf(b.w, "var %s = ", name)
	b.w.WriteArray(ss)
	b.p()
}

type fromTo struct {
	from, to uint16
}

func (b *builder) writeSortedMap(name string, ss *stringSet, index func(s string) uint16) {
	ss.sortFunc(func(a, b string) bool {
		return index(a) < index(b)
	})
	m := []fromTo{}
	for _, s := range ss.s {
		m = append(m, fromTo{index(s), index(ss.update[s])})
	}
	b.writeSlice(name, m)
}

const base = 'z' - 'a' + 1

func strToInt(s string) uint {
	v := uint(0)
	for i := 0; i < len(s); i++ {
		v *= base
		v += uint(s[i] - 'a')
	}
	return v
}

// converts the given integer to the original ASCII string passed to strToInt.
// len(s) must match the number of characters obtained.
func intToStr(v uint, s []byte) {
	for i := len(s) - 1; i >= 0; i-- {
		s[i] = byte(v%base) + 'a'
		v /= base
	}
}

func (b *builder) writeBitVector(name string, ss []string) {
	vec := make([]uint8, int(math.Ceil(math.Pow(base, float64(len(ss[0])))/8)))
	for _, s := range ss {
		v := strToInt(s)
		vec[v/8] |= 1 << (v % 8)
	}
	b.writeSlice(name, vec)
}

// TODO: convert this type into a list or two-stage trie.
func (b *builder) writeMapFunc(name string, m map[string]string, f func(string) uint16) {
	b.comment(name)
	v := reflect.ValueOf(m)
	sz := v.Len() * (2 + int(v.Type().Key().Size()))
	for _, k := range m {
		sz += len(k)
	}
	b.addSize(sz)
	keys := []string{}
	b.pf(`var %s = map[string]uint16{`, name)
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		b.pf("\t%q: %v,", k, f(m[k]))
	}
	b.p("}")
}

func (b *builder) writeMap(name string, m interface{}) {
	b.comment(name)
	v := reflect.ValueOf(m)
	sz := v.Len() * (2 + int(v.Type().Key().Size()) + int(v.Type().Elem().Size()))
	b.addSize(sz)
	f := strings.FieldsFunc(fmt.Sprintf("%#v", m), func(r rune) bool {
		return strings.IndexRune("{}, ", r) != -1
	})
	sort.Strings(f[1:])
	b.pf(`var %s = %s{`, name, f[0])
	for _, kv := range f[1:] {
		b.pf("\t%s,", kv)
	}
	b.p("}")
}

func (b *builder) langIndex(s string) uint16 {
	if s == "und" {
		return 0
	}
	if i, ok := b.lang.find(s); ok {
		return uint16(i)
	}
	return uint16(strToInt(s)) + uint16(len(b.lang.s))
}

// inc advances the string to its lexicographical successor.
func inc(s string) string {
	const maxTagLength = 4
	var buf [maxTagLength]byte
	intToStr(strToInt(strings.ToLower(s))+1, buf[:len(s)])
	for i := 0; i < len(s); i++ {
		if s[i] <= 'Z' {
			buf[i] -= 'a' - 'A'
		}
	}
	return string(buf[:len(s)])
}

func (b *builder) parseIndices() {
	meta := b.supp.Metadata

	for k, v := range b.registry {
		var ss *stringSet
		switch v.typ {
		case "language":
			if len(k) == 2 || v.suppressScript != "" || v.scope == "special" {
				b.lang.add(k)
				continue
			} else {
				ss = &b.langNoIndex
			}
		case "region":
			ss = &b.region
		case "script":
			ss = &b.script
		case "variant":
			ss = &b.variant
		default:
			continue
		}
		ss.add(k)
	}
	// Include any language for which there is data.
	for _, lang := range b.data.Locales() {
		if x := b.data.RawLDML(lang); false ||
			x.LocaleDisplayNames != nil ||
			x.Characters != nil ||
			x.Delimiters != nil ||
			x.Measurement != nil ||
			x.Dates != nil ||
			x.Numbers != nil ||
			x.Units != nil ||
			x.ListPatterns != nil ||
			x.Collations != nil ||
			x.Segmentations != nil ||
			x.Rbnf != nil ||
			x.Annotations != nil ||
			x.Metadata != nil {

			from := strings.Split(lang, "_")
			if lang := from[0]; lang != "root" {
				b.lang.add(lang)
			}
		}
	}
	// Include locales for plural rules, which uses a different structure.
	for _, plurals := range b.data.Supplemental().Plurals {
		for _, rules := range plurals.PluralRules {
			for _, lang := range strings.Split(rules.Locales, " ") {
				if lang = strings.Split(lang, "_")[0]; lang != "root" {
					b.lang.add(lang)
				}
			}
		}
	}
	// Include languages in likely subtags.
	for _, m := range b.supp.LikelySubtags.LikelySubtag {
		from := strings.Split(m.From, "_")
		b.lang.add(from[0])
	}
	// Include ISO-639 alpha-3 bibliographic entries.
	for _, a := range meta.Alias.LanguageAlias {
		if a.Reason == "bibliographic" {
			b.langNoIndex.add(a.Type)
		}
	}
	// Include regions in territoryAlias (not all are in the IANA registry!)
	for _, reg := range b.supp.Metadata.Alias.TerritoryAlias {
		if len(reg.Type) == 2 {
			b.region.add(reg.Type)
		}
	}

	for _, s := range b.lang.s {
		if len(s) == 3 {
			b.langNoIndex.remove(s)
		}
	}
	b.writeConst("numLanguages", len(b.lang.slice())+len(b.langNoIndex.slice()))
	b.writeConst("numScripts", len(b.script.slice()))
	b.writeConst("numRegions", len(b.region.slice()))

	// Add dummy codes at the start of each list to represent "unspecified".
	b.lang.add("---")
	b.script.add("----")
	b.region.add("---")

	// common locales
	b.locale.parse(meta.DefaultContent.Locales)
}

// TODO: region inclusion data will probably not be use used in future matchers.

func (b *builder) computeRegionGroups() {
	b.groups = make(map[int]index)

	// Create group indices.
	for i := 1; b.region.s[i][0] < 'A'; i++ { // Base M49 indices on regionID.
		b.groups[i] = index(len(b.groups))
	}
	for _, g := range b.supp.TerritoryContainment.Group {
		// Skip UN and EURO zone as they are flattening the containment
		// relationship.
		if g.Type == "EZ" || g.Type == "UN" {
			continue
		}
		group := b.region.index(g.Type)
		if _, ok := b.groups[group]; !ok {
			b.groups[group] = index(len(b.groups))
		}
	}
	if len(b.groups) > 32 {
		log.Fatalf("only 32 groups supported, found %d", len(b.groups))
	}
	b.writeConst("nRegionGroups", len(b.groups))
}

var langConsts = []string{
	"af", "am", "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es",
	"et", "fa", "fi", "fil", "fr", "gu", "he", "hi", "hr", "hu", "hy", "id", "is",
	"it", "ja", "ka", "kk", "km", "kn", "ko", "ky", "lo", "lt", "lv", "mk", "ml",
	"mn", "mo", "mr", "ms", "mul", "my", "nb", "ne", "nl", "no", "pa", "pl", "pt",
	"ro", "ru", "sh", "si", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th",
	"tl", "tn", "tr", "uk", "ur", "uz", "vi", "zh", "zu",

	// constants for grandfathered tags (if not already defined)
	"jbo", "ami", "bnn", "hak", "tlh", "lb", "nv", "pwn", "tao", "tay", "tsu",
	"nn", "sfb", "vgt", "sgg", "cmn", "nan", "hsn",
}

// writeLanguage generates all tables needed for language canonicalization.
func (b *builder) writeLanguage() {
	meta := b.supp.Metadata

	b.writeConst("nonCanonicalUnd", b.lang.index("und"))
	b.writeConsts(func(s string) int { return int(b.langIndex(s)) }, langConsts...)
	b.writeConst("langPrivateStart", b.langIndex("qaa"))
	b.writeConst("langPrivateEnd", b.langIndex("qtz"))

	// Get language codes that need to be mapped (overlong 3-letter codes,
	// deprecated 2-letter codes, legacy and grandfathered tags.)
	langAliasMap := stringSet{}
	aliasTypeMap := map[string]langAliasType{}

	// altLangISO3 get the alternative ISO3 names that need to be mapped.
	altLangISO3 := stringSet{}
	// Add dummy start to avoid the use of index 0.
	altLangISO3.add("---")
	altLangISO3.updateLater("---", "aa")

	lang := b.lang.clone()
	for _, a := range meta.Alias.LanguageAlias {
		if a.Replacement == "" {
			a.Replacement = "und"
		}
		// TODO: support mapping to tags
		repl := strings.SplitN(a.Replacement, "_", 2)[0]
		if a.Reason == "overlong" {
			if len(a.Replacement) == 2 && len(a.Type) == 3 {
				lang.updateLater(a.Replacement, a.Type)
			}
		} else if len(a.Type) <= 3 {
			switch a.Reason {
			case "macrolanguage":
				aliasTypeMap[a.Type] = langMacro
			case "deprecated":
				// handled elsewhere
				continue
			case "bibliographic", "legacy":
				if a.Type == "no" {
					continue
				}
				aliasTypeMap[a.Type] = langLegacy
			default:
				log.Fatalf("new %s alias: %s", a.Reason, a.Type)
			}
			langAliasMap.add(a.Type)
			langAliasMap.updateLater(a.Type, repl)
		}
	}
	// Manually add the mapping of "nb" (Norwegian) to its macro language.
	// This can be removed if CLDR adopts this change.
	langAliasMap.add("nb")
	langAliasMap.updateLater("nb", "no")
	aliasTypeMap["nb"] = langMacro

	for k, v := range b.registry {
		// Also add deprecated values for 3-letter ISO codes, which CLDR omits.
		if v.typ == "language" && v.deprecated != "" && v.preferred != "" {
			langAliasMap.add(k)
			langAliasMap.updateLater(k, v.preferred)
			aliasTypeMap[k] = langDeprecated
		}
	}
	// Fix CLDR mappings.
	lang.updateLater("tl", "tgl")
	lang.updateLater("sh", "hbs")
	lang.updateLater("mo", "mol")
	lang.updateLater("no", "nor")
	lang.updateLater("tw", "twi")
	lang.updateLater("nb", "nob")
	lang.updateLater("ak", "aka")
	lang.updateLater("bh", "bih")

	// Ensure that each 2-letter code is matched with a 3-letter code.
	for _, v := range lang.s[1:] {
		s, ok := lang.update[v]
		if !ok {
			if s, ok = lang.update[langAliasMap.update[v]]; !ok {
				continue
			}
			lang.update[v] = s
		}
		if v[0] != s[0] {
			altLangISO3.add(s)
			altLangISO3.updateLater(s, v)
		}
	}

	// Complete canonicalized language tags.
	lang.freeze()
	for i, v := range lang.s {
		// We can avoid these manual entries by using the IANA registry directly.
		// Seems easier to update the list manually, as changes are rare.
		// The panic in this loop will trigger if we miss an entry.
		add := ""
		if s, ok := lang.update[v]; ok {
			if s[0] == v[0] {
				add = s[1:]
			} else {
				add = string([]byte{0, byte(altLangISO3.index(s))})
			}
		} else if len(v) == 3 {
			add = "\x00"
		} else {
			log.Panicf("no data for long form of %q", v)
		}
		lang.s[i] += add
	}
	b.writeConst("lang", tag.Index(lang.join()))

	b.writeConst("langNoIndexOffset", len(b.lang.s))

	// space of all valid 3-letter language identifiers.
	b.writeBitVector("langNoIndex", b.langNoIndex.slice())

	altLangIndex := []uint16{}
	for i, s := range altLangISO3.slice() {
		altLangISO3.s[i] += string([]byte{byte(len(altLangIndex))})
		if i > 0 {
			idx := b.lang.index(altLangISO3.update[s])
			altLangIndex = append(altLangIndex, uint16(idx))
		}
	}
	b.writeConst("altLangISO3", tag.Index(altLangISO3.join()))
	b.writeSlice("altLangIndex", altLangIndex)

	b.writeSortedMap("langAliasMap", &langAliasMap, b.langIndex)
	types := make([]langAliasType, len(langAliasMap.s))
	for i, s := range langAliasMap.s {
		types[i] = aliasTypeMap[s]
	}
	b.writeSlice("langAliasTypes", types)
}

var scriptConsts = []string{
	"Latn", "Hani", "Hans", "Hant", "Qaaa", "Qaai", "Qabx", "Zinh", "Zyyy",
	"Zzzz",
}

func (b *builder) writeScript() {
	b.writeConsts(b.script.index, scriptConsts...)
	b.writeConst("script", tag.Index(b.script.join()))

	supp := make([]uint8, len(b.lang.slice()))
	for i, v := range b.lang.slice()[1:] {
		if sc := b.registry[v].suppressScript; sc != "" {
			supp[i+1] = uint8(b.script.index(sc))
		}
	}
	b.writeSlice("suppressScript", supp)

	// There is only one deprecated script in CLDR. This value is hard-coded.
	// We check here if the code must be updated.
	for _, a := range b.supp.Metadata.Alias.ScriptAlias {
		if a.Type != "Qaai" {
			log.Panicf("unexpected deprecated stript %q", a.Type)
		}
	}
}

func parseM49(s string) int16 {
	if len(s) == 0 {
		return 0
	}
	v, err := strconv.ParseUint(s, 10, 10)
	failOnError(err)
	return int16(v)
}

var regionConsts = []string{
	"001", "419", "BR", "CA", "ES", "GB", "MD", "PT", "UK", "US",
	"ZZ", "XA", "XC", "XK", // Unofficial tag for Kosovo.
}

func (b *builder) writeRegion() {
	b.writeConsts(b.region.index, regionConsts...)

	isoOffset := b.region.index("AA")
	m49map := make([]int16, len(b.region.slice()))
	fromM49map := make(map[int16]int)
	altRegionISO3 := ""
	altRegionIDs := []uint16{}

	b.writeConst("isoRegionOffset", isoOffset)

	// 2-letter region lookup and mapping to numeric codes.
	regionISO := b.region.clone()
	regionISO.s = regionISO.s[isoOffset:]
	regionISO.sorted = false

	regionTypes := make([]byte, len(b.region.s))

	// Is the region valid BCP 47?
	for s, e := range b.registry {
		if len(s) == 2 && s == strings.ToUpper(s) {
			i := b.region.index(s)
			for _, d := range e.description {
				if strings.Contains(d, "Private use") {
					regionTypes[i] = iso3166UserAssigned
				}
			}
			regionTypes[i] |= bcp47Region
		}
	}

	// Is the region a valid ccTLD?
	r := gen.OpenIANAFile("domains/root/db")
	defer r.Close()

	buf, err := ioutil.ReadAll(r)
	failOnError(err)
	re := regexp.MustCompile(`"/domains/root/db/([a-z]{2}).html"`)
	for _, m := range re.FindAllSubmatch(buf, -1) {
		i := b.region.index(strings.ToUpper(string(m[1])))
		regionTypes[i] |= ccTLD
	}

	b.writeSlice("regionTypes", regionTypes)

	iso3Set := make(map[string]int)
	update := func(iso2, iso3 string) {
		i := regionISO.index(iso2)
		if j, ok := iso3Set[iso3]; !ok && iso3[0] == iso2[0] {
			regionISO.s[i] += iso3[1:]
			iso3Set[iso3] = -1
		} else {
			if ok && j >= 0 {
				regionISO.s[i] += string([]byte{0, byte(j)})
			} else {
				iso3Set[iso3] = len(altRegionISO3)
				regionISO.s[i] += string([]byte{0, byte(len(altRegionISO3))})
				altRegionISO3 += iso3
				altRegionIDs = append(altRegionIDs, uint16(isoOffset+i))
			}
		}
	}
	for _, tc := range b.supp.CodeMappings.TerritoryCodes {
		i := regionISO.index(tc.Type) + isoOffset
		if d := m49map[i]; d != 0 {
			log.Panicf("%s found as a duplicate UN.M49 code of %03d", tc.Numeric, d)
		}
		m49 := parseM49(tc.Numeric)
		m49map[i] = m49
		if r := fromM49map[m49]; r == 0 {
			fromM49map[m49] = i
		} else if r != i {
			dep := b.registry[regionISO.s[r-isoOffset]].deprecated
			if t := b.registry[tc.Type]; t != nil && dep != "" && (t.deprecated == "" || t.deprecated > dep) {
				fromM49map[m49] = i
			}
		}
	}
	for _, ta := range b.supp.Metadata.Alias.TerritoryAlias {
		if len(ta.Type) == 3 && ta.Type[0] <= '9' && len(ta.Replacement) == 2 {
			from := parseM49(ta.Type)
			if r := fromM49map[from]; r == 0 {
				fromM49map[from] = regionISO.index(ta.Replacement) + isoOffset
			}
		}
	}
	for _, tc := range b.supp.CodeMappings.TerritoryCodes {
		if len(tc.Alpha3) == 3 {
			update(tc.Type, tc.Alpha3)
		}
	}
	// This entries are not included in territoryCodes. Mostly 3-letter variants
	// of deleted codes and an entry for QU.
	for _, m := range []struct{ iso2, iso3 string }{
		{"CT", "CTE"},
		{"DY", "DHY"},
		{"HV", "HVO"},
		{"JT", "JTN"},
		{"MI", "MID"},
		{"NH", "NHB"},
		{"NQ", "ATN"},
		{"PC", "PCI"},
		{"PU", "PUS"},
		{"PZ", "PCZ"},
		{"RH", "RHO"},
		{"VD", "VDR"},
		{"WK", "WAK"},
		// These three-letter codes are used for others as well.
		{"FQ", "ATF"},
	} {
		update(m.iso2, m.iso3)
	}
	for i, s := range regionISO.s {
		if len(s) != 4 {
			regionISO.s[i] = s + "  "
		}
	}
	b.writeConst("regionISO", tag.Index(regionISO.join()))
	b.writeConst("altRegionISO3", altRegionISO3)
	b.writeSlice("altRegionIDs", altRegionIDs)

	// Create list of deprecated regions.
	// TODO: consider inserting SF -> FI. Not included by CLDR, but is the only
	// Transitionally-reserved mapping not included.
	regionOldMap := stringSet{}
	// Include regions in territoryAlias (not all are in the IANA registry!)
	for _, reg := range b.supp.Metadata.Alias.TerritoryAlias {
		if len(reg.Type) == 2 && reg.Reason == "deprecated" && len(reg.Replacement) == 2 {
			regionOldMap.add(reg.Type)
			regionOldMap.updateLater(reg.Type, reg.Replacement)
			i, _ := regionISO.find(reg.Type)
			j, _ := regionISO.find(reg.Replacement)
			if k := m49map[i+isoOffset]; k == 0 {
				m49map[i+isoOffset] = m49map[j+isoOffset]
			}
		}
	}
	b.writeSortedMap("regionOldMap", &regionOldMap, func(s string) uint16 {
		return uint16(b.region.index(s))
	})
	// 3-digit region lookup, groupings.
	for i := 1; i < isoOffset; i++ {
		m := parseM49(b.region.s[i])
		m49map[i] = m
		fromM49map[m] = i
	}
	b.writeSlice("m49", m49map)

	const (
		searchBits = 7
		regionBits = 9
	)
	if len(m49map) >= 1<<regionBits {
		log.Fatalf("Maximum number of regions exceeded: %d > %d", len(m49map), 1<<regionBits)
	}
	m49Index := [9]int16{}
	fromM49 := []uint16{}
	m49 := []int{}
	for k, _ := range fromM49map {
		m49 = append(m49, int(k))
	}
	sort.Ints(m49)
	for _, k := range m49[1:] {
		val := (k & (1<<searchBits - 1)) << regionBits
		fromM49 = append(fromM49, uint16(val|fromM49map[int16(k)]))
		m49Index[1:][k>>searchBits] = int16(len(fromM49))
	}
	b.writeSlice("m49Index", m49Index)
	b.writeSlice("fromM49", fromM49)
}

const (
	// TODO: put these lists in regionTypes as user data? Could be used for
	// various optimizations and refinements and could be exposed in the API.
	iso3166Except = "AC CP DG EA EU FX IC SU TA UK"
	iso3166Trans  = "AN BU CS NT TP YU ZR" // SF is not in our set of Regions.
	// DY and RH are actually not deleted, but indeterminately reserved.
	iso3166DelCLDR = "CT DD DY FQ HV JT MI NH NQ PC PU PZ RH VD WK YD"
)

const (
	iso3166UserAssigned = 1 << iota
	ccTLD
	bcp47Region
)

func find(list []string, s string) int {
	for i, t := range list {
		if t == s {
			return i
		}
	}
	return -1
}

// writeVariants generates per-variant information and creates a map from variant
// name to index value. We assign index values such that sorting multiple
// variants by index value will result in the correct order.
// There are two types of variants: specialized and general. Specialized variants
// are only applicable to certain language or language-script pairs. Generalized
// variants apply to any language. Generalized variants always sort after
// specialized variants.  We will therefore always assign a higher index value
// to a generalized variant than any other variant. Generalized variants are
// sorted alphabetically among themselves.
// Specialized variants may also sort after other specialized variants. Such
// variants will be ordered after any of the variants they may follow.
// We assume that if a variant x is followed by a variant y, then for any prefix
// p of x, p-x is a prefix of y. This allows us to order tags based on the
// maximum of the length of any of its prefixes.
// TODO: it is possible to define a set of Prefix values on variants such that
// a total order cannot be defined to the point that this algorithm breaks.
// In other words, we cannot guarantee the same order of variants for the
// future using the same algorithm or for non-compliant combinations of
// variants. For this reason, consider using simple alphabetic sorting
// of variants and ignore Prefix restrictions altogether.
func (b *builder) writeVariant() {
	generalized := stringSet{}
	specialized := stringSet{}
	specializedExtend := stringSet{}
	// Collate the variants by type and check assumptions.
	for _, v := range b.variant.slice() {
		e := b.registry[v]
		if len(e.prefix) == 0 {
			generalized.add(v)
			continue
		}
		c := strings.Split(e.prefix[0], "-")
		hasScriptOrRegion := false
		if len(c) > 1 {
			_, hasScriptOrRegion = b.script.find(c[1])
			if !hasScriptOrRegion {
				_, hasScriptOrRegion = b.region.find(c[1])

			}
		}
		if len(c) == 1 || len(c) == 2 && hasScriptOrRegion {
			// Variant is preceded by a language.
			specialized.add(v)
			continue
		}
		// Variant is preceded by another variant.
		specializedExtend.add(v)
		prefix := c[0] + "-"
		if hasScriptOrRegion {
			prefix += c[1]
		}
		for _, p := range e.prefix {
			// Verify that the prefix minus the last element is a prefix of the
			// predecessor element.
			i := strings.LastIndex(p, "-")
			pred := b.registry[p[i+1:]]
			if find(pred.prefix, p[:i]) < 0 {
				log.Fatalf("prefix %q for variant %q not consistent with predecessor spec", p, v)
			}
			// The sorting used below does not work in the general case. It works
			// if we assume that variants that may be followed by others only have
			// prefixes of the same length. Verify this.
			count := strings.Count(p[:i], "-")
			for _, q := range pred.prefix {
				if c := strings.Count(q, "-"); c != count {
					log.Fatalf("variant %q preceding %q has a prefix %q of size %d; want %d", p[i+1:], v, q, c, count)
				}
			}
			if !strings.HasPrefix(p, prefix) {
				log.Fatalf("prefix %q of variant %q should start with %q", p, v, prefix)
			}
		}
	}

	// Sort extended variants.
	a := specializedExtend.s
	less := func(v, w string) bool {
		// Sort by the maximum number of elements.
		maxCount := func(s string) (max int) {
			for _, p := range b.registry[s].prefix {
				if c := strings.Count(p, "-"); c > max {
					max = c
				}
			}
			return
		}
		if cv, cw := maxCount(v), maxCount(w); cv != cw {
			return cv < cw
		}
		// Sort by name as tie breaker.
		return v < w
	}
	sort.Sort(funcSorter{less, sort.StringSlice(a)})
	specializedExtend.frozen = true

	// Create index from variant name to index.
	variantIndex := make(map[string]uint8)
	add := func(s []string) {
		for _, v := range s {
			variantIndex[v] = uint8(len(variantIndex))
		}
	}
	add(specialized.slice())
	add(specializedExtend.s)
	numSpecialized := len(variantIndex)
	add(generalized.slice())
	if n := len(variantIndex); n > 255 {
		log.Fatalf("maximum number of variants exceeded: was %d; want <= 255", n)
	}
	b.writeMap("variantIndex", variantIndex)
	b.writeConst("variantNumSpecialized", numSpecialized)
}

func (b *builder) writeLanguageInfo() {
}

// writeLikelyData writes tables that are used both for finding parent relations and for
// language matching.  Each entry contains additional bits to indicate the status of the
// data to know when it cannot be used for parent relations.
func (b *builder) writeLikelyData() {
	const (
		isList = 1 << iota
		scriptInFrom
		regionInFrom
	)
	type ( // generated types
		likelyScriptRegion struct {
			region uint16
			script uint8
			flags  uint8
		}
		likelyLangScript struct {
			lang   uint16
			script uint8
			flags  uint8
		}
		likelyLangRegion struct {
			lang   uint16
			region uint16
		}
		// likelyTag is used for getting likely tags for group regions, where
		// the likely region might be a region contained in the group.
		likelyTag struct {
			lang   uint16
			region uint16
			script uint8
		}
	)
	var ( // generated variables
		likelyRegionGroup = make([]likelyTag, len(b.groups))
		likelyLang        = make([]likelyScriptRegion, len(b.lang.s))
		likelyRegion      = make([]likelyLangScript, len(b.region.s))
		likelyScript      = make([]likelyLangRegion, len(b.script.s))
		likelyLangList    = []likelyScriptRegion{}
		likelyRegionList  = []likelyLangScript{}
	)
	type fromTo struct {
		from, to []string
	}
	langToOther := map[int][]fromTo{}
	regionToOther := map[int][]fromTo{}
	for _, m := range b.supp.LikelySubtags.LikelySubtag {
		from := strings.Split(m.From, "_")
		to := strings.Split(m.To, "_")
		if len(to) != 3 {
			log.Fatalf("invalid number of subtags in %q: found %d, want 3", m.To, len(to))
		}
		if len(from) > 3 {
			log.Fatalf("invalid number of subtags: found %d, want 1-3", len(from))
		}
		if from[0] != to[0] && from[0] != "und" {
			log.Fatalf("unexpected language change in expansion: %s -> %s", from, to)
		}
		if len(from) == 3 {
			if from[2] != to[2] {
				log.Fatalf("unexpected region change in expansion: %s -> %s", from, to)
			}
			if from[0] != "und" {
				log.Fatalf("unexpected fully specified from tag: %s -> %s", from, to)
			}
		}
		if len(from) == 1 || from[0] != "und" {
			id := 0
			if from[0] != "und" {
				id = b.lang.index(from[0])
			}
			langToOther[id] = append(langToOther[id], fromTo{from, to})
		} else if len(from) == 2 && len(from[1]) == 4 {
			sid := b.script.index(from[1])
			likelyScript[sid].lang = uint16(b.langIndex(to[0]))
			likelyScript[sid].region = uint16(b.region.index(to[2]))
		} else {
			r := b.region.index(from[len(from)-1])
			if id, ok := b.groups[r]; ok {
				if from[0] != "und" {
					log.Fatalf("region changed unexpectedly: %s -> %s", from, to)
				}
				likelyRegionGroup[id].lang = uint16(b.langIndex(to[0]))
				likelyRegionGroup[id].script = uint8(b.script.index(to[1]))
				likelyRegionGroup[id].region = uint16(b.region.index(to[2]))
			} else {
				regionToOther[r] = append(regionToOther[r], fromTo{from, to})
			}
		}
	}
	b.writeType(likelyLangRegion{})
	b.writeSlice("likelyScript", likelyScript)

	for id := range b.lang.s {
		list := langToOther[id]
		if len(list) == 1 {
			likelyLang[id].region = uint16(b.region.index(list[0].to[2]))
			likelyLang[id].script = uint8(b.script.index(list[0].to[1]))
		} else if len(list) > 1 {
			likelyLang[id].flags = isList
			likelyLang[id].region = uint16(len(likelyLangList))
			likelyLang[id].script = uint8(len(list))
			for _, x := range list {
				flags := uint8(0)
				if len(x.from) > 1 {
					if x.from[1] == x.to[2] {
						flags = regionInFrom
					} else {
						flags = scriptInFrom
					}
				}
				likelyLangList = append(likelyLangList, likelyScriptRegion{
					region: uint16(b.region.index(x.to[2])),
					script: uint8(b.script.index(x.to[1])),
					flags:  flags,
				})
			}
		}
	}
	// TODO: merge suppressScript data with this table.
	b.writeType(likelyScriptRegion{})
	b.writeSlice("likelyLang", likelyLang)
	b.writeSlice("likelyLangList", likelyLangList)

	for id := range b.region.s {
		list := regionToOther[id]
		if len(list) == 1 {
			likelyRegion[id].lang = uint16(b.langIndex(list[0].to[0]))
			likelyRegion[id].script = uint8(b.script.index(list[0].to[1]))
			if len(list[0].from) > 2 {
				likelyRegion[id].flags = scriptInFrom
			}
		} else if len(list) > 1 {
			likelyRegion[id].flags = isList
			likelyRegion[id].lang = uint16(len(likelyRegionList))
			likelyRegion[id].script = uint8(len(list))
			for i, x := range list {
				if len(x.from) == 2 && i != 0 || i > 0 && len(x.from) != 3 {
					log.Fatalf("unspecified script must be first in list: %v at %d", x.from, i)
				}
				x := likelyLangScript{
					lang:   uint16(b.langIndex(x.to[0])),
					script: uint8(b.script.index(x.to[1])),
				}
				if len(list[0].from) > 2 {
					x.flags = scriptInFrom
				}
				likelyRegionList = append(likelyRegionList, x)
			}
		}
	}
	b.writeType(likelyLangScript{})
	b.writeSlice("likelyRegion", likelyRegion)
	b.writeSlice("likelyRegionList", likelyRegionList)

	b.writeType(likelyTag{})
	b.writeSlice("likelyRegionGroup", likelyRegionGroup)
}

type mutualIntelligibility struct {
	want, have uint16
	distance   uint8
	oneway     bool
}

type scriptIntelligibility struct {
	wantLang, haveLang     uint16
	wantScript, haveScript uint8
	distance               uint8
	// Always oneway
}

type regionIntelligibility struct {
	lang     uint16 // compact language id
	script   uint8  // 0 means any
	group    uint8  // 0 means any; if bit 7 is set it means inverse
	distance uint8
	// Always twoway.
}

// writeMatchData writes tables with languages and scripts for which there is
// mutual intelligibility. The data is based on CLDR's languageMatching data.
// Note that we use a different algorithm than the one defined by CLDR and that
// we slightly modify the data. For example, we convert scores to confidence levels.
// We also drop all region-related data as we use a different algorithm to
// determine region equivalence.
func (b *builder) writeMatchData() {
	lm := b.supp.LanguageMatching.LanguageMatches
	cldr.MakeSlice(&lm).SelectAnyOf("type", "written_new")

	regionHierarchy := map[string][]string{}
	for _, g := range b.supp.TerritoryContainment.Group {
		regions := strings.Split(g.Contains, " ")
		regionHierarchy[g.Type] = append(regionHierarchy[g.Type], regions...)
	}
	regionToGroups := make([]uint8, len(b.region.s))

	idToIndex := map[string]uint8{}
	for i, mv := range lm[0].MatchVariable {
		if i > 6 {
			log.Fatalf("Too many groups: %d", i)
		}
		idToIndex[mv.Id] = uint8(i + 1)
		// TODO: also handle '-'
		for _, r := range strings.Split(mv.Value, "+") {
			todo := []string{r}
			for k := 0; k < len(todo); k++ {
				r := todo[k]
				regionToGroups[b.region.index(r)] |= 1 << uint8(i)
				todo = append(todo, regionHierarchy[r]...)
			}
		}
	}
	b.writeSlice("regionToGroups", regionToGroups)

	b.writeType(mutualIntelligibility{})
	b.writeType(scriptIntelligibility{})
	b.writeType(regionIntelligibility{})

	matchLang := []mutualIntelligibility{{
		// TODO: remove once CLDR is fixed.
		want:     uint16(b.langIndex("sr")),
		have:     uint16(b.langIndex("hr")),
		distance: uint8(5),
	}, {
		want:     uint16(b.langIndex("sr")),
		have:     uint16(b.langIndex("bs")),
		distance: uint8(5),
	}}
	matchScript := []scriptIntelligibility{}
	matchRegion := []regionIntelligibility{}
	// Convert the languageMatch entries in lists keyed by desired language.
	for _, m := range lm[0].LanguageMatch {
		// Different versions of CLDR use different separators.
		desired := strings.Replace(m.Desired, "-", "_", -1)
		supported := strings.Replace(m.Supported, "-", "_", -1)
		d := strings.Split(desired, "_")
		s := strings.Split(supported, "_")
		if len(d) != len(s) {
			log.Fatalf("not supported: desired=%q; supported=%q", desired, supported)
			continue
		}
		distance, _ := strconv.ParseInt(m.Distance, 10, 8)
		switch len(d) {
		case 2:
			if desired == supported && desired == "*_*" {
				continue
			}
			// language-script pair.
			matchScript = append(matchScript, scriptIntelligibility{
				wantLang:   uint16(b.langIndex(d[0])),
				haveLang:   uint16(b.langIndex(s[0])),
				wantScript: uint8(b.script.index(d[1])),
				haveScript: uint8(b.script.index(s[1])),
				distance:   uint8(distance),
			})
			if m.Oneway != "true" {
				matchScript = append(matchScript, scriptIntelligibility{
					wantLang:   uint16(b.langIndex(s[0])),
					haveLang:   uint16(b.langIndex(d[0])),
					wantScript: uint8(b.script.index(s[1])),
					haveScript: uint8(b.script.index(d[1])),
					distance:   uint8(distance),
				})
			}
		case 1:
			if desired == supported && desired == "*" {
				continue
			}
			if distance == 1 {
				// nb == no is already handled by macro mapping. Check there
				// really is only this case.
				if d[0] != "no" || s[0] != "nb" {
					log.Fatalf("unhandled equivalence %s == %s", s[0], d[0])
				}
				continue
			}
			// TODO: consider dropping oneway field and just doubling the entry.
			matchLang = append(matchLang, mutualIntelligibility{
				want:     uint16(b.langIndex(d[0])),
				have:     uint16(b.langIndex(s[0])),
				distance: uint8(distance),
				oneway:   m.Oneway == "true",
			})
		case 3:
			if desired == supported && desired == "*_*_*" {
				continue
			}
			if desired != supported { // (Weird but correct.)
				log.Fatalf("not supported: desired=%q; supported=%q", desired, supported)
				continue
			}
			ri := regionIntelligibility{
				lang:     b.langIndex(d[0]),
				distance: uint8(distance),
			}
			if d[1] != "*" {
				ri.script = uint8(b.script.index(d[1]))
			}
			switch {
			case d[2] == "*":
				ri.group = 0x80 // not contained in anything
			case strings.HasPrefix(d[2], "$!"):
				ri.group = 0x80
				d[2] = "$" + d[2][len("$!"):]
				fallthrough
			case strings.HasPrefix(d[2], "$"):
				ri.group |= idToIndex[d[2]]
			}
			matchRegion = append(matchRegion, ri)
		default:
			log.Fatalf("not supported: desired=%q; supported=%q", desired, supported)
		}
	}
	sort.SliceStable(matchLang, func(i, j int) bool {
		return matchLang[i].distance < matchLang[j].distance
	})
	b.writeSlice("matchLang", matchLang)

	sort.SliceStable(matchScript, func(i, j int) bool {
		return matchScript[i].distance < matchScript[j].distance
	})
	b.writeSlice("matchScript", matchScript)

	sort.SliceStable(matchRegion, func(i, j int) bool {
		return matchRegion[i].distance < matchRegion[j].distance
	})
	b.writeSlice("matchRegion", matchRegion)
}

func (b *builder) writeRegionInclusionData() {
	var (
		// mm holds for each group the set of groups with a distance of 1.
		mm = make(map[int][]index)

		// containment holds for each group the transitive closure of
		// containment of other groups.
		containment = make(map[index][]index)
	)
	for _, g := range b.supp.TerritoryContainment.Group {
		// Skip UN and EURO zone as they are flattening the containment
		// relationship.
		if g.Type == "EZ" || g.Type == "UN" {
			continue
		}
		group := b.region.index(g.Type)
		groupIdx := b.groups[group]
		for _, mem := range strings.Split(g.Contains, " ") {
			r := b.region.index(mem)
			mm[r] = append(mm[r], groupIdx)
			if g, ok := b.groups[r]; ok {
				mm[group] = append(mm[group], g)
				containment[groupIdx] = append(containment[groupIdx], g)
			}
		}
	}

	regionContainment := make([]uint32, len(b.groups))
	for _, g := range b.groups {
		l := containment[g]

		// Compute the transitive closure of containment.
		for i := 0; i < len(l); i++ {
			l = append(l, containment[l[i]]...)
		}

		// Compute the bitmask.
		regionContainment[g] = 1 << g
		for _, v := range l {
			regionContainment[g] |= 1 << v
		}
	}
	b.writeSlice("regionContainment", regionContainment)

	regionInclusion := make([]uint8, len(b.region.s))
	bvs := make(map[uint32]index)
	// Make the first bitvector positions correspond with the groups.
	for r, i := range b.groups {
		bv := uint32(1 << i)
		for _, g := range mm[r] {
			bv |= 1 << g
		}
		bvs[bv] = i
		regionInclusion[r] = uint8(bvs[bv])
	}
	for r := 1; r < len(b.region.s); r++ {
		if _, ok := b.groups[r]; !ok {
			bv := uint32(0)
			for _, g := range mm[r] {
				bv |= 1 << g
			}
			if bv == 0 {
				// Pick the world for unspecified regions.
				bv = 1 << b.groups[b.region.index("001")]
			}
			if _, ok := bvs[bv]; !ok {
				bvs[bv] = index(len(bvs))
			}
			regionInclusion[r] = uint8(bvs[bv])
		}
	}
	b.writeSlice("regionInclusion", regionInclusion)
	regionInclusionBits := make([]uint32, len(bvs))
	for k, v := range bvs {
		regionInclusionBits[v] = uint32(k)
	}
	// Add bit vectors for increasingly large distances until a fixed point is reached.
	regionInclusionNext := []uint8{}
	for i := 0; i < len(regionInclusionBits); i++ {
		bits := regionInclusionBits[i]
		next := bits
		for i := uint(0); i < uint(len(b.groups)); i++ {
			if bits&(1<<i) != 0 {
				next |= regionInclusionBits[i]
			}
		}
		if _, ok := bvs[next]; !ok {
			bvs[next] = index(len(bvs))
			regionInclusionBits = append(regionInclusionBits, next)
		}
		regionInclusionNext = append(regionInclusionNext, uint8(bvs[next]))
	}
	b.writeSlice("regionInclusionBits", regionInclusionBits)
	b.writeSlice("regionInclusionNext", regionInclusionNext)
}

type parentRel struct {
	lang       uint16
	script     uint8
	maxScript  uint8
	toRegion   uint16
	fromRegion []uint16
}

func (b *builder) writeParents() {
	b.writeType(parentRel{})

	parents := []parentRel{}

	// Construct parent overrides.
	n := 0
	for _, p := range b.data.Supplemental().ParentLocales.ParentLocale {
		// Skipping non-standard scripts to root is implemented using addTags.
		if p.Parent == "root" {
			continue
		}

		sub := strings.Split(p.Parent, "_")
		parent := parentRel{lang: b.langIndex(sub[0])}
		if len(sub) == 2 {
			// TODO: check that all undefined scripts are indeed Latn in these
			// cases.
			parent.maxScript = uint8(b.script.index("Latn"))
			parent.toRegion = uint16(b.region.index(sub[1]))
		} else {
			parent.script = uint8(b.script.index(sub[1]))
			parent.maxScript = parent.script
			parent.toRegion = uint16(b.region.index(sub[2]))
		}
		for _, c := range strings.Split(p.Locales, " ") {
			region := b.region.index(c[strings.LastIndex(c, "_")+1:])
			parent.fromRegion = append(parent.fromRegion, uint16(region))
		}
		parents = append(parents, parent)
		n += len(parent.fromRegion)
	}
	b.writeSliceAddSize("parents", n*2, parents)
}

func main() {
	gen.Init()

	gen.Repackage("gen_common.go", "common.go", "language")

	w := gen.NewCodeWriter()
	defer w.WriteGoFile("tables.go", "language")

	fmt.Fprintln(w, `import "golang.org/x/text/internal/tag"`)

	b := newBuilder(w)
	gen.WriteCLDRVersion(w)

	b.parseIndices()
	b.writeType(fromTo{})
	b.writeLanguage()
	b.writeScript()
	b.writeRegion()
	b.writeVariant()
	// TODO: b.writeLocale()
	b.computeRegionGroups()
	b.writeLikelyData()
	b.writeMatchData()
	b.writeRegionInclusionData()
	b.writeParents()
}
