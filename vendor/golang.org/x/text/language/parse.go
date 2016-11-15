// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/text/internal/tag"
)

// isAlpha returns true if the byte is not a digit.
// b must be an ASCII letter or digit.
func isAlpha(b byte) bool {
	return b > '9'
}

// isAlphaNum returns true if the string contains only ASCII letters or digits.
func isAlphaNum(s []byte) bool {
	for _, c := range s {
		if !('a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9') {
			return false
		}
	}
	return true
}

// errSyntax is returned by any of the parsing functions when the
// input is not well-formed, according to BCP 47.
// TODO: return the position at which the syntax error occurred?
var errSyntax = errors.New("language: tag is not well-formed")

// ValueError is returned by any of the parsing functions when the
// input is well-formed but the respective subtag is not recognized
// as a valid value.
type ValueError struct {
	v [8]byte
}

func mkErrInvalid(s []byte) error {
	var e ValueError
	copy(e.v[:], s)
	return e
}

func (e ValueError) tag() []byte {
	n := bytes.IndexByte(e.v[:], 0)
	if n == -1 {
		n = 8
	}
	return e.v[:n]
}

// Error implements the error interface.
func (e ValueError) Error() string {
	return fmt.Sprintf("language: subtag %q is well-formed but unknown", e.tag())
}

// Subtag returns the subtag for which the error occurred.
func (e ValueError) Subtag() string {
	return string(e.tag())
}

// scanner is used to scan BCP 47 tokens, which are separated by _ or -.
type scanner struct {
	b     []byte
	bytes [max99thPercentileSize]byte
	token []byte
	start int // start position of the current token
	end   int // end position of the current token
	next  int // next point for scan
	err   error
	done  bool
}

func makeScannerString(s string) scanner {
	scan := scanner{}
	if len(s) <= len(scan.bytes) {
		scan.b = scan.bytes[:copy(scan.bytes[:], s)]
	} else {
		scan.b = []byte(s)
	}
	scan.init()
	return scan
}

// makeScanner returns a scanner using b as the input buffer.
// b is not copied and may be modified by the scanner routines.
func makeScanner(b []byte) scanner {
	scan := scanner{b: b}
	scan.init()
	return scan
}

func (s *scanner) init() {
	for i, c := range s.b {
		if c == '_' {
			s.b[i] = '-'
		}
	}
	s.scan()
}

// restToLower converts the string between start and end to lower case.
func (s *scanner) toLower(start, end int) {
	for i := start; i < end; i++ {
		c := s.b[i]
		if 'A' <= c && c <= 'Z' {
			s.b[i] += 'a' - 'A'
		}
	}
}

func (s *scanner) setError(e error) {
	if s.err == nil || (e == errSyntax && s.err != errSyntax) {
		s.err = e
	}
}

// resizeRange shrinks or grows the array at position oldStart such that
// a new string of size newSize can fit between oldStart and oldEnd.
// Sets the scan point to after the resized range.
func (s *scanner) resizeRange(oldStart, oldEnd, newSize int) {
	s.start = oldStart
	if end := oldStart + newSize; end != oldEnd {
		diff := end - oldEnd
		if end < cap(s.b) {
			b := make([]byte, len(s.b)+diff)
			copy(b, s.b[:oldStart])
			copy(b[end:], s.b[oldEnd:])
			s.b = b
		} else {
			s.b = append(s.b[end:], s.b[oldEnd:]...)
		}
		s.next = end + (s.next - s.end)
		s.end = end
	}
}

// replace replaces the current token with repl.
func (s *scanner) replace(repl string) {
	s.resizeRange(s.start, s.end, len(repl))
	copy(s.b[s.start:], repl)
}

// gobble removes the current token from the input.
// Caller must call scan after calling gobble.
func (s *scanner) gobble(e error) {
	s.setError(e)
	if s.start == 0 {
		s.b = s.b[:+copy(s.b, s.b[s.next:])]
		s.end = 0
	} else {
		s.b = s.b[:s.start-1+copy(s.b[s.start-1:], s.b[s.end:])]
		s.end = s.start - 1
	}
	s.next = s.start
}

// deleteRange removes the given range from s.b before the current token.
func (s *scanner) deleteRange(start, end int) {
	s.setError(errSyntax)
	s.b = s.b[:start+copy(s.b[start:], s.b[end:])]
	diff := end - start
	s.next -= diff
	s.start -= diff
	s.end -= diff
}

// scan parses the next token of a BCP 47 string.  Tokens that are larger
// than 8 characters or include non-alphanumeric characters result in an error
// and are gobbled and removed from the output.
// It returns the end position of the last token consumed.
func (s *scanner) scan() (end int) {
	end = s.end
	s.token = nil
	for s.start = s.next; s.next < len(s.b); {
		i := bytes.IndexByte(s.b[s.next:], '-')
		if i == -1 {
			s.end = len(s.b)
			s.next = len(s.b)
			i = s.end - s.start
		} else {
			s.end = s.next + i
			s.next = s.end + 1
		}
		token := s.b[s.start:s.end]
		if i < 1 || i > 8 || !isAlphaNum(token) {
			s.gobble(errSyntax)
			continue
		}
		s.token = token
		return end
	}
	if n := len(s.b); n > 0 && s.b[n-1] == '-' {
		s.setError(errSyntax)
		s.b = s.b[:len(s.b)-1]
	}
	s.done = true
	return end
}

// acceptMinSize parses multiple tokens of the given size or greater.
// It returns the end position of the last token consumed.
func (s *scanner) acceptMinSize(min int) (end int) {
	end = s.end
	s.scan()
	for ; len(s.token) >= min; s.scan() {
		end = s.end
	}
	return end
}

// Parse parses the given BCP 47 string and returns a valid Tag. If parsing
// failed it returns an error and any part of the tag that could be parsed.
// If parsing succeeded but an unknown value was found, it returns
// ValueError. The Tag returned in this case is just stripped of the unknown
// value. All other values are preserved. It accepts tags in the BCP 47 format
// and extensions to this standard defined in
// http://www.unicode.org/reports/tr35/#Unicode_Language_and_Locale_Identifiers.
// The resulting tag is canonicalized using the default canonicalization type.
func Parse(s string) (t Tag, err error) {
	return Default.Parse(s)
}

// Parse parses the given BCP 47 string and returns a valid Tag. If parsing
// failed it returns an error and any part of the tag that could be parsed.
// If parsing succeeded but an unknown value was found, it returns
// ValueError. The Tag returned in this case is just stripped of the unknown
// value. All other values are preserved. It accepts tags in the BCP 47 format
// and extensions to this standard defined in
// http://www.unicode.org/reports/tr35/#Unicode_Language_and_Locale_Identifiers.
// The resulting tag is canonicalized using the the canonicalization type c.
func (c CanonType) Parse(s string) (t Tag, err error) {
	// TODO: consider supporting old-style locale key-value pairs.
	if s == "" {
		return und, errSyntax
	}
	if len(s) <= maxAltTaglen {
		b := [maxAltTaglen]byte{}
		for i, c := range s {
			// Generating invalid UTF-8 is okay as it won't match.
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			} else if c == '_' {
				c = '-'
			}
			b[i] = byte(c)
		}
		if t, ok := grandfathered(b); ok {
			return t, nil
		}
	}
	scan := makeScannerString(s)
	t, err = parse(&scan, s)
	t, changed := t.canonicalize(c)
	if changed {
		t.remakeString()
	}
	return t, err
}

func parse(scan *scanner, s string) (t Tag, err error) {
	t = und
	var end int
	if n := len(scan.token); n <= 1 {
		scan.toLower(0, len(scan.b))
		if n == 0 || scan.token[0] != 'x' {
			return t, errSyntax
		}
		end = parseExtensions(scan)
	} else if n >= 4 {
		return und, errSyntax
	} else { // the usual case
		t, end = parseTag(scan)
		if n := len(scan.token); n == 1 {
			t.pExt = uint16(end)
			end = parseExtensions(scan)
		} else if end < len(scan.b) {
			scan.setError(errSyntax)
			scan.b = scan.b[:end]
		}
	}
	if int(t.pVariant) < len(scan.b) {
		if end < len(s) {
			s = s[:end]
		}
		if len(s) > 0 && tag.Compare(s, scan.b) == 0 {
			t.str = s
		} else {
			t.str = string(scan.b)
		}
	} else {
		t.pVariant, t.pExt = 0, 0
	}
	return t, scan.err
}

// parseTag parses language, script, region and variants.
// It returns a Tag and the end position in the input that was parsed.
func parseTag(scan *scanner) (t Tag, end int) {
	var e error
	// TODO: set an error if an unknown lang, script or region is encountered.
	t.lang, e = getLangID(scan.token)
	scan.setError(e)
	scan.replace(t.lang.String())
	langStart := scan.start
	end = scan.scan()
	for len(scan.token) == 3 && isAlpha(scan.token[0]) {
		// From http://tools.ietf.org/html/bcp47, <lang>-<extlang> tags are equivalent
		// to a tag of the form <extlang>.
		lang, e := getLangID(scan.token)
		if lang != 0 {
			t.lang = lang
			copy(scan.b[langStart:], lang.String())
			scan.b[langStart+3] = '-'
			scan.start = langStart + 4
		}
		scan.gobble(e)
		end = scan.scan()
	}
	if len(scan.token) == 4 && isAlpha(scan.token[0]) {
		t.script, e = getScriptID(script, scan.token)
		if t.script == 0 {
			scan.gobble(e)
		}
		end = scan.scan()
	}
	if n := len(scan.token); n >= 2 && n <= 3 {
		t.region, e = getRegionID(scan.token)
		if t.region == 0 {
			scan.gobble(e)
		} else {
			scan.replace(t.region.String())
		}
		end = scan.scan()
	}
	scan.toLower(scan.start, len(scan.b))
	t.pVariant = byte(end)
	end = parseVariants(scan, end, t)
	t.pExt = uint16(end)
	return t, end
}

var separator = []byte{'-'}

// parseVariants scans tokens as long as each token is a valid variant string.
// Duplicate variants are removed.
func parseVariants(scan *scanner, end int, t Tag) int {
	start := scan.start
	varIDBuf := [4]uint8{}
	variantBuf := [4][]byte{}
	varID := varIDBuf[:0]
	variant := variantBuf[:0]
	last := -1
	needSort := false
	for ; len(scan.token) >= 4; scan.scan() {
		// TODO: measure the impact of needing this conversion and redesign
		// the data structure if there is an issue.
		v, ok := variantIndex[string(scan.token)]
		if !ok {
			// unknown variant
			// TODO: allow user-defined variants?
			scan.gobble(mkErrInvalid(scan.token))
			continue
		}
		varID = append(varID, v)
		variant = append(variant, scan.token)
		if !needSort {
			if last < int(v) {
				last = int(v)
			} else {
				needSort = true
				// There is no legal combinations of more than 7 variants
				// (and this is by no means a useful sequence).
				const maxVariants = 8
				if len(varID) > maxVariants {
					break
				}
			}
		}
		end = scan.end
	}
	if needSort {
		sort.Sort(variantsSort{varID, variant})
		k, l := 0, -1
		for i, v := range varID {
			w := int(v)
			if l == w {
				// Remove duplicates.
				continue
			}
			varID[k] = varID[i]
			variant[k] = variant[i]
			k++
			l = w
		}
		if str := bytes.Join(variant[:k], separator); len(str) == 0 {
			end = start - 1
		} else {
			scan.resizeRange(start, end, len(str))
			copy(scan.b[scan.start:], str)
			end = scan.end
		}
	}
	return end
}

type variantsSort struct {
	i []uint8
	v [][]byte
}

func (s variantsSort) Len() int {
	return len(s.i)
}

func (s variantsSort) Swap(i, j int) {
	s.i[i], s.i[j] = s.i[j], s.i[i]
	s.v[i], s.v[j] = s.v[j], s.v[i]
}

func (s variantsSort) Less(i, j int) bool {
	return s.i[i] < s.i[j]
}

type bytesSort [][]byte

func (b bytesSort) Len() int {
	return len(b)
}

func (b bytesSort) Swap(i, j int) {
	b[i], b[j] = b[j], b[i]
}

func (b bytesSort) Less(i, j int) bool {
	return bytes.Compare(b[i], b[j]) == -1
}

// parseExtensions parses and normalizes the extensions in the buffer.
// It returns the last position of scan.b that is part of any extension.
// It also trims scan.b to remove excess parts accordingly.
func parseExtensions(scan *scanner) int {
	start := scan.start
	exts := [][]byte{}
	private := []byte{}
	end := scan.end
	for len(scan.token) == 1 {
		extStart := scan.start
		ext := scan.token[0]
		end = parseExtension(scan)
		extension := scan.b[extStart:end]
		if len(extension) < 3 || (ext != 'x' && len(extension) < 4) {
			scan.setError(errSyntax)
			end = extStart
			continue
		} else if start == extStart && (ext == 'x' || scan.start == len(scan.b)) {
			scan.b = scan.b[:end]
			return end
		} else if ext == 'x' {
			private = extension
			break
		}
		exts = append(exts, extension)
	}
	sort.Sort(bytesSort(exts))
	if len(private) > 0 {
		exts = append(exts, private)
	}
	scan.b = scan.b[:start]
	if len(exts) > 0 {
		scan.b = append(scan.b, bytes.Join(exts, separator)...)
	} else if start > 0 {
		// Strip trailing '-'.
		scan.b = scan.b[:start-1]
	}
	return end
}

// parseExtension parses a single extension and returns the position of
// the extension end.
func parseExtension(scan *scanner) int {
	start, end := scan.start, scan.end
	switch scan.token[0] {
	case 'u':
		attrStart := end
		scan.scan()
		for last := []byte{}; len(scan.token) > 2; scan.scan() {
			if bytes.Compare(scan.token, last) != -1 {
				// Attributes are unsorted. Start over from scratch.
				p := attrStart + 1
				scan.next = p
				attrs := [][]byte{}
				for scan.scan(); len(scan.token) > 2; scan.scan() {
					attrs = append(attrs, scan.token)
					end = scan.end
				}
				sort.Sort(bytesSort(attrs))
				copy(scan.b[p:], bytes.Join(attrs, separator))
				break
			}
			last = scan.token
			end = scan.end
		}
		var last, key []byte
		for attrEnd := end; len(scan.token) == 2; last = key {
			key = scan.token
			keyEnd := scan.end
			end = scan.acceptMinSize(3)
			// TODO: check key value validity
			if keyEnd == end || bytes.Compare(key, last) != 1 {
				// We have an invalid key or the keys are not sorted.
				// Start scanning keys from scratch and reorder.
				p := attrEnd + 1
				scan.next = p
				keys := [][]byte{}
				for scan.scan(); len(scan.token) == 2; {
					keyStart, keyEnd := scan.start, scan.end
					end = scan.acceptMinSize(3)
					if keyEnd != end {
						keys = append(keys, scan.b[keyStart:end])
					} else {
						scan.setError(errSyntax)
						end = keyStart
					}
				}
				sort.Sort(bytesSort(keys))
				reordered := bytes.Join(keys, separator)
				if e := p + len(reordered); e < end {
					scan.deleteRange(e, end)
					end = e
				}
				copy(scan.b[p:], bytes.Join(keys, separator))
				break
			}
		}
	case 't':
		scan.scan()
		if n := len(scan.token); n >= 2 && n <= 3 && isAlpha(scan.token[1]) {
			_, end = parseTag(scan)
			scan.toLower(start, end)
		}
		for len(scan.token) == 2 && !isAlpha(scan.token[1]) {
			end = scan.acceptMinSize(3)
		}
	case 'x':
		end = scan.acceptMinSize(1)
	default:
		end = scan.acceptMinSize(2)
	}
	return end
}

// Compose creates a Tag from individual parts, which may be of type Tag, Base,
// Script, Region, Variant, []Variant, Extension, []Extension or error. If a
// Base, Script or Region or slice of type Variant or Extension is passed more
// than once, the latter will overwrite the former. Variants and Extensions are
// accumulated, but if two extensions of the same type are passed, the latter
// will replace the former. A Tag overwrites all former values and typically
// only makes sense as the first argument. The resulting tag is returned after
// canonicalizing using the Default CanonType. If one or more errors are
// encountered, one of the errors is returned.
func Compose(part ...interface{}) (t Tag, err error) {
	return Default.Compose(part...)
}

// Compose creates a Tag from individual parts, which may be of type Tag, Base,
// Script, Region, Variant, []Variant, Extension, []Extension or error. If a
// Base, Script or Region or slice of type Variant or Extension is passed more
// than once, the latter will overwrite the former. Variants and Extensions are
// accumulated, but if two extensions of the same type are passed, the latter
// will replace the former. A Tag overwrites all former values and typically
// only makes sense as the first argument. The resulting tag is returned after
// canonicalizing using CanonType c. If one or more errors are encountered,
// one of the errors is returned.
func (c CanonType) Compose(part ...interface{}) (t Tag, err error) {
	var b builder
	if err = b.update(part...); err != nil {
		return und, err
	}
	t, _ = b.tag.canonicalize(c)

	if len(b.ext) > 0 || len(b.variant) > 0 {
		sort.Sort(sortVariant(b.variant))
		sort.Strings(b.ext)
		if b.private != "" {
			b.ext = append(b.ext, b.private)
		}
		n := maxCoreSize + tokenLen(b.variant...) + tokenLen(b.ext...)
		buf := make([]byte, n)
		p := t.genCoreBytes(buf)
		t.pVariant = byte(p)
		p += appendTokens(buf[p:], b.variant...)
		t.pExt = uint16(p)
		p += appendTokens(buf[p:], b.ext...)
		t.str = string(buf[:p])
	} else if b.private != "" {
		t.str = b.private
		t.remakeString()
	}
	return
}

type builder struct {
	tag Tag

	private string // the x extension
	ext     []string
	variant []string

	err error
}

func (b *builder) addExt(e string) {
	if e == "" {
	} else if e[0] == 'x' {
		b.private = e
	} else {
		b.ext = append(b.ext, e)
	}
}

var errInvalidArgument = errors.New("invalid Extension or Variant")

func (b *builder) update(part ...interface{}) (err error) {
	replace := func(l *[]string, s string, eq func(a, b string) bool) bool {
		if s == "" {
			b.err = errInvalidArgument
			return true
		}
		for i, v := range *l {
			if eq(v, s) {
				(*l)[i] = s
				return true
			}
		}
		return false
	}
	for _, x := range part {
		switch v := x.(type) {
		case Tag:
			b.tag.lang = v.lang
			b.tag.region = v.region
			b.tag.script = v.script
			if v.str != "" {
				b.variant = nil
				for x, s := "", v.str[v.pVariant:v.pExt]; s != ""; {
					x, s = nextToken(s)
					b.variant = append(b.variant, x)
				}
				b.ext, b.private = nil, ""
				for i, e := int(v.pExt), ""; i < len(v.str); {
					i, e = getExtension(v.str, i)
					b.addExt(e)
				}
			}
		case Base:
			b.tag.lang = v.langID
		case Script:
			b.tag.script = v.scriptID
		case Region:
			b.tag.region = v.regionID
		case Variant:
			if !replace(&b.variant, v.variant, func(a, b string) bool { return a == b }) {
				b.variant = append(b.variant, v.variant)
			}
		case Extension:
			if !replace(&b.ext, v.s, func(a, b string) bool { return a[0] == b[0] }) {
				b.addExt(v.s)
			}
		case []Variant:
			b.variant = nil
			for _, x := range v {
				b.update(x)
			}
		case []Extension:
			b.ext, b.private = nil, ""
			for _, e := range v {
				b.update(e)
			}
		// TODO: support parsing of raw strings based on morphology or just extensions?
		case error:
			err = v
		}
	}
	return
}

func tokenLen(token ...string) (n int) {
	for _, t := range token {
		n += len(t) + 1
	}
	return
}

func appendTokens(b []byte, token ...string) int {
	p := 0
	for _, t := range token {
		b[p] = '-'
		copy(b[p+1:], t)
		p += 1 + len(t)
	}
	return p
}

type sortVariant []string

func (s sortVariant) Len() int {
	return len(s)
}

func (s sortVariant) Swap(i, j int) {
	s[j], s[i] = s[i], s[j]
}

func (s sortVariant) Less(i, j int) bool {
	return variantIndex[s[i]] < variantIndex[s[j]]
}

func findExt(list []string, x byte) int {
	for i, e := range list {
		if e[0] == x {
			return i
		}
	}
	return -1
}

// getExtension returns the name, body and end position of the extension.
func getExtension(s string, p int) (end int, ext string) {
	if s[p] == '-' {
		p++
	}
	if s[p] == 'x' {
		return len(s), s[p:]
	}
	end = nextExtension(s, p)
	return end, s[p:end]
}

// nextExtension finds the next extension within the string, searching
// for the -<char>- pattern from position p.
// In the fast majority of cases, language tags will have at most
// one extension and extensions tend to be small.
func nextExtension(s string, p int) int {
	for n := len(s) - 3; p < n; {
		if s[p] == '-' {
			if s[p+2] == '-' {
				return p
			}
			p += 3
		} else {
			p++
		}
	}
	return len(s)
}

var errInvalidWeight = errors.New("ParseAcceptLanguage: invalid weight")

// ParseAcceptLanguage parses the contents of a Accept-Language header as
// defined in http://www.ietf.org/rfc/rfc2616.txt and returns a list of Tags and
// a list of corresponding quality weights. It is more permissive than RFC 2616
// and may return non-nil slices even if the input is not valid.
// The Tags will be sorted by highest weight first and then by first occurrence.
// Tags with a weight of zero will be dropped. An error will be returned if the
// input could not be parsed.
func ParseAcceptLanguage(s string) (tag []Tag, q []float32, err error) {
	var entry string
	for s != "" {
		if entry, s = split(s, ','); entry == "" {
			continue
		}

		entry, weight := split(entry, ';')

		// Scan the language.
		t, err := Parse(entry)
		if err != nil {
			id, ok := acceptFallback[entry]
			if !ok {
				return nil, nil, err
			}
			t = Tag{lang: id}
		}

		// Scan the optional weight.
		w := 1.0
		if weight != "" {
			weight = consume(weight, 'q')
			weight = consume(weight, '=')
			// consume returns the empty string when a token could not be
			// consumed, resulting in an error for ParseFloat.
			if w, err = strconv.ParseFloat(weight, 32); err != nil {
				return nil, nil, errInvalidWeight
			}
			// Drop tags with a quality weight of 0.
			if w <= 0 {
				continue
			}
		}

		tag = append(tag, t)
		q = append(q, float32(w))
	}
	sortStable(&tagSort{tag, q})
	return tag, q, nil
}

// consume removes a leading token c from s and returns the result or the empty
// string if there is no such token.
func consume(s string, c byte) string {
	if s == "" || s[0] != c {
		return ""
	}
	return strings.TrimSpace(s[1:])
}

func split(s string, c byte) (head, tail string) {
	if i := strings.IndexByte(s, c); i >= 0 {
		return strings.TrimSpace(s[:i]), strings.TrimSpace(s[i+1:])
	}
	return strings.TrimSpace(s), ""
}

// Add hack mapping to deal with a small number of cases that that occur
// in Accept-Language (with reasonable frequency).
var acceptFallback = map[string]langID{
	"english": _en,
	"deutsch": _de,
	"italian": _it,
	"french":  _fr,
	"*":       _mul, // defined in the spec to match all languages.
}

type tagSort struct {
	tag []Tag
	q   []float32
}

func (s *tagSort) Len() int {
	return len(s.q)
}

func (s *tagSort) Less(i, j int) bool {
	return s.q[i] > s.q[j]
}

func (s *tagSort) Swap(i, j int) {
	s.tag[i], s.tag[j] = s.tag[j], s.tag[i]
	s.q[i], s.q[j] = s.q[j], s.q[i]
}
