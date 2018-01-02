// Package idn implements encoding from and to punycode as speficied by RFC 3492.
package idn

import (
	"bytes"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/miekg/dns"
)

// Implementation idea from RFC itself and from from IDNA::Punycode created by
// Tatsuhiko Miyagawa <miyagawa@bulknews.net> and released under Perl Artistic
// License in 2002.

const (
	_MIN  rune = 1
	_MAX  rune = 26
	_SKEW rune = 38
	_BASE rune = 36
	_BIAS rune = 72
	_N    rune = 128
	_DAMP rune = 700

	_DELIMITER = '-'
	_PREFIX    = "xn--"
)

// ToPunycode converts unicode domain names to DNS-appropriate punycode names.
// This function will return an empty string result for domain names with
// invalid unicode strings. This function expects domain names in lowercase.
func ToPunycode(s string) string {
	// Early check to see if encoding is needed.
	// This will prevent making heap allocations when not needed.
	if !needToPunycode(s) {
		return s
	}

	tokens := dns.SplitDomainName(s)
	switch {
	case s == "":
		return ""
	case tokens == nil: // s == .
		return "."
	case s[len(s)-1] == '.':
		tokens = append(tokens, "")
	}

	for i := range tokens {
		t := encode([]byte(tokens[i]))
		if t == nil {
			return ""
		}
		tokens[i] = string(t)
	}
	return strings.Join(tokens, ".")
}

// FromPunycode returns unicode domain name from provided punycode string.
// This function expects punycode strings in lowercase.
func FromPunycode(s string) string {
	// Early check to see if decoding is needed.
	// This will prevent making heap allocations when not needed.
	if !needFromPunycode(s) {
		return s
	}

	tokens := dns.SplitDomainName(s)
	switch {
	case s == "":
		return ""
	case tokens == nil: // s == .
		return "."
	case s[len(s)-1] == '.':
		tokens = append(tokens, "")
	}
	for i := range tokens {
		tokens[i] = string(decode([]byte(tokens[i])))
	}
	return strings.Join(tokens, ".")
}

// digitval converts single byte into meaningful value that's used to calculate decoded unicode character.
const errdigit = 0xffff

func digitval(code rune) rune {
	switch {
	case code >= 'A' && code <= 'Z':
		return code - 'A'
	case code >= 'a' && code <= 'z':
		return code - 'a'
	case code >= '0' && code <= '9':
		return code - '0' + 26
	}
	return errdigit
}

// lettercode finds BASE36 byte (a-z0-9) based on calculated number.
func lettercode(digit rune) rune {
	switch {
	case digit >= 0 && digit <= 25:
		return digit + 'a'
	case digit >= 26 && digit <= 36:
		return digit - 26 + '0'
	}
	panic("dns: not reached")
}

// adapt calculates next bias to be used for next iteration delta.
func adapt(delta rune, numpoints int, firsttime bool) rune {
	if firsttime {
		delta /= _DAMP
	} else {
		delta /= 2
	}

	var k rune
	for delta = delta + delta/rune(numpoints); delta > (_BASE-_MIN)*_MAX/2; k += _BASE {
		delta /= _BASE - _MIN
	}

	return k + ((_BASE-_MIN+1)*delta)/(delta+_SKEW)
}

// next finds minimal rune (one with lowest codepoint value) that should be equal or above boundary.
func next(b []rune, boundary rune) rune {
	if len(b) == 0 {
		panic("dns: invalid set of runes to determine next one")
	}
	m := b[0]
	for _, x := range b[1:] {
		if x >= boundary && (m < boundary || x < m) {
			m = x
		}
	}
	return m
}

// preprune converts unicode rune to lower case. At this time it's not
// supporting all things described in RFCs.
func preprune(r rune) rune {
	if unicode.IsUpper(r) {
		r = unicode.ToLower(r)
	}
	return r
}

// tfunc is a function that helps calculate each character weight.
func tfunc(k, bias rune) rune {
	switch {
	case k <= bias:
		return _MIN
	case k >= bias+_MAX:
		return _MAX
	}
	return k - bias
}

// needToPunycode returns true for strings that require punycode encoding
// (contain unicode characters).
func needToPunycode(s string) bool {
	// This function is very similar to bytes.Runes. We don't use bytes.Runes
	// because it makes a heap allocation that's not needed here.
	for i := 0; len(s) > 0; i++ {
		r, l := utf8.DecodeRuneInString(s)
		if r > 0x7f {
			return true
		}
		s = s[l:]
	}
	return false
}

// needFromPunycode returns true for strings that require punycode decoding.
func needFromPunycode(s string) bool {
	if s == "." {
		return false
	}

	off := 0
	end := false
	pl := len(_PREFIX)
	sl := len(s)

	// If s starts with _PREFIX.
	if sl > pl && s[off:off+pl] == _PREFIX {
		return true
	}

	for {
		// Find the part after the next ".".
		off, end = dns.NextLabel(s, off)
		if end {
			return false
		}
		// If this parts starts with _PREFIX.
		if sl-off > pl && s[off:off+pl] == _PREFIX {
			return true
		}
	}
}

// encode transforms Unicode input bytes (that represent DNS label) into
// punycode bytestream. This function would return nil if there's an invalid
// character in the label.
func encode(input []byte) []byte {
	n, bias := _N, _BIAS

	b := bytes.Runes(input)
	for i := range b {
		if !isValidRune(b[i]) {
			return nil
		}

		b[i] = preprune(b[i])
	}

	basic := make([]byte, 0, len(b))
	for _, ltr := range b {
		if ltr <= 0x7f {
			basic = append(basic, byte(ltr))
		}
	}
	basiclen := len(basic)
	fulllen := len(b)
	if basiclen == fulllen {
		return basic
	}

	var out bytes.Buffer

	out.WriteString(_PREFIX)
	if basiclen > 0 {
		out.Write(basic)
		out.WriteByte(_DELIMITER)
	}

	var (
		ltr, nextltr rune
		delta, q     rune // delta calculation (see rfc)
		t, k, cp     rune // weight and codepoint calculation
	)

	s := &bytes.Buffer{}
	for h := basiclen; h < fulllen; n, delta = n+1, delta+1 {
		nextltr = next(b, n)
		s.Truncate(0)
		s.WriteRune(nextltr)
		delta, n = delta+(nextltr-n)*rune(h+1), nextltr

		for _, ltr = range b {
			if ltr < n {
				delta++
			}
			if ltr == n {
				q = delta
				for k = _BASE; ; k += _BASE {
					t = tfunc(k, bias)
					if q < t {
						break
					}
					cp = t + ((q - t) % (_BASE - t))
					out.WriteRune(lettercode(cp))
					q = (q - t) / (_BASE - t)
				}

				out.WriteRune(lettercode(q))

				bias = adapt(delta, h+1, h == basiclen)
				h, delta = h+1, 0
			}
		}
	}
	return out.Bytes()
}

// decode transforms punycode input bytes (that represent DNS label) into Unicode bytestream.
func decode(b []byte) []byte {
	src := b // b would move and we need to keep it

	n, bias := _N, _BIAS
	if !bytes.HasPrefix(b, []byte(_PREFIX)) {
		return b
	}
	out := make([]rune, 0, len(b))
	b = b[len(_PREFIX):]
	for pos := len(b) - 1; pos >= 0; pos-- {
		// only last delimiter is our interest
		if b[pos] == _DELIMITER {
			out = append(out, bytes.Runes(b[:pos])...)
			b = b[pos+1:] // trim source string
			break
		}
	}
	if len(b) == 0 {
		return src
	}
	var (
		i, oldi, w rune
		ch         byte
		t, digit   rune
		ln         int
	)

	for i = 0; len(b) > 0; i++ {
		oldi, w = i, 1
		for k := _BASE; len(b) > 0; k += _BASE {
			ch, b = b[0], b[1:]
			digit = digitval(rune(ch))
			if digit == errdigit {
				return src
			}
			i += digit * w
			if i < 0 {
				// safety check for rune overflow
				return src
			}

			t = tfunc(k, bias)
			if digit < t {
				break
			}

			w *= _BASE - t
		}
		ln = len(out) + 1
		bias = adapt(i-oldi, ln, oldi == 0)
		n += i / rune(ln)
		i = i % rune(ln)
		// insert
		out = append(out, 0)
		copy(out[i+1:], out[i:])
		out[i] = n
	}

	var ret bytes.Buffer
	for _, r := range out {
		ret.WriteRune(r)
	}
	return ret.Bytes()
}

// isValidRune checks if the character is valid. We will look for the
// character property in the code points list. For now we aren't checking special
// rules in case of contextual property
func isValidRune(r rune) bool {
	return findProperty(r) == propertyPVALID
}

// findProperty will try to check the code point property of the given
// character. It will use a binary search algorithm as we have a slice of
// ordered ranges (average case performance O(log n))
func findProperty(r rune) property {
	imin, imax := 0, len(codePoints)

	for imax >= imin {
		imid := (imin + imax) / 2

		codePoint := codePoints[imid]
		if (codePoint.start == r && codePoint.end == 0) || (codePoint.start <= r && codePoint.end >= r) {
			return codePoint.state
		}

		if (codePoint.end > 0 && codePoint.end < r) || (codePoint.end == 0 && codePoint.start < r) {
			imin = imid + 1
		} else {
			imax = imid - 1
		}
	}

	return propertyUnknown
}
