package goabnf

import (
	"math"
	"strconv"
	"strings"
)

// int32ToNumval converts an integer value into its numeric-string representation
// in the given base. It is the inverse of numvalToInt32 and produces the form
// stored in an ElemNumVal's Elems (e.g. 97 in base "x" -> "61").
func int32ToNumval(v int32, base string) string {
	switch base {
	case "B", "b":
		return strconv.FormatInt(int64(v), 2)
	case "D", "d":
		return strconv.FormatInt(int64(v), 10)
	case "X", "x":
		return strings.ToUpper(strconv.FormatInt(int64(v), 16))
	}
	return strconv.FormatInt(int64(v), 10)
}

// baseRadix maps an ABNF num-val base marker (bin/dec/hex) to its integer radix.
func baseRadix(base string) int {
	switch base {
	case "B", "b":
		return 2
	case "D", "d":
		return 10
	case "X", "x":
		return 16
	}
	return 10
}

// numvalToUint64 parses a num-val literal in the given base into its numeric
// value. ok is false only when the literal does not fit in 64 bits (or is
// malformed), which lets callers degrade gracefully instead of overflowing
// silently. This is the single overflow-safe entry point every other conversion
// builds on.
//
// A num-val is a plain non-negative integer, not intrinsically a Unicode scalar:
// values beyond U+10FFFF are legitimate (e.g. for non-textual / on-wire
// grammars). Unicode is handled as the common case at match time; this layer is
// no longer Unicode-bounded.
func numvalToUint64(str, base string) (uint64, bool) {
	v, err := strconv.ParseUint(str, baseRadix(base), 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// numvalToInt32 converts a num-val into an int32 codepoint. It never panics: a
// value that does not fit a positive int32 is saturated to math.MaxInt32, which
// is above the maximum Unicode scalar (U+10FFFF) and therefore matches no decoded
// rune. A range bound that merely straddles the Unicode ceiling still covers its
// representable (Unicode) subset, because the saturated upper bound stays above
// every rune.
func numvalToInt32(str, base string) int32 {
	v, ok := numvalToUint64(str, base)
	if !ok || v > math.MaxInt32 {
		return math.MaxInt32
	}
	return int32(v)
}

// numvalToRune converts a num-val into the corresponding rune. It never panics;
// a value beyond the Unicode range yields a rune that no UTF-8 input can produce,
// so callers matching against decoded runes simply find no match. Callers that
// require text validity (e.g. a num-val series, which must denote real
// characters) should additionally check utf8.ValidRune.
func numvalToRune(str, base string) rune {
	return rune(numvalToInt32(str, base))
}

// checkBounds enforces the Unicode-range invariant (value <= U+10FFFF). It is
// the strict-validation path used by SemvalABNF: with validation enabled (the
// default, and the common textual case) a num-val outside the Unicode range is
// rejected. The conversion helpers above are deliberately broader -- a num-val is
// a plain integer and larger values are legitimate for non-textual grammars -- so
// without validation such values are accepted and handled (matched as no rune).
func checkBounds(str, base string) error {
	str = strings.TrimLeft(str, "0")
	switch base {
	// Whatever the base, the higher value we arbitrary decide to support is
	// the maximal Unicode character i.e. U+10FFFF.
	//
	// Every value (represented as integer) is then in interval [0;1_114_111].
	// For each base, we can compute the number of characters of an acceptable
	// num-val as ceil(log(base,1_114_111)).
	//
	// We also have to ensure that it does not go over the boundaries even if
	// of an acceptable length, e.g., 0xFFFFFF is not.

	case "B", "b":
		// 21 = ceil(log(2, max_unicode))
		// U+10FFFF = Ob100001111111111111111
		if len(str) > 21 ||
			(len(str) == 21 && (str[1] != '0' || str[2] != '0' || str[3] != '0' || str[4] != '0')) {
			return &ErrTooLargeNumeral{
				Base:  base,
				Value: str,
			}
		}

	case "D", "d":
		// 7 = ceil(log(10, max_unicode))
		// U+10FFFF = 0d1114111
		if len(str) > 7 ||
			(len(str) == 7 && (str[0] > '1' || (str[0] == '1' && (str[1] > '1' || (str[1] == '1' && (str[2] > '1' || (str[2] == '1' && (str[3] > '4' || (str[3] == '4' && (str[4] > '1' || (str[4] == '1' && (str[5] > '1' || (str[5] == '1' && str[6] > '1'))))))))))))) {
			return &ErrTooLargeNumeral{
				Base:  base,
				Value: str,
			}
		}

	case "X", "x":
		// 6 = ceil(log(16, max_unicode))
		// U+10FFFF = 0x10FFFF
		if len(str) > 6 ||
			(len(str) == 6 && (str[0] > '1' || (str[0] == '1' && (str[1] > '0')))) {
			return &ErrTooLargeNumeral{
				Base:  base,
				Value: str,
			}
		}
	}

	return nil
}

// GetRule returns the rule by the given rulename, whether
// it is a core rule or present in the grammar, or nil if not found.
// It validates the RFC 5234 Section 2.1 "rule names are case insensitive".
func GetRule(rulename string, rulemap map[string]*Rule) *Rule {
	// First look in the newly defined rules, as we could override definitions
	if rule := getRuleIn(rulename, rulemap); rule != nil {
		return rule
	}
	return getRuleIn(rulename, coreRules)
}

func getRuleIn(rulename string, rulemap map[string]*Rule) *Rule {
	for _, rule := range rulemap {
		if strings.EqualFold(rulename, rule.Name) {
			return rule
		}
	}
	return nil
}
