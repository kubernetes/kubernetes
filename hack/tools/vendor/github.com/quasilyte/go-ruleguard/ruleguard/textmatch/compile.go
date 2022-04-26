package textmatch

import (
	"regexp"
	"regexp/syntax"
	"unicode"
)

func compile(s string) (Pattern, error) {
	reSyntax, err := syntax.Parse(s, syntax.Perl)
	if err == nil {
		if optimized := compileOptimized(s, reSyntax); optimized != nil {
			return optimized, nil
		}
	}
	return regexp.Compile(s)
}

func compileOptimized(s string, re *syntax.Regexp) Pattern {
	// .*
	isAny := func(re *syntax.Regexp) bool {
		return re.Op == syntax.OpStar && re.Sub[0].Op == syntax.OpAnyCharNotNL
	}
	// "literal"
	isLit := func(re *syntax.Regexp) bool {
		return re.Op == syntax.OpLiteral
	}
	// ^
	isBegin := func(re *syntax.Regexp) bool {
		return re.Op == syntax.OpBeginText
	}
	// $
	isEnd := func(re *syntax.Regexp) bool {
		return re.Op == syntax.OpEndText
	}

	// TODO: analyze what kind of regexps people use in rules
	// more often and optimize those as well.

	// lit => strings.Contains($input, lit)
	if re.Op == syntax.OpLiteral {
		return &containsLiteralMatcher{value: newInputValue(string(re.Rune))}
	}

	// `.*` lit `.*` => strings.Contains($input, lit)
	if re.Op == syntax.OpConcat && len(re.Sub) == 3 {
		if isAny(re.Sub[0]) && isLit(re.Sub[1]) && isAny(re.Sub[2]) {
			return &containsLiteralMatcher{value: newInputValue(string(re.Sub[1].Rune))}
		}
	}

	// `^` lit => strings.HasPrefix($input, lit)
	if re.Op == syntax.OpConcat && len(re.Sub) == 2 {
		if isBegin(re.Sub[0]) && isLit(re.Sub[1]) {
			return &prefixLiteralMatcher{value: newInputValue(string(re.Sub[1].Rune))}
		}
	}

	// lit `$` => strings.HasSuffix($input, lit)
	if re.Op == syntax.OpConcat && len(re.Sub) == 2 {
		if isLit(re.Sub[0]) && isEnd(re.Sub[1]) {
			return &suffixLiteralMatcher{value: newInputValue(string(re.Sub[0].Rune))}
		}
	}

	// `^` lit `$` => $input == lit
	if re.Op == syntax.OpConcat && len(re.Sub) == 3 {
		if isBegin(re.Sub[0]) && isLit(re.Sub[1]) && isEnd(re.Sub[2]) {
			return &eqLiteralMatcher{value: newInputValue(string(re.Sub[1].Rune))}
		}
	}

	// `^\p{Lu}` => prefixRunePredMatcher:unicode.IsUpper
	// `^\p{Ll}` => prefixRunePredMatcher:unicode.IsLower
	switch s {
	case `^\p{Lu}`:
		return &prefixRunePredMatcher{pred: unicode.IsUpper}
	case `^\p{Ll}`:
		return &prefixRunePredMatcher{pred: unicode.IsLower}
	}

	// Can't optimize.
	return nil
}
