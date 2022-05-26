package strcase

import "unicode"

// SplitFn defines how to split a string into words
type SplitFn func(prev, curr, next rune) SplitAction

// NewSplitFn returns a SplitFn based on the options provided.
//
// NewSplitFn covers the majority of common options that other strcase
// libraries provide and should allow you to simply create a custom caser.
// For more complicated use cases, feel free to write your own SplitFn
//nolint:gocyclo
func NewSplitFn(
	delimiters []rune,
	splitOptions ...SplitOption,
) SplitFn {
	var splitCase, splitAcronym, splitBeforeNumber, splitAfterNumber, preserveNumberFormatting bool

	for _, option := range splitOptions {
		switch option {
		case SplitCase:
			splitCase = true
		case SplitAcronym:
			splitAcronym = true
		case SplitBeforeNumber:
			splitBeforeNumber = true
		case SplitAfterNumber:
			splitAfterNumber = true
		case PreserveNumberFormatting:
			preserveNumberFormatting = true
		}
	}

	return func(prev, curr, next rune) SplitAction {
		// The most common case will be that it's just a letter
		// There are safe cases to process
		if isLower(curr) && !isNumber(prev) {
			return Noop
		}
		if isUpper(prev) && isUpper(curr) && isUpper(next) {
			return Noop
		}

		if preserveNumberFormatting {
			if (curr == '.' || curr == ',') &&
				isNumber(prev) && isNumber(next) {
				return Noop
			}
		}

		if unicode.IsSpace(curr) {
			return SkipSplit
		}
		for _, d := range delimiters {
			if curr == d {
				return SkipSplit
			}
		}

		if splitBeforeNumber {
			if isNumber(curr) && !isNumber(prev) {
				if preserveNumberFormatting && (prev == '.' || prev == ',') {
					return Noop
				}
				return Split
			}
		}

		if splitAfterNumber {
			if isNumber(prev) && !isNumber(curr) {
				return Split
			}
		}

		if splitCase {
			if !isUpper(prev) && isUpper(curr) {
				return Split
			}
		}

		if splitAcronym {
			if isUpper(prev) && isUpper(curr) && isLower(next) {
				return Split
			}
		}

		return Noop
	}
}

// SplitOption are options that allow for configuring NewSplitFn
type SplitOption int

const (
	// SplitCase - FooBar -> Foo_Bar
	SplitCase SplitOption = iota
	// SplitAcronym - FOOBar -> Foo_Bar
	// It won't preserve FOO's case. If you want, you can set the Caser's initialisms so FOO will be in all caps
	SplitAcronym
	// SplitBeforeNumber - port80 -> port_80
	SplitBeforeNumber
	// SplitAfterNumber - 200status -> 200_status
	SplitAfterNumber
	// PreserveNumberFormatting - a.b.2,000.3.c -> a_b_2,000.3_c
	PreserveNumberFormatting
)

// SplitAction defines if and how to split a string
type SplitAction int

const (
	// Noop - Continue to next character
	Noop SplitAction = iota
	// Split - Split between words
	// e.g. to split between wordsWithoutDelimiters
	Split
	// SkipSplit - Split the word and drop the character
	// e.g. to split words with delimiters
	SkipSplit
	// Skip - Remove the character completely
	Skip
)

//nolint:gocyclo
func defaultSplitFn(prev, curr, next rune) SplitAction {
	// The most common case will be that it's just a letter so let lowercase letters return early since we know what they should do
	if isLower(curr) {
		return Noop
	}
	// Delimiters are _, -, ., and unicode spaces
	// Handle . lower down as it needs to happen after number exceptions
	if curr == '_' || curr == '-' || isSpace(curr) {
		return SkipSplit
	}

	if isUpper(curr) {
		if isLower(prev) {
			// fooBar
			return Split
		} else if isUpper(prev) && isLower(next) {
			// FOOBar
			return Split
		}
	}

	// Do numeric exceptions last to avoid perf penalty
	if unicode.IsNumber(prev) {
		// v4.3 is not split
		if (curr == '.' || curr == ',') && unicode.IsNumber(next) {
			return Noop
		}
		if !unicode.IsNumber(curr) && curr != '.' {
			return Split
		}
	}
	// While period is a default delimiter, keep it down here to avoid
	// penalty for other delimiters
	if curr == '.' {
		return SkipSplit
	}

	return Noop
}
