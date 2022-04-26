package strcase

import "strings"

// WordCase is an enumeration of the ways to format a word.
type WordCase int

const (
	// Original - Preserve the original input strcase
	Original WordCase = iota
	// LowerCase - All letters lower cased (example)
	LowerCase
	// UpperCase - All letters upper cased (EXAMPLE)
	UpperCase
	// TitleCase - Only first letter upper cased (Example)
	TitleCase
	// CamelCase - TitleCase except lower case first word (exampleText)
	// Notably, even if the first word is an initialism, it will be lower
	// cased. This is important for code generators where capital letters
	// mean exported functions. i.e. jsonString(), not JSONString()
	CamelCase
)

// We have 3 convert functions for performance reasons
// The general convert could handle everything, but is not optimized
//
// The other two functions are optimized for the general use cases - that is the non-custom caser functions
// Case 1: Any Case and supports Go Initialisms
// Case 2: UpperCase words, which don't need to support initialisms since everything is in upper case

// convertWithoutInitialims only works for to UpperCase and LowerCase
//nolint:gocyclo
func convertWithoutInitialisms(input string, delimiter rune, wordCase WordCase) string {
	input = strings.TrimSpace(input)
	runes := []rune(input)
	if len(runes) == 0 {
		return ""
	}

	var b strings.Builder
	b.Grow(len(input) * 2) // In case we need to write delimiters where they weren't before

	var prev, curr rune
	next := runes[0] // 0 length will have already returned so safe to index
	inWord := false
	firstWord := true
	for i := 0; i < len(runes); i++ {
		prev = curr
		curr = next
		if i+1 == len(runes) {
			next = 0
		} else {
			next = runes[i+1]
		}

		switch defaultSplitFn(prev, curr, next) {
		case SkipSplit:
			if inWord && delimiter != 0 {
				b.WriteRune(delimiter)
			}
			inWord = false
			continue
		case Split:
			if inWord && delimiter != 0 {
				b.WriteRune(delimiter)
			}
			inWord = false
		}
		switch wordCase {
		case UpperCase:
			b.WriteRune(toUpper(curr))
		case LowerCase:
			b.WriteRune(toLower(curr))
		case TitleCase:
			if inWord {
				b.WriteRune(toLower(curr))
			} else {
				b.WriteRune(toUpper(curr))
			}
		case CamelCase:
			if inWord {
				b.WriteRune(toLower(curr))
			} else if firstWord {
				b.WriteRune(toLower(curr))
				firstWord = false
			} else {
				b.WriteRune(toUpper(curr))
			}
		default:
			// Must be original case
			b.WriteRune(curr)
		}
		inWord = inWord || true
	}
	return b.String()
}

// convertWithGoInitialisms changes a input string to a certain case with a
// delimiter, respecting go initialisms but not skip runes
//nolint:gocyclo
func convertWithGoInitialisms(input string, delimiter rune, wordCase WordCase) string {
	input = strings.TrimSpace(input)
	runes := []rune(input)
	if len(runes) == 0 {
		return ""
	}

	var b strings.Builder
	b.Grow(len(input) * 2) // In case we need to write delimiters where they weren't before

	firstWord := true

	addWord := func(start, end int) {
		if start == end {
			return
		}

		if !firstWord && delimiter != 0 {
			b.WriteRune(delimiter)
		}

		// Don't bother with initialisms if the word is longer than 5
		// A quick proxy to avoid the extra memory allocations
		if end-start <= 5 {
			key := strings.ToUpper(string(runes[start:end]))
			if golintInitialisms[key] {
				if !firstWord || wordCase != CamelCase {
					b.WriteString(key)
					firstWord = false
					return
				}
			}
		}

		for i := start; i < end; i++ {
			r := runes[i]
			switch wordCase {
			case UpperCase:
				panic("use convertWithoutInitialisms instead")
			case LowerCase:
				b.WriteRune(toLower(r))
			case TitleCase:
				if i == start {
					b.WriteRune(toUpper(r))
				} else {
					b.WriteRune(toLower(r))
				}
			case CamelCase:
				if !firstWord && i == start {
					b.WriteRune(toUpper(r))
				} else {
					b.WriteRune(toLower(r))
				}
			default:
				b.WriteRune(r)
			}
		}
		firstWord = false
	}

	var prev, curr rune
	next := runes[0] // 0 length will have already returned so safe to index
	wordStart := 0
	for i := 0; i < len(runes); i++ {
		prev = curr
		curr = next
		if i+1 == len(runes) {
			next = 0
		} else {
			next = runes[i+1]
		}

		switch defaultSplitFn(prev, curr, next) {
		case Split:
			addWord(wordStart, i)
			wordStart = i
		case SkipSplit:
			addWord(wordStart, i)
			wordStart = i + 1
		}
	}

	if wordStart != len(runes) {
		addWord(wordStart, len(runes))
	}
	return b.String()
}

// convert changes a input string to a certain case with a delimiter,
// respecting arbitrary initialisms and skip characters
//nolint:gocyclo
func convert(input string, fn SplitFn, delimiter rune, wordCase WordCase,
	initialisms map[string]bool) string {
	input = strings.TrimSpace(input)
	runes := []rune(input)
	if len(runes) == 0 {
		return ""
	}

	var b strings.Builder
	b.Grow(len(input) * 2) // In case we need to write delimiters where they weren't before

	firstWord := true
	var skipIndexes []int

	addWord := func(start, end int) {
		// If you have nothing good to say, say nothing at all
		if start == end || len(skipIndexes) == end-start {
			skipIndexes = nil
			return
		}

		// If you have something to say, start with a delimiter
		if !firstWord && delimiter != 0 {
			b.WriteRune(delimiter)
		}

		// Check if you're an initialism
		// Note - we don't check skip characters here since initialisms
		// will probably never have junk characters in between
		// I'm open to it if there is a use case
		if initialisms != nil {
			var word strings.Builder
			for i := start; i < end; i++ {
				word.WriteRune(toUpper(runes[i]))
			}
			key := word.String()
			if initialisms[key] {
				if !firstWord || wordCase != CamelCase {
					b.WriteString(key)
					firstWord = false
					return
				}
			}
		}

		skipIdx := 0
		for i := start; i < end; i++ {
			if len(skipIndexes) > 0 && skipIdx < len(skipIndexes) && i == skipIndexes[skipIdx] {
				skipIdx++
				continue
			}
			r := runes[i]
			switch wordCase {
			case UpperCase:
				b.WriteRune(toUpper(r))
			case LowerCase:
				b.WriteRune(toLower(r))
			case TitleCase:
				if i == start {
					b.WriteRune(toUpper(r))
				} else {
					b.WriteRune(toLower(r))
				}
			case CamelCase:
				if !firstWord && i == start {
					b.WriteRune(toUpper(r))
				} else {
					b.WriteRune(toLower(r))
				}
			default:
				b.WriteRune(r)
			}
		}
		firstWord = false
		skipIndexes = nil
	}

	var prev, curr rune
	next := runes[0] // 0 length will have already returned so safe to index
	wordStart := 0
	for i := 0; i < len(runes); i++ {
		prev = curr
		curr = next
		if i+1 == len(runes) {
			next = 0
		} else {
			next = runes[i+1]
		}

		switch fn(prev, curr, next) {
		case Skip:
			skipIndexes = append(skipIndexes, i)
		case Split:
			addWord(wordStart, i)
			wordStart = i
		case SkipSplit:
			addWord(wordStart, i)
			wordStart = i + 1
		}
	}

	if wordStart != len(runes) {
		addWord(wordStart, len(runes))
	}
	return b.String()
}
