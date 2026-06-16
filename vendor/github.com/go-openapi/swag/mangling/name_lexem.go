// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package mangling

import (
	"bytes"
	"strings"
	"unicode"
	"unicode/utf8"
)

type (
	lexemKind uint8

	nameLexem struct {
		original          string
		matchedInitialism string
		kind              lexemKind
	}
)

const (
	lexemKindCasualName lexemKind = iota
	lexemKindInitialismName
)

func newInitialismNameLexem(original, matchedInitialism string) nameLexem {
	return nameLexem{
		kind:              lexemKindInitialismName,
		original:          original,
		matchedInitialism: matchedInitialism,
	}
}

func newCasualNameLexem(original string) nameLexem {
	return nameLexem{
		kind:     lexemKindCasualName,
		original: trim(original), // TODO: save on calls to trim
	}
}

// WriteTitleized writes the titleized lexeme to a bytes.Buffer.
//
// If the first letter cannot be capitalized, it doesn't write anything and return false,
// so the caller may attempt some workaround strategy.
func (l nameLexem) WriteTitleized(w *bytes.Buffer, alwaysUpper bool) bool {
	if l.kind == lexemKindInitialismName {
		w.WriteString(l.matchedInitialism)

		return true
	}

	if len(l.original) == 0 {
		return true
	}

	if len(l.original) == 1 {
		// identifier is too short: casing will depend on the context
		firstByte := l.original[0]
		switch {
		case 'A' <= firstByte && firstByte <= 'Z':
			// safe
			w.WriteByte(firstByte)

			return true
		case alwaysUpper && 'a' <= firstByte && firstByte <= 'z':
			w.WriteByte(firstByte - 'a' + 'A')

			return true
		default:

			// not a letter: skip and let the caller decide
			return false
		}
	}

	if firstByte := l.original[0]; firstByte < utf8.RuneSelf {
		// ASCII
		switch {
		case 'A' <= firstByte && firstByte <= 'Z':
			// already an upper case letter
			w.WriteString(l.original)

			return true
		case 'a' <= firstByte && firstByte <= 'z':
			w.WriteByte(firstByte - 'a' + 'A')
			w.WriteString(l.original[1:])

			return true
		default:
			// not a good candidate: doesn't start with a letter
			return false
		}
	}

	// unicode
	firstRune, idx := utf8.DecodeRuneInString(l.original)
	if !unicode.IsLetter(firstRune) || !unicode.IsUpper(unicode.ToUpper(firstRune)) {
		// not a good candidate: doesn't start with a letter
		// or a rune for which case doesn't make sense (e.g. East-Asian runes etc)
		return false
	}

	rest := l.original[idx:]
	w.WriteRune(unicode.ToUpper(firstRune))
	w.WriteString(strings.ToLower(rest))

	return true
}

// WriteLower is like write titleized but it writes a lower-case version of the lexeme.
//
// Similarly, there is no writing if the casing of the first rune doesn't make sense.
func (l nameLexem) WriteLower(w *bytes.Buffer, alwaysLower bool) bool {
	if l.kind == lexemKindInitialismName {
		w.WriteString(lower(l.matchedInitialism))

		return true
	}

	if len(l.original) == 0 {
		return true
	}

	if len(l.original) == 1 {
		// identifier is too short: casing will depend on the context
		firstByte := l.original[0]
		switch {
		case 'a' <= firstByte && firstByte <= 'z':
			// safe
			w.WriteByte(firstByte)

			return true
		case alwaysLower && 'A' <= firstByte && firstByte <= 'Z':
			w.WriteByte(firstByte - 'A' + 'a')

			return true
		default:

			// not a letter: skip and let the caller decide
			return false
		}
	}

	if firstByte := l.original[0]; firstByte < utf8.RuneSelf {
		// ASCII
		switch {
		case 'a' <= firstByte && firstByte <= 'z':
			// already a lower case letter
			w.WriteString(l.original)

			return true
		case 'A' <= firstByte && firstByte <= 'Z':
			w.WriteByte(firstByte - 'A' + 'a')
			w.WriteString(l.original[1:])

			return true
		default:
			// not a good candidate: doesn't start with a letter
			return false
		}
	}

	// unicode
	firstRune, idx := utf8.DecodeRuneInString(l.original)
	if !unicode.IsLetter(firstRune) || !unicode.IsLower(unicode.ToLower(firstRune)) {
		// not a good candidate: doesn't start with a letter
		// or a rune for which case doesn't make sense (e.g. East-Asian runes etc)
		return false
	}

	rest := l.original[idx:]
	w.WriteRune(unicode.ToLower(firstRune))
	w.WriteString(rest)

	return true
}

func (l nameLexem) GetOriginal() string {
	return l.original
}

func (l nameLexem) IsInitialism() bool {
	return l.kind == lexemKindInitialismName
}
