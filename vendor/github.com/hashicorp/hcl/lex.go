package hcl

import (
	"unicode"
	"unicode/utf8"
)

type lexModeValue byte

const (
	lexModeUnknown lexModeValue = iota
	lexModeHcl
	lexModeJson
)

// lexMode returns whether we're going to be parsing in JSON
// mode or HCL mode.
func lexMode(v []byte) lexModeValue {
	var (
		r      rune
		w      int
		offset int
	)

	for {
		r, w = utf8.DecodeRune(v[offset:])
		offset += w
		if unicode.IsSpace(r) {
			continue
		}
		if r == '{' {
			return lexModeJson
		}
		break
	}

	return lexModeHcl
}
