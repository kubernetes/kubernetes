package hcl

import (
	"unicode"
)

type lexModeValue byte

const (
	lexModeUnknown lexModeValue = iota
	lexModeHcl
	lexModeJson
)

// lexMode returns whether we're going to be parsing in JSON
// mode or HCL mode.
func lexMode(v string) lexModeValue {
	for _, r := range v {
		if unicode.IsSpace(r) {
			continue
		}

		if r == '{' {
			return lexModeJson
		} else {
			return lexModeHcl
		}
	}

	return lexModeHcl
}
