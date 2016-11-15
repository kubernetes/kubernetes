// Parsing keys handling both bare and quoted keys.

package toml

import (
	"bytes"
	"fmt"
	"unicode"
)

func parseKey(key string) ([]string, error) {
	groups := []string{}
	var buffer bytes.Buffer
	inQuotes := false
	escapeNext := false
	ignoreSpace := true
	expectDot := false

	for _, char := range key {
		if ignoreSpace {
			if char == ' ' {
				continue
			}
			ignoreSpace = false
		}
		if escapeNext {
			buffer.WriteRune(char)
			escapeNext = false
			continue
		}
		switch char {
		case '\\':
			escapeNext = true
			continue
		case '"':
			inQuotes = !inQuotes
			expectDot = false
		case '.':
			if inQuotes {
				buffer.WriteRune(char)
			} else {
				groups = append(groups, buffer.String())
				buffer.Reset()
				ignoreSpace = true
				expectDot = false
			}
		case ' ':
			if inQuotes {
				buffer.WriteRune(char)
			} else {
				expectDot = true
			}
		default:
			if !inQuotes && !isValidBareChar(char) {
				return nil, fmt.Errorf("invalid bare character: %c", char)
			}
			if !inQuotes && expectDot {
				return nil, fmt.Errorf("what?")
			}
			buffer.WriteRune(char)
			expectDot = false
		}
	}
	if inQuotes {
		return nil, fmt.Errorf("mismatched quotes")
	}
	if escapeNext {
		return nil, fmt.Errorf("unfinished escape sequence")
	}
	if buffer.Len() > 0 {
		groups = append(groups, buffer.String())
	}
	if len(groups) == 0 {
		return nil, fmt.Errorf("empty key")
	}
	return groups, nil
}

func isValidBareChar(r rune) bool {
	return isAlphanumeric(r) || r == '-' || unicode.IsNumber(r)
}
