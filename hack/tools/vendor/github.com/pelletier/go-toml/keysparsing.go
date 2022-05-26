// Parsing keys handling both bare and quoted keys.

package toml

import (
	"errors"
	"fmt"
)

// Convert the bare key group string to an array.
// The input supports double quotation and single quotation,
// but escape sequences are not supported. Lexers must unescape them beforehand.
func parseKey(key string) ([]string, error) {
	runes := []rune(key)
	var groups []string

	if len(key) == 0 {
		return nil, errors.New("empty key")
	}

	idx := 0
	for idx < len(runes) {
		for ; idx < len(runes) && isSpace(runes[idx]); idx++ {
			// skip leading whitespace
		}
		if idx >= len(runes) {
			break
		}
		r := runes[idx]
		if isValidBareChar(r) {
			// parse bare key
			startIdx := idx
			endIdx := -1
			idx++
			for idx < len(runes) {
				r = runes[idx]
				if isValidBareChar(r) {
					idx++
				} else if r == '.' {
					endIdx = idx
					break
				} else if isSpace(r) {
					endIdx = idx
					for ; idx < len(runes) && isSpace(runes[idx]); idx++ {
						// skip trailing whitespace
					}
					if idx < len(runes) && runes[idx] != '.' {
						return nil, fmt.Errorf("invalid key character after whitespace: %c", runes[idx])
					}
					break
				} else {
					return nil, fmt.Errorf("invalid bare key character: %c", r)
				}
			}
			if endIdx == -1 {
				endIdx = idx
			}
			groups = append(groups, string(runes[startIdx:endIdx]))
		} else if r == '\'' {
			// parse single quoted key
			idx++
			startIdx := idx
			for {
				if idx >= len(runes) {
					return nil, fmt.Errorf("unclosed single-quoted key")
				}
				r = runes[idx]
				if r == '\'' {
					groups = append(groups, string(runes[startIdx:idx]))
					idx++
					break
				}
				idx++
			}
		} else if r == '"' {
			// parse double quoted key
			idx++
			startIdx := idx
			for {
				if idx >= len(runes) {
					return nil, fmt.Errorf("unclosed double-quoted key")
				}
				r = runes[idx]
				if r == '"' {
					groups = append(groups, string(runes[startIdx:idx]))
					idx++
					break
				}
				idx++
			}
		} else if r == '.' {
			idx++
			if idx >= len(runes) {
				return nil, fmt.Errorf("unexpected end of key")
			}
			r = runes[idx]
			if !isValidBareChar(r) && r != '\'' && r != '"' && r != ' ' {
				return nil, fmt.Errorf("expecting key part after dot")
			}
		} else {
			return nil, fmt.Errorf("invalid key character: %c", r)
		}
	}
	if len(groups) == 0 {
		return nil, fmt.Errorf("empty key")
	}
	return groups, nil
}

func isValidBareChar(r rune) bool {
	return isAlphanumeric(r) || r == '-' || isDigit(r)
}
