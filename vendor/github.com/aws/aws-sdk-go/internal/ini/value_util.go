package ini

import (
	"fmt"
)

// getStringValue will return a quoted string and the amount
// of bytes read
//
// an error will be returned if the string is not properly formatted
func getStringValue(b []rune) (int, error) {
	if b[0] != '"' {
		return 0, NewParseError("strings must start with '\"'")
	}

	endQuote := false
	i := 1

	for ; i < len(b) && !endQuote; i++ {
		if escaped := isEscaped(b[:i], b[i]); b[i] == '"' && !escaped {
			endQuote = true
			break
		} else if escaped {
			/*c, err := getEscapedByte(b[i])
			if err != nil {
				return 0, err
			}

			b[i-1] = c
			b = append(b[:i], b[i+1:]...)
			i--*/

			continue
		}
	}

	if !endQuote {
		return 0, NewParseError("missing '\"' in string value")
	}

	return i + 1, nil
}

// getBoolValue will return a boolean and the amount
// of bytes read
//
// an error will be returned if the boolean is not of a correct
// value
func getBoolValue(b []rune) (int, error) {
	if len(b) < 4 {
		return 0, NewParseError("invalid boolean value")
	}

	n := 0
	for _, lv := range literalValues {
		if len(lv) > len(b) {
			continue
		}

		if isLitValue(lv, b) {
			n = len(lv)
		}
	}

	if n == 0 {
		return 0, NewParseError("invalid boolean value")
	}

	return n, nil
}

// getNumericalValue will return a numerical string, the amount
// of bytes read, and the base of the number
//
// an error will be returned if the number is not of a correct
// value
func getNumericalValue(b []rune) (int, int, error) {
	if !isDigit(b[0]) {
		return 0, 0, NewParseError("invalid digit value")
	}

	i := 0
	helper := numberHelper{}

loop:
	for negativeIndex := 0; i < len(b); i++ {
		negativeIndex++

		if !isDigit(b[i]) {
			switch b[i] {
			case '-':
				if helper.IsNegative() || negativeIndex != 1 {
					return 0, 0, NewParseError("parse error '-'")
				}

				n := getNegativeNumber(b[i:])
				i += (n - 1)
				helper.Determine(b[i])
				continue
			case '.':
				if err := helper.Determine(b[i]); err != nil {
					return 0, 0, err
				}
			case 'e', 'E':
				if err := helper.Determine(b[i]); err != nil {
					return 0, 0, err
				}

				negativeIndex = 0
			case 'b':
				if helper.numberFormat == hex {
					break
				}
				fallthrough
			case 'o', 'x':
				if i == 0 && b[i] != '0' {
					return 0, 0, NewParseError("incorrect base format, expected leading '0'")
				}

				if i != 1 {
					return 0, 0, NewParseError(fmt.Sprintf("incorrect base format found %s at %d index", string(b[i]), i))
				}

				if err := helper.Determine(b[i]); err != nil {
					return 0, 0, err
				}
			default:
				if isWhitespace(b[i]) {
					break loop
				}

				if isNewline(b[i:]) {
					break loop
				}

				if !(helper.numberFormat == hex && isHexByte(b[i])) {
					if i+2 < len(b) && !isNewline(b[i:i+2]) {
						return 0, 0, NewParseError("invalid numerical character")
					} else if !isNewline([]rune{b[i]}) {
						return 0, 0, NewParseError("invalid numerical character")
					}

					break loop
				}
			}
		}
	}

	return helper.Base(), i, nil
}

// isDigit will return whether or not something is an integer
func isDigit(b rune) bool {
	return b >= '0' && b <= '9'
}

func hasExponent(v []rune) bool {
	return contains(v, 'e') || contains(v, 'E')
}

func isBinaryByte(b rune) bool {
	switch b {
	case '0', '1':
		return true
	default:
		return false
	}
}

func isOctalByte(b rune) bool {
	switch b {
	case '0', '1', '2', '3', '4', '5', '6', '7':
		return true
	default:
		return false
	}
}

func isHexByte(b rune) bool {
	if isDigit(b) {
		return true
	}
	return (b >= 'A' && b <= 'F') ||
		(b >= 'a' && b <= 'f')
}

func getValue(b []rune) (int, error) {
	i := 0

	for i < len(b) {
		if isNewline(b[i:]) {
			break
		}

		if isOp(b[i:]) {
			break
		}

		valid, n, err := isValid(b[i:])
		if err != nil {
			return 0, err
		}

		if !valid {
			break
		}

		i += n
	}

	return i, nil
}

// getNegativeNumber will return a negative number from a
// byte slice. This will iterate through all characters until
// a non-digit has been found.
func getNegativeNumber(b []rune) int {
	if b[0] != '-' {
		return 0
	}

	i := 1
	for ; i < len(b); i++ {
		if !isDigit(b[i]) {
			return i
		}
	}

	return i
}

// isEscaped will return whether or not the character is an escaped
// character.
func isEscaped(value []rune, b rune) bool {
	if len(value) == 0 {
		return false
	}

	switch b {
	case '\'': // single quote
	case '"': // quote
	case 'n': // newline
	case 't': // tab
	case '\\': // backslash
	default:
		return false
	}

	return value[len(value)-1] == '\\'
}

func getEscapedByte(b rune) (rune, error) {
	switch b {
	case '\'': // single quote
		return '\'', nil
	case '"': // quote
		return '"', nil
	case 'n': // newline
		return '\n', nil
	case 't': // table
		return '\t', nil
	case '\\': // backslash
		return '\\', nil
	default:
		return b, NewParseError(fmt.Sprintf("invalid escaped character %c", b))
	}
}

func removeEscapedCharacters(b []rune) []rune {
	for i := 0; i < len(b); i++ {
		if isEscaped(b[:i], b[i]) {
			c, err := getEscapedByte(b[i])
			if err != nil {
				return b
			}

			b[i-1] = c
			b = append(b[:i], b[i+1:]...)
			i--
		}
	}

	return b
}
