package v2

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

var (
	// according to rfc7230
	reToken            = regexp.MustCompile(`^[^"(),/:;<=>?@[\]{}[:space:][:cntrl:]]+`)
	reQuotedValue      = regexp.MustCompile(`^[^\\"]+`)
	reEscapedCharacter = regexp.MustCompile(`^[[:blank:][:graph:]]`)
)

// parseForwardedHeader is a benevolent parser of Forwarded header defined in rfc7239. The header contains
// a comma-separated list of forwarding key-value pairs. Each list element is set by single proxy. The
// function parses only the first element of the list, which is set by the very first proxy. It returns a map
// of corresponding key-value pairs and an unparsed slice of the input string.
//
// Examples of Forwarded header values:
//
//  1. Forwarded: For=192.0.2.43; Proto=https,For="[2001:db8:cafe::17]",For=unknown
//  2. Forwarded: for="192.0.2.43:443"; host="registry.example.org", for="10.10.05.40:80"
//
// The first will be parsed into {"for": "192.0.2.43", "proto": "https"} while the second into
// {"for": "192.0.2.43:443", "host": "registry.example.org"}.
func parseForwardedHeader(forwarded string) (map[string]string, string, error) {
	// Following are states of forwarded header parser. Any state could transition to a failure.
	const (
		// terminating state; can transition to Parameter
		stateElement = iota
		// terminating state; can transition to KeyValueDelimiter
		stateParameter
		// can transition to Value
		stateKeyValueDelimiter
		// can transition to one of { QuotedValue, PairEnd }
		stateValue
		// can transition to one of { EscapedCharacter, PairEnd }
		stateQuotedValue
		// can transition to one of { QuotedValue }
		stateEscapedCharacter
		// terminating state; can transition to one of { Parameter, Element }
		statePairEnd
	)

	var (
		parameter string
		value     string
		parse     = forwarded[:]
		res       = map[string]string{}
		state     = stateElement
	)

Loop:
	for {
		// skip spaces unless in quoted value
		if state != stateQuotedValue && state != stateEscapedCharacter {
			parse = strings.TrimLeftFunc(parse, unicode.IsSpace)
		}

		if len(parse) == 0 {
			if state != stateElement && state != statePairEnd && state != stateParameter {
				return nil, parse, fmt.Errorf("unexpected end of input")
			}
			// terminating
			break
		}

		switch state {
		// terminate at list element delimiter
		case stateElement:
			if parse[0] == ',' {
				parse = parse[1:]
				break Loop
			}
			state = stateParameter

		// parse parameter (the key of key-value pair)
		case stateParameter:
			match := reToken.FindString(parse)
			if len(match) == 0 {
				return nil, parse, fmt.Errorf("failed to parse token at position %d", len(forwarded)-len(parse))
			}
			parameter = strings.ToLower(match)
			parse = parse[len(match):]
			state = stateKeyValueDelimiter

		// parse '='
		case stateKeyValueDelimiter:
			if parse[0] != '=' {
				return nil, parse, fmt.Errorf("expected '=', not '%c' at position %d", parse[0], len(forwarded)-len(parse))
			}
			parse = parse[1:]
			state = stateValue

		// parse value or quoted value
		case stateValue:
			if parse[0] == '"' {
				parse = parse[1:]
				state = stateQuotedValue
			} else {
				value = reToken.FindString(parse)
				if len(value) == 0 {
					return nil, parse, fmt.Errorf("failed to parse value at position %d", len(forwarded)-len(parse))
				}
				if _, exists := res[parameter]; exists {
					return nil, parse, fmt.Errorf("duplicate parameter %q at position %d", parameter, len(forwarded)-len(parse))
				}
				res[parameter] = value
				parse = parse[len(value):]
				value = ""
				state = statePairEnd
			}

		// parse a part of quoted value until the first backslash
		case stateQuotedValue:
			match := reQuotedValue.FindString(parse)
			value += match
			parse = parse[len(match):]
			switch {
			case len(parse) == 0:
				return nil, parse, fmt.Errorf("unterminated quoted string")
			case parse[0] == '"':
				res[parameter] = value
				value = ""
				parse = parse[1:]
				state = statePairEnd
			case parse[0] == '\\':
				parse = parse[1:]
				state = stateEscapedCharacter
			}

		// parse escaped character in a quoted string, ignore the backslash
		// transition back to QuotedValue state
		case stateEscapedCharacter:
			c := reEscapedCharacter.FindString(parse)
			if len(c) == 0 {
				return nil, parse, fmt.Errorf("invalid escape sequence at position %d", len(forwarded)-len(parse)-1)
			}
			value += c
			parse = parse[1:]
			state = stateQuotedValue

		// expect either a new key-value pair, new list or end of input
		case statePairEnd:
			switch parse[0] {
			case ';':
				parse = parse[1:]
				state = stateParameter
			case ',':
				state = stateElement
			default:
				return nil, parse, fmt.Errorf("expected ',' or ';', not %c at position %d", parse[0], len(forwarded)-len(parse))
			}
		}
	}

	return res, parse, nil
}
