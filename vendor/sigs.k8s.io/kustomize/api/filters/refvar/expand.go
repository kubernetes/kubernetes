// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package refvar

import (
	"fmt"
	"log"
	"strings"
)

const (
	operator        = '$'
	referenceOpener = '('
	referenceCloser = ')'
)

// syntaxWrap returns the input string wrapped by the expansion syntax.
func syntaxWrap(input string) string {
	var sb strings.Builder
	sb.WriteByte(operator)
	sb.WriteByte(referenceOpener)
	sb.WriteString(input)
	sb.WriteByte(referenceCloser)
	return sb.String()
}

// MappingFunc maps a string to anything.
type MappingFunc func(string) interface{}

// MakePrimitiveReplacer returns a MappingFunc that uses a map to do
// replacements, and a histogram to count map hits.
//
// Func behavior:
//
// If the input key is NOT found in the map, the key is wrapped up as
// as a variable declaration string and returned, e.g. key FOO becomes $(FOO).
// This string is presumably put back where it was found, and might get replaced
// later.
//
// If the key is found in the map, the value is returned if it is a primitive
// type (string, bool, number), and the hit is counted.
//
// If it's not a primitive type (e.g. a map, struct, func, etc.) then this
// function doesn't know what to do with it and it returns the key wrapped up
// again as if it had not been replaced.  This should probably be an error.
func MakePrimitiveReplacer(
	counts map[string]int, someMap map[string]interface{}) MappingFunc {
	return func(key string) interface{} {
		if value, ok := someMap[key]; ok {
			switch typedV := value.(type) {
			case string, int, int32, int64, float32, float64, bool:
				counts[key]++
				return typedV
			default:
				// If the value is some complicated type (e.g. a map or struct),
				// this function doesn't know how to jam it into a string,
				// so just pretend it was a cache miss.
				// Likely this should be an error instead of a silent failure,
				// since the programmer passed an impossible value.
				log.Printf(
					"MakePrimitiveReplacer: bad replacement type=%T val=%v",
					typedV, typedV)
				return syntaxWrap(key)
			}
		}
		// If unable to return the mapped variable, return it
		// as it was found, and a later mapping might be able to
		// replace it.
		return syntaxWrap(key)
	}
}

// DoReplacements replaces variable references in the input string
// using the mapping function.
func DoReplacements(input string, mapping MappingFunc) interface{} {
	var buf strings.Builder
	checkpoint := 0
	for cursor := 0; cursor < len(input); cursor++ {
		if input[cursor] == operator && cursor+1 < len(input) {
			// Copy the portion of the input string since the last
			// checkpoint into the buffer
			buf.WriteString(input[checkpoint:cursor])

			// Attempt to read the variable name as defined by the
			// syntax from the input string
			read, isVar, advance := tryReadVariableName(input[cursor+1:])

			if isVar {
				// We were able to read a variable name correctly;
				// apply the mapping to the variable name and copy the
				// bytes into the buffer
				mapped := mapping(read)
				if input == syntaxWrap(read) {
					// Preserve the type of variable
					return mapped
				}

				// Variable is used in a middle of a string
				buf.WriteString(fmt.Sprintf("%v", mapped))
			} else {
				// Not a variable name; copy the read bytes into the buffer
				buf.WriteString(read)
			}

			// Advance the cursor in the input string to account for
			// bytes consumed to read the variable name expression
			cursor += advance

			// Advance the checkpoint in the input string
			checkpoint = cursor + 1
		}
	}

	// Return the buffer and any remaining unwritten bytes in the
	// input string.
	return buf.String() + input[checkpoint:]
}

// tryReadVariableName attempts to read a variable name from the input
// string and returns the content read from the input, whether that content
// represents a variable name to perform mapping on, and the number of bytes
// consumed in the input string.
//
// The input string is assumed not to contain the initial operator.
func tryReadVariableName(input string) (string, bool, int) {
	switch input[0] {
	case operator:
		// Escaped operator; return it.
		return input[0:1], false, 1
	case referenceOpener:
		// Scan to expression closer
		for i := 1; i < len(input); i++ {
			if input[i] == referenceCloser {
				return input[1:i], true, i + 1
			}
		}

		// Incomplete reference; return it.
		return string(operator) + string(referenceOpener), false, 1
	default:
		// Not the beginning of an expression, ie, an operator
		// that doesn't begin an expression.  Return the operator
		// and the first rune in the string.
		return string(operator) + string(input[0]), false, 1
	}
}
