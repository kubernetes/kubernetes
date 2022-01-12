// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"strings"
)

// DeriveSeqIndentStyle derives the sequence indentation annotation value for the resource,
// originalYAML is the input yaml string,
// the style is decided by deriving the existing sequence indentation of first sequence node
func DeriveSeqIndentStyle(originalYAML string) string {
	lines := strings.Split(originalYAML, "\n")
	for i, line := range lines {
		elems := strings.SplitN(line, "- ", 2)
		if len(elems) != 2 {
			continue
		}
		// prefix of "- " must be sequence of spaces
		if strings.Trim(elems[0], " ") != "" {
			continue
		}
		numSpacesBeforeSeqElem := len(elems[0])

		// keyLine is the line before the first sequence element
		keyLine := keyLineBeforeSeqElem(lines, i)
		if keyLine == "" {
			// there is no keyLine for this sequence node
			// all of those lines are comments
			continue
		}
		numSpacesBeforeKeyElem := len(keyLine) - len(strings.TrimLeft(keyLine, " "))
		trimmedKeyLine := strings.Trim(keyLine, " ")
		if strings.Count(trimmedKeyLine, ":") != 1 || !strings.HasSuffix(trimmedKeyLine, ":") {
			// if the key line doesn't contain only one : that too at the end,
			// this is not a sequence node, it is a wrapped sequence node string
			// ignore it
			continue
		}

		if numSpacesBeforeSeqElem == numSpacesBeforeKeyElem {
			return string(CompactSequenceStyle)
		}

		if numSpacesBeforeSeqElem-numSpacesBeforeKeyElem == 2 {
			return string(WideSequenceStyle)
		}
	}

	return string(CompactSequenceStyle)
}

// keyLineBeforeSeqElem iterates through the lines before the first seqElement
// and tries to find the non-comment key line for the sequence node
func keyLineBeforeSeqElem(lines []string, seqElemIndex int) string {
	// start with the previous line of sequence element
	i := seqElemIndex - 1
	for ; i >= 0; i-- {
		line := lines[i]
		trimmedLine := strings.Trim(line, " ")
		if strings.HasPrefix(trimmedLine, "#") { // commented line
			continue
		}
		// we have a non-commented line which can have a trailing comment
		parts := strings.SplitN(line, "#", 2)
		return parts[0] // throw away the trailing comment part
	}
	return ""

}
