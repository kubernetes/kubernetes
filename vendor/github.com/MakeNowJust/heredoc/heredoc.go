// Copyright (c) 2014 TSUYUSATO Kitsune
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php

// Package heredoc provides the here-document with keeping indent.
//
// Golang supports raw-string syntax.
//     doc := `
//     	Foo
//     	Bar
//     `
// But raw-string cannot recognize indent. Thus such content is indented string, equivalent to
//     "\n\tFoo\n\tBar\n"
// I dont't want this!
//
// However this problem is solved by package heredoc.
//     doc := heredoc.Doc(`
//     	Foo
//     	Bar
//     `)
// It is equivalent to
//     "Foo\nBar\n"
package heredoc

import (
	"fmt"
	"strings"
	"unicode"
)

// heredoc.Doc retutns unindented string as here-document.
//
// Process of making here-document:
//     1. Find most little indent size. (Skip empty lines)
//     2. Remove this indents of lines.
func Doc(raw string) string {
	skipFirstLine := false
	if raw[0] == '\n' {
		raw = raw[1:]
	} else {
		skipFirstLine = true
	}

	minIndentSize := int(^uint(0) >> 1) // Max value of type int
	lines := strings.Split(raw, "\n")

	// 1.
	for i, line := range lines {
		if i == 0 && skipFirstLine {
			continue
		}

		indentSize := 0
		for _, r := range []rune(line) {
			if unicode.IsSpace(r) {
				indentSize += 1
			} else {
				break
			}
		}

		if len(line) == indentSize {
			if i == len(lines)-1 && indentSize < minIndentSize {
				lines[i] = ""
			}
		} else if indentSize < minIndentSize {
			minIndentSize = indentSize
		}
	}

	// 2.
	for i, line := range lines {
		if i == 0 && skipFirstLine {
			continue
		}

		if len(lines[i]) >= minIndentSize {
			lines[i] = line[minIndentSize:]
		}
	}

	return strings.Join(lines, "\n")
}

// heredoc.Docf returns unindented and formatted string as here-document.
// This format is same with package fmt's format.
func Docf(raw string, args ...interface{}) string {
	return fmt.Sprintf(Doc(raw), args...)
}
