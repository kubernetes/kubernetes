// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package kunstruct provides unstructured from api machinery and factory for creating unstructured
package kunstruct

import (
	"fmt"
	"strconv"
	"strings"
)

// A PathSection contains a list of nested fields, which may end with an
// indexable value. For instance, foo.bar resolves to a PathSection with 2
// fields and no index, while foo[0].bar resolves to two path sections, the
// first containing the field foo and the index 0, and the second containing
// the field bar, with no index. The latter PathSection references the bar
// field of the first item in the foo list
type PathSection struct {
	fields []string
	idx    int
}

func newPathSection() PathSection {
	return PathSection{idx: -1}
}

func appendNonEmpty(section *PathSection, field string) {
	if len(field) != 0 {
		section.fields = append(section.fields, field)
	}
}

func parseFields(path string) (result []PathSection, err error) {
	section := newPathSection()
	if !strings.Contains(path, "[") {
		section.fields = strings.Split(path, ".")
		result = append(result, section)
		return result, nil
	}

	start := 0
	insideParentheses := false
	for i, c := range path {
		switch c {
		case '.':
			if !insideParentheses {
				appendNonEmpty(&section, path[start:i])
				start = i + 1
			}
		case '[':
			if !insideParentheses {
				appendNonEmpty(&section, path[start:i])
				start = i + 1
				insideParentheses = true
			} else {
				return nil, fmt.Errorf("nested parentheses are not allowed: %s", path)
			}
		case ']':
			if insideParentheses {
				// Assign this index to the current
				// PathSection, save it to the result, then begin
				// a new PathSection
				tmpIdx, err := strconv.Atoi(path[start:i])
				if err == nil {
					// We have detected an integer so an array.
					section.idx = tmpIdx
				} else {
					// We have detected the downwardapi syntax
					appendNonEmpty(&section, path[start:i])
				}
				result = append(result, section)
				section = newPathSection()

				start = i + 1
				insideParentheses = false
			} else {
				return nil, fmt.Errorf("invalid field path %s", path)
			}
		}
	}
	if start < len(path)-1 {
		appendNonEmpty(&section, path[start:])
		result = append(result, section)
	}

	for _, section := range result {
		for i, f := range section.fields {
			if strings.HasPrefix(f, "\"") || strings.HasPrefix(f, "'") {
				section.fields[i] = strings.Trim(f, "\"'")
			}
		}
	}
	return result, nil
}
