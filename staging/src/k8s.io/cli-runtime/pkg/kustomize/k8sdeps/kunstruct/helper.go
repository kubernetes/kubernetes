/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package kunstruct provides unstructured from api machinery and factory for creating unstructured
package kunstruct

import (
	"fmt"
	"strings"
)

func parseFields(path string) ([]string, error) {
	if !strings.Contains(path, "[") {
		return strings.Split(path, "."), nil
	}

	var fields []string
	start := 0
	insideParentheses := false
	for i := range path {
		switch path[i] {
		case '.':
			if !insideParentheses {
				fields = append(fields, path[start:i])
				start = i + 1
			}
		case '[':
			if !insideParentheses {
				if i == start {
					start = i + 1
				} else {
					fields = append(fields, path[start:i])
					start = i + 1
				}
				insideParentheses = true
			} else {
				return nil, fmt.Errorf("nested parentheses are not allowed: %s", path)
			}
		case ']':
			if insideParentheses {
				fields = append(fields, path[start:i])
				start = i + 1
				insideParentheses = false
			} else {
				return nil, fmt.Errorf("invalid field path %s", path)
			}
		}
	}
	if start < len(path)-1 {
		fields = append(fields, path[start:])
	}
	for i, f := range fields {
		if strings.HasPrefix(f, "\"") || strings.HasPrefix(f, "'") {
			fields[i] = strings.Trim(f, "\"'")
		}
	}
	return fields, nil
}
