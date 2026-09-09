// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package utils

import "strings"

// TODO: Move these to kyaml

// PathSplitter splits a delimited string, permitting escaped delimiters.
func PathSplitter(path string, delimiter string) []string {
	ps := strings.Split(path, delimiter)
	var res []string

	// allow path to start with forward slash
	// i.e. /a/b/c
	if len(ps) > 1 && ps[0] == "" {
		ps = ps[1:]
	}

	res = append(res, ps[0])
	for i := 1; i < len(ps); i++ {
		last := len(res) - 1
		if strings.HasSuffix(res[last], `\`) {
			res[last] = strings.TrimSuffix(res[last], `\`) + delimiter + ps[i]
		} else {
			res = append(res, ps[i])
		}
	}
	return res
}

// SmarterPathSplitter splits a path, retaining bracketed elements.
// If the element is a list entry identifier (defined by the '='),
// it will retain the brackets.
// E.g. "[name=com.foo.someapp]" survives as one thing after splitting
// "spec.template.spec.containers.[name=com.foo.someapp].image"
// See kyaml/yaml/match.go for use of list entry identifiers.
// If the element is a mapping entry identifier, it will remove the
// brackets.
// E.g. "a.b.c" survives as one thing after splitting
// "metadata.annotations.[a.b.c]
// This function uses `PathSplitter`, so it also respects escaped delimiters.
func SmarterPathSplitter(path string, delimiter string) []string {
	var result []string
	split := PathSplitter(path, delimiter)

	for i := 0; i < len(split); i++ {
		elem := split[i]
		if strings.HasPrefix(elem, "[") && !strings.HasSuffix(elem, "]") {
			// continue until we find the matching "]"
			bracketed := []string{elem}
			for i < len(split)-1 {
				i++
				bracketed = append(bracketed, split[i])
				if strings.HasSuffix(split[i], "]") {
					break
				}
			}
			bracketedStr := strings.Join(bracketed, delimiter)
			if strings.Contains(bracketedStr, "=") {
				result = append(result, bracketedStr)
			} else {
				result = append(result, strings.Trim(bracketedStr, "[]"))
			}
		} else {
			result = append(result, elem)
		}
	}
	return result
}
