// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package utils

import "strings"

// PathSplitter splits a slash delimited string, permitting escaped slashes.
func PathSplitter(path string) []string {
	ps := strings.Split(path, "/")
	var res []string
	res = append(res, ps[0])
	for i := 1; i < len(ps); i++ {
		last := len(res) - 1
		if strings.HasSuffix(res[last], `\`) {
			res[last] = strings.TrimSuffix(res[last], `\`) + "/" + ps[i]
		} else {
			res = append(res, ps[i])
		}
	}
	return res
}
