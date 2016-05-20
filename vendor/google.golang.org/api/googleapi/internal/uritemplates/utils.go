// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uritemplates

func Expand(path string, values map[string]string) (string, error) {
	template, err := parse(path)
	if err != nil {
		return "", err
	}
	return template.Expand(values), nil
}
