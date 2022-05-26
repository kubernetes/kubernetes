// Copyright 2020 Frederik Zipp. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gocyclo

import (
	"go/ast"
	"strings"
)

type directives []string

func (ds directives) HasIgnore() bool {
	return ds.isPresent("ignore")
}

func (ds directives) isPresent(name string) bool {
	for _, d := range ds {
		if d == name {
			return true
		}
	}
	return false
}

func parseDirectives(doc *ast.CommentGroup) directives {
	if doc == nil {
		return directives{}
	}
	const prefix = "//gocyclo:"
	var ds directives
	for _, comment := range doc.List {
		if strings.HasPrefix(comment.Text, prefix) {
			ds = append(ds, strings.TrimSpace(strings.TrimPrefix(comment.Text, prefix)))
		}
	}
	return ds
}
