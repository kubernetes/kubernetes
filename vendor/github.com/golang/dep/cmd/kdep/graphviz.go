// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"hash/fnv"
	"strings"
)

type graphviz struct {
	ps []*gvnode
	b  bytes.Buffer
	h  map[string]uint32
}

type gvnode struct {
	project  string
	version  string
	children []string
}

func (g graphviz) New() *graphviz {
	ga := &graphviz{
		ps: []*gvnode{},
		h:  make(map[string]uint32),
	}
	return ga
}

func (g graphviz) output() bytes.Buffer {
	g.b.WriteString("digraph {\n\tnode [shape=box];")

	for _, gvp := range g.ps {
		// Create node string
		g.b.WriteString(fmt.Sprintf("\n\t%d [label=\"%s\"];", gvp.hash(), gvp.label()))
	}

	// Store relations to avoid duplication
	rels := make(map[string]bool)

	// Create relations
	for _, dp := range g.ps {
		for _, bsc := range dp.children {
			for pr, hsh := range g.h {
				if isPathPrefix(bsc, pr) {
					r := fmt.Sprintf("\n\t%d -> %d", g.h[dp.project], hsh)

					if _, ex := rels[r]; !ex {
						g.b.WriteString(r + ";")
						rels[r] = true
					}

				}
			}
		}
	}

	g.b.WriteString("\n}")
	return g.b
}

func (g *graphviz) createNode(project, version string, children []string) {
	pr := &gvnode{
		project:  project,
		version:  version,
		children: children,
	}

	g.h[pr.project] = pr.hash()
	g.ps = append(g.ps, pr)
}

func (dp gvnode) hash() uint32 {
	h := fnv.New32a()
	h.Write([]byte(dp.project))
	return h.Sum32()
}

func (dp gvnode) label() string {
	label := []string{dp.project}

	if dp.version != "" {
		label = append(label, dp.version)
	}

	return strings.Join(label, "\\n")
}

// isPathPrefix ensures that the literal string prefix is a path tree match and
// guards against possibilities like this:
//
// github.com/sdboyer/foo
// github.com/sdboyer/foobar/baz
//
// Verify that prefix is path match and either the input is the same length as
// the match (in which case we know they're equal), or that the next character
// is a "/". (Import paths are defined to always use "/", not the OS-specific
// path separator.)
func isPathPrefix(path, pre string) bool {
	pathlen, prflen := len(path), len(pre)
	if pathlen < prflen || path[0:prflen] != pre {
		return false
	}

	return prflen == pathlen || strings.Index(path[prflen:], "/") == 0
}
