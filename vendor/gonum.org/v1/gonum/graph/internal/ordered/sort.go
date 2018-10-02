// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ordered

import "gonum.org/v1/gonum/graph"

// ByID implements the sort.Interface sorting a slice of graph.Node
// by ID.
type ByID []graph.Node

func (n ByID) Len() int           { return len(n) }
func (n ByID) Less(i, j int) bool { return n[i].ID() < n[j].ID() }
func (n ByID) Swap(i, j int)      { n[i], n[j] = n[j], n[i] }

// BySliceValues implements the sort.Interface sorting a slice of
// []int64 lexically by the values of the []int64.
type BySliceValues [][]int64

func (c BySliceValues) Len() int { return len(c) }
func (c BySliceValues) Less(i, j int) bool {
	a, b := c[i], c[j]
	l := len(a)
	if len(b) < l {
		l = len(b)
	}
	for k, v := range a[:l] {
		if v < b[k] {
			return true
		}
		if v > b[k] {
			return false
		}
	}
	return len(a) < len(b)
}
func (c BySliceValues) Swap(i, j int) { c[i], c[j] = c[j], c[i] }

// BySliceIDs implements the sort.Interface sorting a slice of
// []graph.Node lexically by the IDs of the []graph.Node.
type BySliceIDs [][]graph.Node

func (c BySliceIDs) Len() int { return len(c) }
func (c BySliceIDs) Less(i, j int) bool {
	a, b := c[i], c[j]
	l := len(a)
	if len(b) < l {
		l = len(b)
	}
	for k, v := range a[:l] {
		if v.ID() < b[k].ID() {
			return true
		}
		if v.ID() > b[k].ID() {
			return false
		}
	}
	return len(a) < len(b)
}
func (c BySliceIDs) Swap(i, j int) { c[i], c[j] = c[j], c[i] }

// Int64s implements the sort.Interface sorting a slice of
// int64.
type Int64s []int64

func (s Int64s) Len() int           { return len(s) }
func (s Int64s) Less(i, j int) bool { return s[i] < s[j] }
func (s Int64s) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Reverse reverses the order of nodes.
func Reverse(nodes []graph.Node) {
	for i, j := 0, len(nodes)-1; i < j; i, j = i+1, j-1 {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	}
}
