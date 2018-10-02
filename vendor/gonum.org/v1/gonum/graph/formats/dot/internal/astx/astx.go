// This file is dual licensed under CC0 and The gonum license.
//
// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Copyright ©2017 Robin Eklind.
// This file is made available under a Creative Commons CC0 1.0
// Universal Public Domain Dedication.

package astx

import (
	"fmt"
	"strings"

	"gonum.org/v1/gonum/graph/formats/dot/ast"
	"gonum.org/v1/gonum/graph/formats/dot/internal/token"
)

// === [ File ] ================================================================

// NewFile returns a new file based on the given graph.
func NewFile(graph interface{}) (*ast.File, error) {
	g, ok := graph.(*ast.Graph)
	if !ok {
		return nil, fmt.Errorf("invalid graph type; expected *ast.Graph, got %T", graph)
	}
	return &ast.File{Graphs: []*ast.Graph{g}}, nil
}

// AppendGraph appends graph to the given file.
func AppendGraph(file, graph interface{}) (*ast.File, error) {
	f, ok := file.(*ast.File)
	if !ok {
		return nil, fmt.Errorf("invalid file type; expected *ast.File, got %T", file)
	}
	g, ok := graph.(*ast.Graph)
	if !ok {
		return nil, fmt.Errorf("invalid graph type; expected *ast.Graph, got %T", graph)
	}
	f.Graphs = append(f.Graphs, g)
	return f, nil
}

// === [ Graphs ] ==============================================================

// NewGraph returns a new graph based on the given graph strictness, direction,
// optional ID and optional statements.
func NewGraph(strict, directed, optID, optStmts interface{}) (*ast.Graph, error) {
	s, ok := strict.(bool)
	if !ok {
		return nil, fmt.Errorf("invalid strictness type; expected bool, got %T", strict)
	}
	d, ok := directed.(bool)
	if !ok {
		return nil, fmt.Errorf("invalid direction type; expected bool, got %T", directed)
	}
	id, ok := optID.(string)
	if optID != nil && !ok {
		return nil, fmt.Errorf("invalid ID type; expected string or nil, got %T", optID)
	}
	stmts, ok := optStmts.([]ast.Stmt)
	if optStmts != nil && !ok {
		return nil, fmt.Errorf("invalid statements type; expected []ast.Stmt or nil, got %T", optStmts)
	}
	return &ast.Graph{Strict: s, Directed: d, ID: id, Stmts: stmts}, nil
}

// === [ Statements ] ==========================================================

// NewStmtList returns a new statement list based on the given statement.
func NewStmtList(stmt interface{}) ([]ast.Stmt, error) {
	s, ok := stmt.(ast.Stmt)
	if !ok {
		return nil, fmt.Errorf("invalid statement type; expected ast.Stmt, got %T", stmt)
	}
	return []ast.Stmt{s}, nil
}

// AppendStmt appends stmt to the given statement list.
func AppendStmt(list, stmt interface{}) ([]ast.Stmt, error) {
	l, ok := list.([]ast.Stmt)
	if !ok {
		return nil, fmt.Errorf("invalid statement list type; expected []ast.Stmt, got %T", list)
	}
	s, ok := stmt.(ast.Stmt)
	if !ok {
		return nil, fmt.Errorf("invalid statement type; expected ast.Stmt, got %T", stmt)
	}
	return append(l, s), nil
}

// --- [ Node statement ] ------------------------------------------------------

// NewNodeStmt returns a new node statement based on the given node and optional
// attributes.
func NewNodeStmt(node, optAttrs interface{}) (*ast.NodeStmt, error) {
	n, ok := node.(*ast.Node)
	if !ok {
		return nil, fmt.Errorf("invalid node type; expected *ast.Node, got %T", node)
	}
	attrs, ok := optAttrs.([]*ast.Attr)
	if optAttrs != nil && !ok {
		return nil, fmt.Errorf("invalid attributes type; expected []*ast.Attr or nil, got %T", optAttrs)
	}
	return &ast.NodeStmt{Node: n, Attrs: attrs}, nil
}

// --- [ Edge statement ] ------------------------------------------------------

// NewEdgeStmt returns a new edge statement based on the given source vertex,
// outgoing edge and optional attributes.
func NewEdgeStmt(from, to, optAttrs interface{}) (*ast.EdgeStmt, error) {
	f, ok := from.(ast.Vertex)
	if !ok {
		return nil, fmt.Errorf("invalid source vertex type; expected ast.Vertex, got %T", from)
	}
	t, ok := to.(*ast.Edge)
	if !ok {
		return nil, fmt.Errorf("invalid outgoing edge type; expected *ast.Edge, got %T", to)
	}
	attrs, ok := optAttrs.([]*ast.Attr)
	if optAttrs != nil && !ok {
		return nil, fmt.Errorf("invalid attributes type; expected []*ast.Attr or nil, got %T", optAttrs)
	}
	return &ast.EdgeStmt{From: f, To: t, Attrs: attrs}, nil
}

// NewEdge returns a new edge based on the given edge direction, destination
// vertex and optional outgoing edge.
func NewEdge(directed, vertex, optTo interface{}) (*ast.Edge, error) {
	d, ok := directed.(bool)
	if !ok {
		return nil, fmt.Errorf("invalid direction type; expected bool, got %T", directed)
	}
	v, ok := vertex.(ast.Vertex)
	if !ok {
		return nil, fmt.Errorf("invalid destination vertex type; expected ast.Vertex, got %T", vertex)
	}
	to, ok := optTo.(*ast.Edge)
	if optTo != nil && !ok {
		return nil, fmt.Errorf("invalid outgoing edge type; expected *ast.Edge or nil, got %T", optTo)
	}
	return &ast.Edge{Directed: d, Vertex: v, To: to}, nil
}

// --- [ Attribute statement ] -------------------------------------------------

// NewAttrStmt returns a new attribute statement based on the given graph
// component kind and attributes.
func NewAttrStmt(kind, optAttrs interface{}) (*ast.AttrStmt, error) {
	k, ok := kind.(ast.Kind)
	if !ok {
		return nil, fmt.Errorf("invalid graph component kind type; expected ast.Kind, got %T", kind)
	}
	attrs, ok := optAttrs.([]*ast.Attr)
	if optAttrs != nil && !ok {
		return nil, fmt.Errorf("invalid attributes type; expected []*ast.Attr or nil, got %T", optAttrs)
	}
	return &ast.AttrStmt{Kind: k, Attrs: attrs}, nil
}

// NewAttrList returns a new attribute list based on the given attribute.
func NewAttrList(attr interface{}) ([]*ast.Attr, error) {
	a, ok := attr.(*ast.Attr)
	if !ok {
		return nil, fmt.Errorf("invalid attribute type; expected *ast.Attr, got %T", attr)
	}
	return []*ast.Attr{a}, nil
}

// AppendAttr appends attr to the given attribute list.
func AppendAttr(list, attr interface{}) ([]*ast.Attr, error) {
	l, ok := list.([]*ast.Attr)
	if !ok {
		return nil, fmt.Errorf("invalid attribute list type; expected []*ast.Attr, got %T", list)
	}
	a, ok := attr.(*ast.Attr)
	if !ok {
		return nil, fmt.Errorf("invalid attribute type; expected *ast.Attr, got %T", attr)
	}
	return append(l, a), nil
}

// AppendAttrList appends the optional attrs to the given optional attribute
// list.
func AppendAttrList(optList, optAttrs interface{}) ([]*ast.Attr, error) {
	list, ok := optList.([]*ast.Attr)
	if optList != nil && !ok {
		return nil, fmt.Errorf("invalid attribute list type; expected []*ast.Attr or nil, got %T", optList)
	}
	attrs, ok := optAttrs.([]*ast.Attr)
	if optAttrs != nil && !ok {
		return nil, fmt.Errorf("invalid attributes type; expected []*ast.Attr or nil, got %T", optAttrs)
	}
	return append(list, attrs...), nil
}

// --- [ Attribute ] -----------------------------------------------------------

// NewAttr returns a new attribute based on the given key-value pair.
func NewAttr(key, val interface{}) (*ast.Attr, error) {
	k, ok := key.(string)
	if !ok {
		return nil, fmt.Errorf("invalid key type; expected string, got %T", key)
	}
	v, ok := val.(string)
	if !ok {
		return nil, fmt.Errorf("invalid value type; expected string, got %T", val)
	}
	return &ast.Attr{Key: k, Val: v}, nil
}

// --- [ Subgraph ] ------------------------------------------------------------

// NewSubgraph returns a new subgraph based on the given optional subgraph ID
// and optional statements.
func NewSubgraph(optID, optStmts interface{}) (*ast.Subgraph, error) {
	id, ok := optID.(string)
	if optID != nil && !ok {
		return nil, fmt.Errorf("invalid ID type; expected string or nil, got %T", optID)
	}
	stmts, ok := optStmts.([]ast.Stmt)
	if optStmts != nil && !ok {
		return nil, fmt.Errorf("invalid statements type; expected []ast.Stmt or nil, got %T", optStmts)
	}
	return &ast.Subgraph{ID: id, Stmts: stmts}, nil
}

// === [ Vertices ] ============================================================

// --- [ Node identifier ] -----------------------------------------------------

// NewNode returns a new node based on the given node id and optional port.
func NewNode(id, optPort interface{}) (*ast.Node, error) {
	i, ok := id.(string)
	if !ok {
		return nil, fmt.Errorf("invalid ID type; expected string, got %T", id)
	}
	port, ok := optPort.(*ast.Port)
	if optPort != nil && !ok {
		return nil, fmt.Errorf("invalid port type; expected *ast.Port or nil, got %T", optPort)
	}
	return &ast.Node{ID: i, Port: port}, nil
}

// NewPort returns a new port based on the given id and optional compass point.
func NewPort(id, optCompassPoint interface{}) (*ast.Port, error) {
	// Note, if optCompassPoint is nil, id may be either an identifier or a
	// compass point.
	//
	// The following strings are valid compass points:
	//
	//    "n", "ne", "e", "se", "s", "sw", "w", "nw", "c" and "_"
	i, ok := id.(string)
	if !ok {
		return nil, fmt.Errorf("invalid ID type; expected string, got %T", id)
	}

	// Early return if optional compass point is absent and ID is a valid compass
	// point.
	if optCompassPoint == nil {
		if compassPoint, ok := getCompassPoint(i); ok {
			return &ast.Port{CompassPoint: compassPoint}, nil
		}
	}

	c, ok := optCompassPoint.(string)
	if optCompassPoint != nil && !ok {
		return nil, fmt.Errorf("invalid compass point type; expected string or nil, got %T", optCompassPoint)
	}
	compassPoint, _ := getCompassPoint(c)
	return &ast.Port{ID: i, CompassPoint: compassPoint}, nil
}

// getCompassPoint returns the corresponding compass point to the given string,
// and a boolean value indicating if such a compass point exists.
func getCompassPoint(s string) (ast.CompassPoint, bool) {
	switch s {
	case "_":
		return ast.CompassPointDefault, true
	case "n":
		return ast.CompassPointNorth, true
	case "ne":
		return ast.CompassPointNorthEast, true
	case "e":
		return ast.CompassPointEast, true
	case "se":
		return ast.CompassPointSouthEast, true
	case "s":
		return ast.CompassPointSouth, true
	case "sw":
		return ast.CompassPointSouthWest, true
	case "w":
		return ast.CompassPointWest, true
	case "nw":
		return ast.CompassPointNorthWest, true
	case "c":
		return ast.CompassPointCenter, true
	}
	return ast.CompassPointNone, false
}

// === [ Identifiers ] =========================================================

// NewID returns a new identifier based on the given ID token.
func NewID(id interface{}) (string, error) {
	i, ok := id.(*token.Token)
	if !ok {
		return "", fmt.Errorf("invalid identifier type; expected *token.Token, got %T", id)
	}
	s := string(i.Lit)

	// As another aid for readability, dot allows double-quoted strings to span
	// multiple physical lines using the standard C convention of a backslash
	// immediately preceding a newline character.
	if strings.HasPrefix(s, `"`) && strings.HasSuffix(s, `"`) {
		// Strip "\\\n" sequences.
		s = strings.Replace(s, "\\\n", "", -1)
	}

	// TODO: Add support for concatenated using a '+' operator.

	return s, nil
}
