// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dot

import (
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/internal/ordered"
)

// Node is a DOT graph node.
type Node interface {
	// DOTID returns a DOT node ID.
	//
	// An ID is one of the following:
	//
	//  - a string of alphabetic ([a-zA-Z\x80-\xff]) characters, underscores ('_').
	//    digits ([0-9]), not beginning with a digit.
	//  - a numeral [-]?(.[0-9]+ | [0-9]+(.[0-9]*)?).
	//  - a double-quoted string ("...") possibly containing escaped quotes (\").
	//  - an HTML string (<...>).
	DOTID() string
}

// Attributers are graph.Graph values that specify top-level DOT
// attributes.
type Attributers interface {
	DOTAttributers() (graph, node, edge encoding.Attributer)
}

// Porter defines the behavior of graph.Edge values that can specify
// connection ports for their end points. The returned port corresponds
// to the DOT node port to be used by the edge, compass corresponds
// to DOT compass point to which the edge will be aimed.
type Porter interface {
	// FromPort returns the port and compass for
	// the From node of a graph.Edge.
	FromPort() (port, compass string)

	// ToPort returns the port and compass for
	// the To node of a graph.Edge.
	ToPort() (port, compass string)
}

// Structurer represents a graph.Graph that can define subgraphs.
type Structurer interface {
	Structure() []Graph
}

// MultiStructurer represents a graph.Multigraph that can define subgraphs.
type MultiStructurer interface {
	Structure() []Multigraph
}

// Graph wraps named graph.Graph values.
type Graph interface {
	graph.Graph
	DOTID() string
}

// Multigraph wraps named graph.Multigraph values.
type Multigraph interface {
	graph.Multigraph
	DOTID() string
}

// Subgrapher wraps graph.Node values that represent subgraphs.
type Subgrapher interface {
	Subgraph() graph.Graph
}

// MultiSubgrapher wraps graph.Node values that represent subgraphs.
type MultiSubgrapher interface {
	Subgraph() graph.Multigraph
}

// Marshal returns the DOT encoding for the graph g, applying the prefix and
// indent to the encoding. Name is used to specify the graph name. If name is
// empty and g implements Graph, the returned string from DOTID will be used.
//
// Graph serialization will work for a graph.Graph without modification,
// however, advanced GraphViz DOT features provided by Marshal depend on
// implementation of the Node, Attributer, Porter, Attributers, Structurer,
// Subgrapher and Graph interfaces.
//
// Attributes and IDs are quoted if needed during marshalling.
func Marshal(g graph.Graph, name, prefix, indent string) ([]byte, error) {
	var p simpleGraphPrinter
	p.indent = indent
	p.prefix = prefix
	p.visited = make(map[edge]bool)
	err := p.print(g, name, false, false)
	if err != nil {
		return nil, err
	}
	return p.buf.Bytes(), nil
}

// MarshalMulti returns the DOT encoding for the multigraph g, applying the
// prefix and indent to the encoding. Name is used to specify the graph name. If
// name is empty and g implements Graph, the returned string from DOTID will be
// used.
//
// Graph serialization will work for a graph.Multigraph without modification,
// however, advanced GraphViz DOT features provided by Marshal depend on
// implementation of the Node, Attributer, Porter, Attributers, Structurer,
// MultiSubgrapher and Multigraph interfaces.
//
// Attributes and IDs are quoted if needed during marshalling.
func MarshalMulti(g graph.Multigraph, name, prefix, indent string) ([]byte, error) {
	var p multiGraphPrinter
	p.indent = indent
	p.prefix = prefix
	p.visited = make(map[line]bool)
	err := p.print(g, name, false, false)
	if err != nil {
		return nil, err
	}
	return p.buf.Bytes(), nil
}

type printer struct {
	buf bytes.Buffer

	prefix string
	indent string
	depth  int

	err error
}

type edge struct {
	inGraph  string
	from, to int64
}

func (p *simpleGraphPrinter) print(g graph.Graph, name string, needsIndent, isSubgraph bool) error {
	if name == "" {
		if g, ok := g.(Graph); ok {
			name = g.DOTID()
		}
	}

	_, isDirected := g.(graph.Directed)
	p.printFrontMatter(name, needsIndent, isSubgraph, isDirected, true)

	if a, ok := g.(Attributers); ok {
		p.writeAttributeComplex(a)
	}
	if s, ok := g.(Structurer); ok {
		for _, g := range s.Structure() {
			_, subIsDirected := g.(graph.Directed)
			if subIsDirected != isDirected {
				return errors.New("dot: mismatched graph type")
			}
			p.buf.WriteByte('\n')
			p.print(g, g.DOTID(), true, true)
		}
	}

	nodes := graph.NodesOf(g.Nodes())
	sort.Sort(ordered.ByID(nodes))

	havePrintedNodeHeader := false
	for _, n := range nodes {
		if s, ok := n.(Subgrapher); ok {
			// If the node is not linked to any other node
			// the graph needs to be written now.
			if g.From(n.ID()).Len() == 0 {
				g := s.Subgraph()
				_, subIsDirected := g.(graph.Directed)
				if subIsDirected != isDirected {
					return errors.New("dot: mismatched graph type")
				}
				if !havePrintedNodeHeader {
					p.newline()
					p.buf.WriteString("// Node definitions.")
					havePrintedNodeHeader = true
				}
				p.newline()
				p.print(g, graphID(g, n), false, true)
			}
			continue
		}
		if !havePrintedNodeHeader {
			p.newline()
			p.buf.WriteString("// Node definitions.")
			havePrintedNodeHeader = true
		}
		p.newline()
		p.writeNode(n)
		if a, ok := n.(encoding.Attributer); ok {
			p.writeAttributeList(a)
		}
		p.buf.WriteByte(';')
	}

	havePrintedEdgeHeader := false
	for _, n := range nodes {
		nid := n.ID()
		to := graph.NodesOf(g.From(nid))
		sort.Sort(ordered.ByID(to))
		for _, t := range to {
			tid := t.ID()
			if isDirected {
				if p.visited[edge{inGraph: name, from: nid, to: tid}] {
					continue
				}
				p.visited[edge{inGraph: name, from: nid, to: tid}] = true
			} else {
				if p.visited[edge{inGraph: name, from: nid, to: tid}] {
					continue
				}
				p.visited[edge{inGraph: name, from: nid, to: tid}] = true
				p.visited[edge{inGraph: name, from: tid, to: n.ID()}] = true
			}

			if !havePrintedEdgeHeader {
				p.buf.WriteByte('\n')
				p.buf.WriteString(strings.TrimRight(p.prefix, " \t\n")) // Trim whitespace suffix.
				p.newline()
				p.buf.WriteString("// Edge definitions.")
				havePrintedEdgeHeader = true
			}
			p.newline()

			if s, ok := n.(Subgrapher); ok {
				g := s.Subgraph()
				_, subIsDirected := g.(graph.Directed)
				if subIsDirected != isDirected {
					return errors.New("dot: mismatched graph type")
				}
				p.print(g, graphID(g, n), false, true)
			} else {
				p.writeNode(n)
			}
			e := g.Edge(nid, tid)
			porter, edgeIsPorter := e.(Porter)
			if edgeIsPorter {
				if e.From().ID() == nid {
					p.writePorts(porter.FromPort())
				} else {
					p.writePorts(porter.ToPort())
				}
			}

			if isDirected {
				p.buf.WriteString(" -> ")
			} else {
				p.buf.WriteString(" -- ")
			}

			if s, ok := t.(Subgrapher); ok {
				g := s.Subgraph()
				_, subIsDirected := g.(graph.Directed)
				if subIsDirected != isDirected {
					return errors.New("dot: mismatched graph type")
				}
				p.print(g, graphID(g, t), false, true)
			} else {
				p.writeNode(t)
			}
			if edgeIsPorter {
				if e.From().ID() == nid {
					p.writePorts(porter.ToPort())
				} else {
					p.writePorts(porter.FromPort())
				}
			}

			if a, ok := g.Edge(nid, tid).(encoding.Attributer); ok {
				p.writeAttributeList(a)
			}

			p.buf.WriteByte(';')
		}
	}

	p.closeBlock("}")

	return nil
}

func (p *printer) printFrontMatter(name string, needsIndent, isSubgraph, isDirected, isStrict bool) error {
	p.buf.WriteString(p.prefix)
	if needsIndent {
		for i := 0; i < p.depth; i++ {
			p.buf.WriteString(p.indent)
		}
	}

	if !isSubgraph && isStrict {
		p.buf.WriteString("strict ")
	}

	if isSubgraph {
		p.buf.WriteString("sub")
	} else if isDirected {
		p.buf.WriteString("di")
	}
	p.buf.WriteString("graph")

	if name != "" {
		p.buf.WriteByte(' ')
		p.buf.WriteString(quoteID(name))
	}

	p.openBlock(" {")
	return nil
}

func (p *printer) writeNode(n graph.Node) {
	p.buf.WriteString(quoteID(nodeID(n)))
}

func (p *printer) writePorts(port, cp string) {
	if port != "" {
		p.buf.WriteByte(':')
		p.buf.WriteString(quoteID(port))
	}
	if cp != "" {
		p.buf.WriteByte(':')
		p.buf.WriteString(cp)
	}
}

func nodeID(n graph.Node) string {
	switch n := n.(type) {
	case Node:
		return n.DOTID()
	default:
		return fmt.Sprint(n.ID())
	}
}

func graphID(g interface{}, n graph.Node) string {
	switch g := g.(type) {
	case Node:
		return g.DOTID()
	default:
		return nodeID(n)
	}
}

func (p *printer) writeAttributeList(a encoding.Attributer) {
	attributes := a.Attributes()
	switch len(attributes) {
	case 0:
	case 1:
		p.buf.WriteString(" [")
		p.buf.WriteString(quoteID(attributes[0].Key))
		p.buf.WriteByte('=')
		p.buf.WriteString(quoteID(attributes[0].Value))
		p.buf.WriteString("]")
	default:
		p.openBlock(" [")
		for _, att := range attributes {
			p.newline()
			p.buf.WriteString(quoteID(att.Key))
			p.buf.WriteByte('=')
			p.buf.WriteString(quoteID(att.Value))
		}
		p.closeBlock("]")
	}
}

var attType = []string{"graph", "node", "edge"}

func (p *printer) writeAttributeComplex(ca Attributers) {
	g, n, e := ca.DOTAttributers()
	haveWrittenBlock := false
	for i, a := range []encoding.Attributer{g, n, e} {
		attributes := a.Attributes()
		if len(attributes) == 0 {
			continue
		}
		if haveWrittenBlock {
			p.buf.WriteByte(';')
		}
		p.newline()
		p.buf.WriteString(attType[i])
		p.openBlock(" [")
		for _, att := range attributes {
			p.newline()
			p.buf.WriteString(quoteID(att.Key))
			p.buf.WriteByte('=')
			p.buf.WriteString(quoteID(att.Value))
		}
		p.closeBlock("]")
		haveWrittenBlock = true
	}
	if haveWrittenBlock {
		p.buf.WriteString(";\n")
	}
}

func (p *printer) newline() {
	p.buf.WriteByte('\n')
	p.buf.WriteString(p.prefix)
	for i := 0; i < p.depth; i++ {
		p.buf.WriteString(p.indent)
	}
}

func (p *printer) openBlock(b string) {
	p.buf.WriteString(b)
	p.depth++
}

func (p *printer) closeBlock(b string) {
	p.depth--
	p.newline()
	p.buf.WriteString(b)
}

type simpleGraphPrinter struct {
	printer
	visited map[edge]bool
}

type multiGraphPrinter struct {
	printer
	visited map[line]bool
}

type line struct {
	inGraph string
	id      int64
}

func (p *multiGraphPrinter) print(g graph.Multigraph, name string, needsIndent, isSubgraph bool) error {
	if name == "" {
		if g, ok := g.(Multigraph); ok {
			name = g.DOTID()
		}
	}

	_, isDirected := g.(graph.Directed)
	p.printFrontMatter(name, needsIndent, isSubgraph, isDirected, false)

	if a, ok := g.(Attributers); ok {
		p.writeAttributeComplex(a)
	}
	if s, ok := g.(MultiStructurer); ok {
		for _, g := range s.Structure() {
			_, subIsDirected := g.(graph.Directed)
			if subIsDirected != isDirected {
				return errors.New("dot: mismatched graph type")
			}
			p.buf.WriteByte('\n')
			p.print(g, g.DOTID(), true, true)
		}
	}

	nodes := graph.NodesOf(g.Nodes())
	sort.Sort(ordered.ByID(nodes))

	havePrintedNodeHeader := false
	for _, n := range nodes {
		if s, ok := n.(MultiSubgrapher); ok {
			// If the node is not linked to any other node
			// the graph needs to be written now.
			if g.From(n.ID()).Len() == 0 {
				g := s.Subgraph()
				_, subIsDirected := g.(graph.Directed)
				if subIsDirected != isDirected {
					return errors.New("dot: mismatched graph type")
				}
				if !havePrintedNodeHeader {
					p.newline()
					p.buf.WriteString("// Node definitions.")
					havePrintedNodeHeader = true
				}
				p.newline()
				p.print(g, graphID(g, n), false, true)
			}
			continue
		}
		if !havePrintedNodeHeader {
			p.newline()
			p.buf.WriteString("// Node definitions.")
			havePrintedNodeHeader = true
		}
		p.newline()
		p.writeNode(n)
		if a, ok := n.(encoding.Attributer); ok {
			p.writeAttributeList(a)
		}
		p.buf.WriteByte(';')
	}

	havePrintedEdgeHeader := false
	for _, n := range nodes {
		nid := n.ID()
		to := graph.NodesOf(g.From(nid))
		sort.Sort(ordered.ByID(to))

		for _, t := range to {
			tid := t.ID()

			lines := graph.LinesOf(g.Lines(nid, tid))
			sort.Sort(ordered.LinesByIDs(lines))

			for _, l := range lines {
				lid := l.ID()
				if p.visited[line{inGraph: name, id: lid}] {
					continue
				}
				p.visited[line{inGraph: name, id: lid}] = true

				if !havePrintedEdgeHeader {
					p.buf.WriteByte('\n')
					p.buf.WriteString(strings.TrimRight(p.prefix, " \t\n")) // Trim whitespace suffix.
					p.newline()
					p.buf.WriteString("// Edge definitions.")
					havePrintedEdgeHeader = true
				}
				p.newline()

				if s, ok := n.(MultiSubgrapher); ok {
					g := s.Subgraph()
					_, subIsDirected := g.(graph.Directed)
					if subIsDirected != isDirected {
						return errors.New("dot: mismatched graph type")
					}
					p.print(g, graphID(g, n), false, true)
				} else {
					p.writeNode(n)
				}

				porter, edgeIsPorter := l.(Porter)
				if edgeIsPorter {
					if l.From().ID() == nid {
						p.writePorts(porter.FromPort())
					} else {
						p.writePorts(porter.ToPort())
					}
				}

				if isDirected {
					p.buf.WriteString(" -> ")
				} else {
					p.buf.WriteString(" -- ")
				}

				if s, ok := t.(MultiSubgrapher); ok {
					g := s.Subgraph()
					_, subIsDirected := g.(graph.Directed)
					if subIsDirected != isDirected {
						return errors.New("dot: mismatched graph type")
					}
					p.print(g, graphID(g, t), false, true)
				} else {
					p.writeNode(t)
				}
				if edgeIsPorter {
					if l.From().ID() == nid {
						p.writePorts(porter.ToPort())
					} else {
						p.writePorts(porter.FromPort())
					}
				}

				if a, ok := l.(encoding.Attributer); ok {
					p.writeAttributeList(a)
				}

				p.buf.WriteByte(';')
			}
		}
	}

	p.closeBlock("}")

	return nil
}

// quoteID quotes the given string if needed in the context of an ID. If s is
// already quoted, or if s does not contain any spaces or special characters
// that need escaping, the original string is returned.
func quoteID(s string) string {
	// To use a keyword as an ID, it must be quoted.
	if isKeyword(s) {
		return strconv.Quote(s)
	}
	// Quote if s is not an ID. This includes strings containing spaces, except
	// if those spaces are used within HTML string IDs (e.g. <foo >).
	if !isID(s) {
		return strconv.Quote(s)
	}
	return s
}

// isKeyword reports whether the given string is a keyword in the DOT language.
func isKeyword(s string) bool {
	// ref: https://www.graphviz.org/doc/info/lang.html
	keywords := []string{"node", "edge", "graph", "digraph", "subgraph", "strict"}
	for _, keyword := range keywords {
		if strings.EqualFold(s, keyword) {
			return true
		}
	}
	return false
}

// FIXME: see if we rewrite this in another way to remove our regexp dependency.

// Regular expression to match identifier and numeral IDs.
var (
	reIdent   = regexp.MustCompile(`^[a-zA-Z\200-\377_][0-9a-zA-Z\200-\377_]*$`)
	reNumeral = regexp.MustCompile(`^[-]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)$`)
)

// isID reports whether the given string is an ID.
//
// An ID is one of the following:
//
// 1. Any string of alphabetic ([a-zA-Z\200-\377]) characters, underscores ('_')
//    or digits ([0-9]), not beginning with a digit;
// 2. a numeral [-]?(.[0-9]+ | [0-9]+(.[0-9]*)? );
// 3. any double-quoted string ("...") possibly containing escaped quotes (\");
// 4. an HTML string (<...>).
func isID(s string) bool {
	// 1. an identifier.
	if reIdent.MatchString(s) {
		return true
	}
	// 2. a numeral.
	if reNumeral.MatchString(s) {
		return true
	}
	// 3. double-quote string ID.
	if len(s) >= 2 && strings.HasPrefix(s, `"`) && strings.HasSuffix(s, `"`) {
		// Check that escape sequences within the double-quotes are valid.
		if _, err := strconv.Unquote(s); err == nil {
			return true
		}
	}
	// 4. HTML ID.
	return isHTMLID(s)
}

// isHTMLID reports whether the given string an HTML ID.
func isHTMLID(s string) bool {
	// HTML IDs have the format /^<.*>$/
	return len(s) >= 2 && strings.HasPrefix(s, "<") && strings.HasSuffix(s, ">")
}
