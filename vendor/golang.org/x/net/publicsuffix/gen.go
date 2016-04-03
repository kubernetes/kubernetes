// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

// This program generates table.go and table_test.go.
// Invoke as:
//
//	go run gen.go -version "xxx"       >table.go
//	go run gen.go -version "xxx" -test >table_test.go
//
// The version is derived from information found at
// https://github.com/publicsuffix/list/commits/master/public_suffix_list.dat
//
// To fetch a particular git revision, such as 5c70ccd250, pass
// -url "https://raw.githubusercontent.com/publicsuffix/list/5c70ccd250/public_suffix_list.dat"

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/net/idna"
)

const (
	nodesBitsChildren   = 9
	nodesBitsICANN      = 1
	nodesBitsTextOffset = 15
	nodesBitsTextLength = 6

	childrenBitsWildcard = 1
	childrenBitsNodeType = 2
	childrenBitsHi       = 14
	childrenBitsLo       = 14
)

var (
	maxChildren   int
	maxTextOffset int
	maxTextLength int
	maxHi         uint32
	maxLo         uint32
)

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func u32max(a, b uint32) uint32 {
	if a < b {
		return b
	}
	return a
}

const (
	nodeTypeNormal     = 0
	nodeTypeException  = 1
	nodeTypeParentOnly = 2
	numNodeType        = 3
)

func nodeTypeStr(n int) string {
	switch n {
	case nodeTypeNormal:
		return "+"
	case nodeTypeException:
		return "!"
	case nodeTypeParentOnly:
		return "o"
	}
	panic("unreachable")
}

var (
	labelEncoding = map[string]uint32{}
	labelsList    = []string{}
	labelsMap     = map[string]bool{}
	rules         = []string{}

	// validSuffix is used to check that the entries in the public suffix list
	// are in canonical form (after Punycode encoding). Specifically, capital
	// letters are not allowed.
	validSuffix = regexp.MustCompile(`^[a-z0-9_\!\*\-\.]+$`)

	crush  = flag.Bool("crush", true, "make the generated node text as small as possible")
	subset = flag.Bool("subset", false, "generate only a subset of the full table, for debugging")
	url    = flag.String("url",
		"https://publicsuffix.org/list/effective_tld_names.dat",
		"URL of the publicsuffix.org list. If empty, stdin is read instead")
	v       = flag.Bool("v", false, "verbose output (to stderr)")
	version = flag.String("version", "", "the effective_tld_names.dat version")
	test    = flag.Bool("test", false, "generate table_test.go")
)

func main() {
	if err := main1(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func main1() error {
	flag.Parse()
	if nodesBitsTextLength+nodesBitsTextOffset+nodesBitsICANN+nodesBitsChildren > 32 {
		return fmt.Errorf("not enough bits to encode the nodes table")
	}
	if childrenBitsLo+childrenBitsHi+childrenBitsNodeType+childrenBitsWildcard > 32 {
		return fmt.Errorf("not enough bits to encode the children table")
	}
	if *version == "" {
		return fmt.Errorf("-version was not specified")
	}
	var r io.Reader = os.Stdin
	if *url != "" {
		res, err := http.Get(*url)
		if err != nil {
			return err
		}
		if res.StatusCode != http.StatusOK {
			return fmt.Errorf("bad GET status for %s: %d", *url, res.Status)
		}
		r = res.Body
		defer res.Body.Close()
	}

	var root node
	icann := false
	buf := new(bytes.Buffer)
	br := bufio.NewReader(r)
	for {
		s, err := br.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		s = strings.TrimSpace(s)
		if strings.Contains(s, "BEGIN ICANN DOMAINS") {
			icann = true
			continue
		}
		if strings.Contains(s, "END ICANN DOMAINS") {
			icann = false
			continue
		}
		if s == "" || strings.HasPrefix(s, "//") {
			continue
		}
		s, err = idna.ToASCII(s)
		if err != nil {
			return err
		}
		if !validSuffix.MatchString(s) {
			return fmt.Errorf("bad publicsuffix.org list data: %q", s)
		}

		if *subset {
			switch {
			case s == "ac.jp" || strings.HasSuffix(s, ".ac.jp"):
			case s == "ak.us" || strings.HasSuffix(s, ".ak.us"):
			case s == "ao" || strings.HasSuffix(s, ".ao"):
			case s == "ar" || strings.HasSuffix(s, ".ar"):
			case s == "arpa" || strings.HasSuffix(s, ".arpa"):
			case s == "cy" || strings.HasSuffix(s, ".cy"):
			case s == "dyndns.org" || strings.HasSuffix(s, ".dyndns.org"):
			case s == "jp":
			case s == "kobe.jp" || strings.HasSuffix(s, ".kobe.jp"):
			case s == "kyoto.jp" || strings.HasSuffix(s, ".kyoto.jp"):
			case s == "om" || strings.HasSuffix(s, ".om"):
			case s == "uk" || strings.HasSuffix(s, ".uk"):
			case s == "uk.com" || strings.HasSuffix(s, ".uk.com"):
			case s == "tw" || strings.HasSuffix(s, ".tw"):
			case s == "zw" || strings.HasSuffix(s, ".zw"):
			case s == "xn--p1ai" || strings.HasSuffix(s, ".xn--p1ai"):
				// xn--p1ai is Russian-Cyrillic "рф".
			default:
				continue
			}
		}

		rules = append(rules, s)

		nt, wildcard := nodeTypeNormal, false
		switch {
		case strings.HasPrefix(s, "*."):
			s, nt = s[2:], nodeTypeParentOnly
			wildcard = true
		case strings.HasPrefix(s, "!"):
			s, nt = s[1:], nodeTypeException
		}
		labels := strings.Split(s, ".")
		for n, i := &root, len(labels)-1; i >= 0; i-- {
			label := labels[i]
			n = n.child(label)
			if i == 0 {
				if nt != nodeTypeParentOnly && n.nodeType == nodeTypeParentOnly {
					n.nodeType = nt
				}
				n.icann = n.icann && icann
				n.wildcard = n.wildcard || wildcard
			}
			labelsMap[label] = true
		}
	}
	labelsList = make([]string, 0, len(labelsMap))
	for label := range labelsMap {
		labelsList = append(labelsList, label)
	}
	sort.Strings(labelsList)

	p := printReal
	if *test {
		p = printTest
	}
	if err := p(buf, &root); err != nil {
		return err
	}

	b, err := format.Source(buf.Bytes())
	if err != nil {
		return err
	}
	_, err = os.Stdout.Write(b)
	return err
}

func printTest(w io.Writer, n *node) error {
	fmt.Fprintf(w, "// generated by go run gen.go; DO NOT EDIT\n\n")
	fmt.Fprintf(w, "package publicsuffix\n\nvar rules = [...]string{\n")
	for _, rule := range rules {
		fmt.Fprintf(w, "%q,\n", rule)
	}
	fmt.Fprintf(w, "}\n\nvar nodeLabels = [...]string{\n")
	if err := n.walk(w, printNodeLabel); err != nil {
		return err
	}
	fmt.Fprintf(w, "}\n")
	return nil
}

func printReal(w io.Writer, n *node) error {
	const header = `// generated by go run gen.go; DO NOT EDIT

package publicsuffix

const version = %q

const (
	nodesBitsChildren   = %d
	nodesBitsICANN      = %d
	nodesBitsTextOffset = %d
	nodesBitsTextLength = %d

	childrenBitsWildcard = %d
	childrenBitsNodeType = %d
	childrenBitsHi       = %d
	childrenBitsLo       = %d
)

const (
	nodeTypeNormal     = %d
	nodeTypeException  = %d
	nodeTypeParentOnly = %d
)

// numTLD is the number of top level domains.
const numTLD = %d

`
	fmt.Fprintf(w, header, *version,
		nodesBitsChildren, nodesBitsICANN, nodesBitsTextOffset, nodesBitsTextLength,
		childrenBitsWildcard, childrenBitsNodeType, childrenBitsHi, childrenBitsLo,
		nodeTypeNormal, nodeTypeException, nodeTypeParentOnly, len(n.children))

	text := makeText()
	if text == "" {
		return fmt.Errorf("internal error: makeText returned no text")
	}
	for _, label := range labelsList {
		offset, length := strings.Index(text, label), len(label)
		if offset < 0 {
			return fmt.Errorf("internal error: could not find %q in text %q", label, text)
		}
		maxTextOffset, maxTextLength = max(maxTextOffset, offset), max(maxTextLength, length)
		if offset >= 1<<nodesBitsTextOffset || length >= 1<<nodesBitsTextLength {
			return fmt.Errorf("text offset/length is too large: %d/%d", offset, length)
		}
		labelEncoding[label] = uint32(offset)<<nodesBitsTextLength | uint32(length)
	}
	fmt.Fprintf(w, "// Text is the combined text of all labels.\nconst text = ")
	for len(text) > 0 {
		n, plus := len(text), ""
		if n > 64 {
			n, plus = 64, " +"
		}
		fmt.Fprintf(w, "%q%s\n", text[:n], plus)
		text = text[n:]
	}

	n.walk(w, assignIndexes)

	fmt.Fprintf(w, `

// nodes is the list of nodes. Each node is represented as a uint32, which
// encodes the node's children, wildcard bit and node type (as an index into
// the children array), ICANN bit and text.
//
// In the //-comment after each node's data, the nodes indexes of the children
// are formatted as (n0x1234-n0x1256), with * denoting the wildcard bit. The
// nodeType is printed as + for normal, ! for exception, and o for parent-only
// nodes that have children but don't match a domain label in their own right.
// An I denotes an ICANN domain.
//
// The layout within the uint32, from MSB to LSB, is:
//	[%2d bits] unused
//	[%2d bits] children index
//	[%2d bits] ICANN bit
//	[%2d bits] text index
//	[%2d bits] text length
var nodes = [...]uint32{
`,
		32-nodesBitsChildren-nodesBitsICANN-nodesBitsTextOffset-nodesBitsTextLength,
		nodesBitsChildren, nodesBitsICANN, nodesBitsTextOffset, nodesBitsTextLength)
	if err := n.walk(w, printNode); err != nil {
		return err
	}
	fmt.Fprintf(w, `}

// children is the list of nodes' children, the parent's wildcard bit and the
// parent's node type. If a node has no children then their children index
// will be in the range [0, 6), depending on the wildcard bit and node type.
//
// The layout within the uint32, from MSB to LSB, is:
//	[%2d bits] unused
//	[%2d bits] wildcard bit
//	[%2d bits] node type
//	[%2d bits] high nodes index (exclusive) of children
//	[%2d bits] low nodes index (inclusive) of children
var children=[...]uint32{
`,
		32-childrenBitsWildcard-childrenBitsNodeType-childrenBitsHi-childrenBitsLo,
		childrenBitsWildcard, childrenBitsNodeType, childrenBitsHi, childrenBitsLo)
	for i, c := range childrenEncoding {
		s := "---------------"
		lo := c & (1<<childrenBitsLo - 1)
		hi := (c >> childrenBitsLo) & (1<<childrenBitsHi - 1)
		if lo != hi {
			s = fmt.Sprintf("n0x%04x-n0x%04x", lo, hi)
		}
		nodeType := int(c>>(childrenBitsLo+childrenBitsHi)) & (1<<childrenBitsNodeType - 1)
		wildcard := c>>(childrenBitsLo+childrenBitsHi+childrenBitsNodeType) != 0
		fmt.Fprintf(w, "0x%08x, // c0x%04x (%s)%s %s\n",
			c, i, s, wildcardStr(wildcard), nodeTypeStr(nodeType))
	}
	fmt.Fprintf(w, "}\n\n")
	fmt.Fprintf(w, "// max children %d (capacity %d)\n", maxChildren, 1<<nodesBitsChildren-1)
	fmt.Fprintf(w, "// max text offset %d (capacity %d)\n", maxTextOffset, 1<<nodesBitsTextOffset-1)
	fmt.Fprintf(w, "// max text length %d (capacity %d)\n", maxTextLength, 1<<nodesBitsTextLength-1)
	fmt.Fprintf(w, "// max hi %d (capacity %d)\n", maxHi, 1<<childrenBitsHi-1)
	fmt.Fprintf(w, "// max lo %d (capacity %d)\n", maxLo, 1<<childrenBitsLo-1)
	return nil
}

type node struct {
	label    string
	nodeType int
	icann    bool
	wildcard bool
	// nodesIndex and childrenIndex are the index of this node in the nodes
	// and the index of its children offset/length in the children arrays.
	nodesIndex, childrenIndex int
	// firstChild is the index of this node's first child, or zero if this
	// node has no children.
	firstChild int
	// children are the node's children, in strictly increasing node label order.
	children []*node
}

func (n *node) walk(w io.Writer, f func(w1 io.Writer, n1 *node) error) error {
	if err := f(w, n); err != nil {
		return err
	}
	for _, c := range n.children {
		if err := c.walk(w, f); err != nil {
			return err
		}
	}
	return nil
}

// child returns the child of n with the given label. The child is created if
// it did not exist beforehand.
func (n *node) child(label string) *node {
	for _, c := range n.children {
		if c.label == label {
			return c
		}
	}
	c := &node{
		label:    label,
		nodeType: nodeTypeParentOnly,
		icann:    true,
	}
	n.children = append(n.children, c)
	sort.Sort(byLabel(n.children))
	return c
}

type byLabel []*node

func (b byLabel) Len() int           { return len(b) }
func (b byLabel) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byLabel) Less(i, j int) bool { return b[i].label < b[j].label }

var nextNodesIndex int

// childrenEncoding are the encoded entries in the generated children array.
// All these pre-defined entries have no children.
var childrenEncoding = []uint32{
	0 << (childrenBitsLo + childrenBitsHi), // Without wildcard bit, nodeTypeNormal.
	1 << (childrenBitsLo + childrenBitsHi), // Without wildcard bit, nodeTypeException.
	2 << (childrenBitsLo + childrenBitsHi), // Without wildcard bit, nodeTypeParentOnly.
	4 << (childrenBitsLo + childrenBitsHi), // With wildcard bit, nodeTypeNormal.
	5 << (childrenBitsLo + childrenBitsHi), // With wildcard bit, nodeTypeException.
	6 << (childrenBitsLo + childrenBitsHi), // With wildcard bit, nodeTypeParentOnly.
}

var firstCallToAssignIndexes = true

func assignIndexes(w io.Writer, n *node) error {
	if len(n.children) != 0 {
		// Assign nodesIndex.
		n.firstChild = nextNodesIndex
		for _, c := range n.children {
			c.nodesIndex = nextNodesIndex
			nextNodesIndex++
		}

		// The root node's children is implicit.
		if firstCallToAssignIndexes {
			firstCallToAssignIndexes = false
			return nil
		}

		// Assign childrenIndex.
		maxChildren = max(maxChildren, len(childrenEncoding))
		if len(childrenEncoding) >= 1<<nodesBitsChildren {
			return fmt.Errorf("children table is too large")
		}
		n.childrenIndex = len(childrenEncoding)
		lo := uint32(n.firstChild)
		hi := lo + uint32(len(n.children))
		maxLo, maxHi = u32max(maxLo, lo), u32max(maxHi, hi)
		if lo >= 1<<childrenBitsLo || hi >= 1<<childrenBitsHi {
			return fmt.Errorf("children lo/hi is too large: %d/%d", lo, hi)
		}
		enc := hi<<childrenBitsLo | lo
		enc |= uint32(n.nodeType) << (childrenBitsLo + childrenBitsHi)
		if n.wildcard {
			enc |= 1 << (childrenBitsLo + childrenBitsHi + childrenBitsNodeType)
		}
		childrenEncoding = append(childrenEncoding, enc)
	} else {
		n.childrenIndex = n.nodeType
		if n.wildcard {
			n.childrenIndex += numNodeType
		}
	}
	return nil
}

func printNode(w io.Writer, n *node) error {
	for _, c := range n.children {
		s := "---------------"
		if len(c.children) != 0 {
			s = fmt.Sprintf("n0x%04x-n0x%04x", c.firstChild, c.firstChild+len(c.children))
		}
		encoding := labelEncoding[c.label]
		if c.icann {
			encoding |= 1 << (nodesBitsTextLength + nodesBitsTextOffset)
		}
		encoding |= uint32(c.childrenIndex) << (nodesBitsTextLength + nodesBitsTextOffset + nodesBitsICANN)
		fmt.Fprintf(w, "0x%08x, // n0x%04x c0x%04x (%s)%s %s %s %s\n",
			encoding, c.nodesIndex, c.childrenIndex, s, wildcardStr(c.wildcard),
			nodeTypeStr(c.nodeType), icannStr(c.icann), c.label,
		)
	}
	return nil
}

func printNodeLabel(w io.Writer, n *node) error {
	for _, c := range n.children {
		fmt.Fprintf(w, "%q,\n", c.label)
	}
	return nil
}

func icannStr(icann bool) string {
	if icann {
		return "I"
	}
	return " "
}

func wildcardStr(wildcard bool) string {
	if wildcard {
		return "*"
	}
	return " "
}

// makeText combines all the strings in labelsList to form one giant string.
// If the crush flag is true, then overlapping strings will be merged: "arpa"
// and "parliament" could yield "arparliament".
func makeText() string {
	if !*crush {
		return strings.Join(labelsList, "")
	}

	beforeLength := 0
	for _, s := range labelsList {
		beforeLength += len(s)
	}

	// Make a copy of labelsList.
	ss := append(make([]string, 0, len(labelsList)), labelsList...)

	// Remove strings that are substrings of other strings.
	for changed := true; changed; {
		changed = false
		for i, s := range ss {
			if s == "" {
				continue
			}
			for j, t := range ss {
				if i != j && t != "" && strings.Contains(s, t) {
					changed = true
					ss[j] = ""
				}
			}
		}
	}

	// Remove the empty strings.
	sort.Strings(ss)
	for len(ss) > 0 && ss[0] == "" {
		ss = ss[1:]
	}

	// Join strings where one suffix matches another prefix.
	for {
		// Find best i, j, k such that ss[i][len-k:] == ss[j][:k],
		// maximizing overlap length k.
		besti := -1
		bestj := -1
		bestk := 0
		for i, s := range ss {
			if s == "" {
				continue
			}
			for j, t := range ss {
				if i == j {
					continue
				}
				for k := bestk + 1; k <= len(s) && k <= len(t); k++ {
					if s[len(s)-k:] == t[:k] {
						besti = i
						bestj = j
						bestk = k
					}
				}
			}
		}
		if bestk > 0 {
			if *v {
				fmt.Fprintf(os.Stderr, "%d-length overlap at (%4d,%4d) out of (%4d,%4d): %q and %q\n",
					bestk, besti, bestj, len(ss), len(ss), ss[besti], ss[bestj])
			}
			ss[besti] += ss[bestj][bestk:]
			ss[bestj] = ""
			continue
		}
		break
	}

	text := strings.Join(ss, "")
	if *v {
		fmt.Fprintf(os.Stderr, "crushed %d bytes to become %d bytes\n", beforeLength, len(text))
	}
	return text
}
