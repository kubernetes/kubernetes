// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	yaml "go.yaml.in/yaml/v3"
	"sigs.k8s.io/kustomize/kyaml/errors"
)

// PathMatcher returns all RNodes matching the path wrapped in a SequenceNode.
// Lists may have multiple elements matching the path, and each matching element
// is added to the return result.
// If Path points to a SequenceNode, the SequenceNode is wrapped in another SequenceNode
// If Path does not contain any lists, the result is still wrapped in a SequenceNode of len == 1
type PathMatcher struct {
	Kind string `yaml:"kind,omitempty"`

	// Path is a slice of parts leading to the RNode to lookup.
	// Each path part may be one of:
	// * FieldMatcher -- e.g. "spec"
	// * Map Key -- e.g. "app.k8s.io/version"
	// * List Entry -- e.g. "[name=nginx]" or "[=-jar]" or "0"
	//
	// Map Keys and Fields are equivalent.
	// See FieldMatcher for more on Fields and Map Keys.
	//
	// List Entries are specified as map entry to match [fieldName=fieldValue].
	// See Elem for more on List Entries.
	//
	// Examples:
	// * spec.template.spec.container with matching name: [name=nginx] -- match 'name': 'nginx'
	// * spec.template.spec.container.argument matching a value: [=-jar] -- match '-jar'
	Path []string `yaml:"path,omitempty"`

	// Matches is set by PathMatch to publish the matched element values for each node.
	// After running  PathMatcher.Filter, each node from the SequenceNode result may be
	// looked up in Matches to find the field values that were matched.
	Matches map[*Node][]string

	// StripComments may be set to remove the comments on the matching Nodes.
	// This is useful for if the nodes are to be printed in FlowStyle.
	StripComments bool

	// Create will cause missing path parts to be created as they are walked.
	//
	// * The leaf Node (final path) will be created with a Kind matching Create
	// * Intermediary Nodes will be created as either a MappingNodes or
	//   SequenceNodes as appropriate for each's Path location.
	// * Nodes identified by an index will only be created if the index indicates
	//   an append operation (i.e. index=len(list))
	Create yaml.Kind `yaml:"create,omitempty"`

	val        *RNode
	field      string
	matchRegex string
}

func (p *PathMatcher) stripComments(n *Node) {
	if n == nil {
		return
	}
	if p.StripComments {
		n.LineComment = ""
		n.HeadComment = ""
		n.FootComment = ""
		for i := range n.Content {
			p.stripComments(n.Content[i])
		}
	}
}

func (p *PathMatcher) Filter(rn *RNode) (*RNode, error) {
	val, err := p.filter(rn)
	if err != nil {
		return nil, err
	}
	p.stripComments(val.YNode())
	return val, err
}

func (p *PathMatcher) filter(rn *RNode) (*RNode, error) {
	p.Matches = map[*Node][]string{}

	if len(p.Path) == 0 {
		// return the element wrapped in a SequenceNode
		p.appendRNode("", rn)
		return p.val, nil
	}

	if IsIdxNumber(p.Path[0]) {
		return p.doIndexSeq(rn)
	}

	if IsListIndex(p.Path[0]) {
		// match seq elements
		return p.doSeq(rn)
	}

	if IsWildcard(p.Path[0]) {
		// match every elements (*)
		return p.doMatchEvery(rn)
	}
	// match a field
	return p.doField(rn)
}

func (p *PathMatcher) doMatchEvery(rn *RNode) (*RNode, error) {
	if err := rn.VisitElements(p.visitEveryElem); err != nil {
		return nil, err
	}

	return p.val, nil
}

func (p *PathMatcher) visitEveryElem(elem *RNode) error {
	fieldName := p.Path[0]
	// recurse on the matching element
	pm := &PathMatcher{Path: p.Path[1:], Create: p.Create}
	add, err := pm.filter(elem)
	for k, v := range pm.Matches {
		p.Matches[k] = v
	}
	if err != nil || add == nil {
		return err
	}
	p.append(fieldName, add.Content()...)

	return nil
}

func (p *PathMatcher) doField(rn *RNode) (*RNode, error) {
	// lookup the field
	field, err := rn.Pipe(Get(p.Path[0]))
	if err != nil || (!IsCreate(p.Create) && field == nil) {
		return nil, err
	}

	if IsCreate(p.Create) && field == nil {
		var nextPart string
		if len(p.Path) > 1 {
			nextPart = p.Path[1]
		}
		nextPartKind := getPathPartKind(nextPart, p.Create)
		field = &RNode{value: &yaml.Node{Kind: nextPartKind}}
		err := rn.PipeE(SetField(p.Path[0], field))
		if err != nil {
			return nil, err
		}
	}

	// recurse on the field, removing the first element of the path
	pm := &PathMatcher{Path: p.Path[1:], Create: p.Create}
	p.val, err = pm.filter(field)
	p.Matches = pm.Matches
	return p.val, err
}

// doIndexSeq iterates over a sequence and appends elements matching the index p.Val
func (p *PathMatcher) doIndexSeq(rn *RNode) (*RNode, error) {
	// parse to index number
	idx, err := strconv.Atoi(p.Path[0])
	if err != nil {
		return nil, err
	}

	elements, err := rn.Elements()
	if err != nil {
		return nil, err
	}

	if len(elements) == idx && IsCreate(p.Create) {
		var nextPart string
		if len(p.Path) > 1 {
			nextPart = p.Path[1]
		}
		elem := &yaml.Node{Kind: getPathPartKind(nextPart, p.Create)}
		err = rn.PipeE(Append(elem))
		if err != nil {
			return nil, errors.WrapPrefixf(err, "failed to append element for %q", p.Path[0])
		}
		elements = append(elements, NewRNode(elem))
	}

	if len(elements) < idx+1 {
		return nil, fmt.Errorf("index %d specified but only %d elements found", idx, len(elements))
	}
	// get target element
	element := elements[idx]

	// recurse on the matching element
	pm := &PathMatcher{Path: p.Path[1:], Create: p.Create}
	add, err := pm.filter(element)
	for k, v := range pm.Matches {
		p.Matches[k] = v
	}
	if err != nil || add == nil {
		return nil, err
	}
	p.append("", add.Content()...)
	return p.val, nil
}

// doSeq iterates over a sequence and appends elements matching the path regex to p.Val
func (p *PathMatcher) doSeq(rn *RNode) (*RNode, error) {
	// parse the field + match pair
	var err error
	p.field, p.matchRegex, err = SplitIndexNameValue(p.Path[0])
	if err != nil {
		return nil, err
	}

	primitiveElement := len(p.field) == 0
	if primitiveElement {
		err = rn.VisitElements(p.visitPrimitiveElem)
	} else {
		err = rn.VisitElements(p.visitElem)
	}
	if err != nil {
		return nil, err
	}
	if !p.val.IsNil() && len(p.val.YNode().Content) == 0 {
		p.val = nil
	}

	if !IsCreate(p.Create) || p.val != nil {
		return p.val, nil
	}

	var elem *yaml.Node
	valueNode := NewScalarRNode(p.matchRegex).YNode()
	if primitiveElement {
		elem = valueNode
	} else {
		elem = &yaml.Node{
			Kind:    yaml.MappingNode,
			Content: []*yaml.Node{{Kind: yaml.ScalarNode, Value: p.field}, valueNode},
		}
	}
	err = rn.PipeE(Append(elem))
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to create element for %q", p.Path[0])
	}
	// re-do the sequence search; this time we'll find the element we just created
	return p.doSeq(rn)
}

func (p *PathMatcher) visitPrimitiveElem(elem *RNode) error {
	r, err := regexp.Compile(p.matchRegex)
	if err != nil {
		return err
	}

	str, err := elem.String()
	if err != nil {
		return err
	}
	str = strings.TrimSpace(str)
	if !r.MatchString(str) {
		return nil
	}

	p.appendRNode("", elem)
	return nil
}

func (p *PathMatcher) visitElem(elem *RNode) error {
	r, err := regexp.Compile(p.matchRegex)
	if err != nil {
		return err
	}

	// check if this elements field matches the regex
	val := elem.Field(p.field)
	if val == nil || val.Value == nil {
		return nil
	}
	str, err := val.Value.String()
	if err != nil {
		return err
	}
	str = strings.TrimSpace(str)
	if !r.MatchString(str) {
		return nil
	}

	// recurse on the matching element
	pm := &PathMatcher{Path: p.Path[1:], Create: p.Create}
	add, err := pm.filter(elem)
	for k, v := range pm.Matches {
		p.Matches[k] = v
	}
	if err != nil || add == nil {
		return err
	}
	p.append(str, add.Content()...)
	return nil
}

func (p *PathMatcher) appendRNode(path string, node *RNode) {
	p.append(path, node.YNode())
}

func (p *PathMatcher) append(path string, nodes ...*Node) {
	if p.val == nil {
		p.val = NewRNode(&Node{Kind: SequenceNode})
	}
	for i := range nodes {
		node := nodes[i]
		p.val.YNode().Content = append(p.val.YNode().Content, node)
		// record the path if specified
		if path != "" {
			p.Matches[node] = append(p.Matches[node], path)
		}
	}
}

func cleanPath(path []string) []string {
	var p []string
	for _, elem := range path {
		elem = strings.TrimSpace(elem)
		if len(elem) == 0 {
			continue
		}
		p = append(p, elem)
	}
	return p
}
