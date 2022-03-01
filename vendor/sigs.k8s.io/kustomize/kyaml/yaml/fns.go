// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/davecgh/go-spew/spew"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/internal/forked/github.com/go-yaml/yaml"
)

// Append creates an ElementAppender
func Append(elements ...*yaml.Node) ElementAppender {
	return ElementAppender{Elements: elements}
}

// ElementAppender adds all element to a SequenceNode's Content.
// Returns Elements[0] if len(Elements) == 1, otherwise returns nil.
type ElementAppender struct {
	Kind string `yaml:"kind,omitempty"`

	// Elem is the value to append.
	Elements []*yaml.Node `yaml:"elements,omitempty"`
}

func (a ElementAppender) Filter(rn *RNode) (*RNode, error) {
	if err := ErrorIfInvalid(rn, yaml.SequenceNode); err != nil {
		return nil, err
	}
	for i := range a.Elements {
		rn.YNode().Content = append(rn.Content(), a.Elements[i])
	}
	if len(a.Elements) == 1 {
		return NewRNode(a.Elements[0]), nil
	}
	return nil, nil
}

// ElementSetter sets the value for an Element in an associative list.
// ElementSetter will append, replace or delete an element in an associative list.
// To append, user a key-value pair that doesn't exist in the sequence. this
// behavior is intended to handle the case that not matching element found. It's
// not designed for this purpose. To append an element, please use ElementAppender.
// To replace, set the key-value pair and a non-nil Element.
// To delete, set the key-value pair and leave the Element as nil.
// Every key must have a corresponding value.
type ElementSetter struct {
	Kind string `yaml:"kind,omitempty"`

	// Element is the new value to set -- remove the existing element if nil
	Element *Node

	// Key is a list of fields on the elements. It is used to find matching elements to
	// update / delete
	Keys []string

	// Value is a list of field values on the elements corresponding to the keys. It is
	// used to find matching elements to update / delete.
	Values []string
}

// isMappingNode returns whether node is a mapping node
func (e ElementSetter) isMappingNode(node *RNode) bool {
	return ErrorIfInvalid(node, yaml.MappingNode) == nil
}

// isMappingSetter returns is this setter intended to set a mapping node
func (e ElementSetter) isMappingSetter() bool {
	return len(e.Keys) > 0 && e.Keys[0] != "" &&
		len(e.Values) > 0 && e.Values[0] != ""
}

func (e ElementSetter) Filter(rn *RNode) (*RNode, error) {
	if len(e.Keys) == 0 {
		e.Keys = append(e.Keys, "")
	}

	if err := ErrorIfInvalid(rn, SequenceNode); err != nil {
		return nil, err
	}

	// build the new Content slice
	var newContent []*yaml.Node
	matchingElementFound := false
	for i := range rn.YNode().Content {
		elem := rn.Content()[i]
		newNode := NewRNode(elem)

		// empty elements are not valid -- they at least need an associative key
		if IsMissingOrNull(newNode) || IsEmptyMap(newNode) {
			continue
		}
		// keep non-mapping node in the Content when we want to match a mapping.
		if !e.isMappingNode(newNode) && e.isMappingSetter() {
			newContent = append(newContent, elem)
			continue
		}

		// check if this is the element we are matching
		var val *RNode
		var err error
		found := true
		for j := range e.Keys {
			if j < len(e.Values) {
				val, err = newNode.Pipe(FieldMatcher{Name: e.Keys[j], StringValue: e.Values[j]})
			}
			if err != nil {
				return nil, err
			}
			if val == nil {
				found = false
				break
			}
		}
		if !found {
			// not the element we are looking for, keep it in the Content
			if len(e.Values) > 0 {
				newContent = append(newContent, elem)
			}
			continue
		}
		matchingElementFound = true

		// deletion operation -- remove the element from the new Content
		if e.Element == nil {
			continue
		}
		// replace operation -- replace the element in the Content
		newContent = append(newContent, e.Element)
	}
	rn.YNode().Content = newContent

	// deletion operation -- return nil
	if IsMissingOrNull(NewRNode(e.Element)) {
		return nil, nil
	}

	// append operation -- add the element to the Content
	if !matchingElementFound {
		rn.YNode().Content = append(rn.YNode().Content, e.Element)
	}

	return NewRNode(e.Element), nil
}

// GetElementByIndex will return a Filter which can be applied to a sequence
// node to get the element specified by the index
func GetElementByIndex(index int) ElementIndexer {
	return ElementIndexer{Index: index}
}

// ElementIndexer picks the element with a specified index. Index starts from
// 0 to len(list) - 1. a hyphen ("-") means the last index.
type ElementIndexer struct {
	Index int
}

// Filter implements Filter
func (i ElementIndexer) Filter(rn *RNode) (*RNode, error) {
	// rn.Elements will return error if rn is not a sequence node.
	elems, err := rn.Elements()
	if err != nil {
		return nil, err
	}
	if i.Index < 0 {
		return elems[len(elems)-1], nil
	}
	if i.Index >= len(elems) {
		return nil, nil
	}
	return elems[i.Index], nil
}

// Clear returns a FieldClearer
func Clear(name string) FieldClearer {
	return FieldClearer{Name: name}
}

// FieldClearer removes the field or map key.
// Returns a RNode with the removed field or map entry.
type FieldClearer struct {
	Kind string `yaml:"kind,omitempty"`

	// Name is the name of the field or key in the map.
	Name string `yaml:"name,omitempty"`

	IfEmpty bool `yaml:"ifEmpty,omitempty"`
}

func (c FieldClearer) Filter(rn *RNode) (*RNode, error) {
	if err := ErrorIfInvalid(rn, yaml.MappingNode); err != nil {
		return nil, err
	}

	for i := 0; i < len(rn.Content()); i += 2 {
		// if name matches, remove these 2 elements from the list because
		// they are treated as a fieldName/fieldValue pair.
		if rn.Content()[i].Value == c.Name {
			if c.IfEmpty {
				if len(rn.Content()[i+1].Content) > 0 {
					continue
				}
			}

			// save the item we are about to remove
			removed := NewRNode(rn.Content()[i+1])
			if len(rn.YNode().Content) > i+2 {
				l := len(rn.YNode().Content)
				// remove from the middle of the list
				rn.YNode().Content = rn.Content()[:i]
				rn.YNode().Content = append(
					rn.YNode().Content,
					rn.Content()[i+2:l]...)
			} else {
				// remove from the end of the list
				rn.YNode().Content = rn.Content()[:i]
			}

			// return the removed field name and value
			return removed, nil
		}
	}
	// nothing removed
	return nil, nil
}

func MatchElement(field, value string) ElementMatcher {
	return ElementMatcher{Keys: []string{field}, Values: []string{value}}
}

func MatchElementList(keys []string, values []string) ElementMatcher {
	return ElementMatcher{Keys: keys, Values: values}
}

func GetElementByKey(key string) ElementMatcher {
	return ElementMatcher{Keys: []string{key}, MatchAnyValue: true}
}

// ElementMatcher returns the first element from a Sequence matching the
// specified key-value pairs. If there's no match, and no configuration error,
// the matcher returns nil, nil.
type ElementMatcher struct {
	Kind string `yaml:"kind,omitempty"`

	// Keys are the list of fields upon which to match this element.
	Keys []string

	// Values are the list of values upon which to match this element.
	Values []string

	// Create will create the Element if it is not found
	Create *RNode `yaml:"create,omitempty"`

	// MatchAnyValue indicates that matcher should only consider the key and ignore
	// the actual value in the list. Values must be empty when MatchAnyValue is
	// set to true.
	MatchAnyValue bool `yaml:"noValue,omitempty"`
}

func (e ElementMatcher) Filter(rn *RNode) (*RNode, error) {
	if len(e.Keys) == 0 {
		e.Keys = append(e.Keys, "")
	}
	if len(e.Values) == 0 {
		e.Values = append(e.Values, "")
	}

	if err := ErrorIfInvalid(rn, yaml.SequenceNode); err != nil {
		return nil, err
	}
	if e.MatchAnyValue && len(e.Values) != 0 && e.Values[0] != "" {
		return nil, fmt.Errorf("Values must be empty when MatchAnyValue is set to true")
	}

	// SequenceNode Content is a slice of ScalarNodes.  Each ScalarNode has a
	// YNode containing the primitive data.
	if len(e.Keys) == 0 || len(e.Keys[0]) == 0 {
		for i := range rn.Content() {
			if rn.Content()[i].Value == e.Values[0] {
				return &RNode{value: rn.Content()[i]}, nil
			}
		}
		if e.Create != nil {
			return rn.Pipe(Append(e.Create.YNode()))
		}
		return nil, nil
	}

	// SequenceNode Content is a slice of MappingNodes.  Each MappingNode has Content
	// with a slice of key-value pairs containing the fields.
	for i := range rn.Content() {
		// cast the entry to a RNode so we can operate on it
		elem := NewRNode(rn.Content()[i])
		var field *RNode
		var err error

		// only check mapping node
		if err = ErrorIfInvalid(elem, yaml.MappingNode); err != nil {
			continue
		}

		if !e.MatchAnyValue && len(e.Keys) != len(e.Values) {
			return nil, fmt.Errorf("length of keys must equal length of values when MatchAnyValue is false")
		}

		matchesElement := true
		for i := range e.Keys {
			if e.MatchAnyValue {
				field, err = elem.Pipe(Get(e.Keys[i]))
			} else {
				field, err = elem.Pipe(MatchField(e.Keys[i], e.Values[i]))
			}
			if !IsFoundOrError(field, err) {
				// this is not the element we are looking for
				matchesElement = false
				break
			}
		}
		if matchesElement {
			return elem, err
		}
	}

	// create the element
	if e.Create != nil {
		return rn.Pipe(Append(e.Create.YNode()))
	}

	return nil, nil
}

func Get(name string) FieldMatcher {
	return FieldMatcher{Name: name}
}

func MatchField(name, value string) FieldMatcher {
	return FieldMatcher{Name: name, Value: NewScalarRNode(value)}
}

func Match(value string) FieldMatcher {
	return FieldMatcher{Value: NewScalarRNode(value)}
}

// FieldMatcher returns the value of a named field or map entry.
type FieldMatcher struct {
	Kind string `yaml:"kind,omitempty"`

	// Name of the field to return
	Name string `yaml:"name,omitempty"`

	// YNode of the field to return.
	// Optional.  Will only need to match field name if unset.
	Value *RNode `yaml:"value,omitempty"`

	StringValue string `yaml:"stringValue,omitempty"`

	StringRegexValue string `yaml:"stringRegexValue,omitempty"`

	// Create will cause the field to be created with this value
	// if it is set.
	Create *RNode `yaml:"create,omitempty"`
}

func (f FieldMatcher) Filter(rn *RNode) (*RNode, error) {
	if f.StringValue != "" && f.Value == nil {
		f.Value = NewScalarRNode(f.StringValue)
	}

	// never match nil or null fields
	if IsMissingOrNull(rn) {
		return nil, nil
	}

	if f.Name == "" {
		if err := ErrorIfInvalid(rn, yaml.ScalarNode); err != nil {
			return nil, err
		}
		switch {
		case f.StringRegexValue != "":
			// TODO(pwittrock): pre-compile this when unmarshalling and cache to a field
			rg, err := regexp.Compile(f.StringRegexValue)
			if err != nil {
				return nil, err
			}
			if match := rg.MatchString(rn.value.Value); match {
				return rn, nil
			}
			return nil, nil
		case GetValue(rn) == GetValue(f.Value):
			return rn, nil
		default:
			return nil, nil
		}
	}

	if err := ErrorIfInvalid(rn, yaml.MappingNode); err != nil {
		return nil, err
	}

	for i := 0; i < len(rn.Content()); i = IncrementFieldIndex(i) {
		isMatchingField := rn.Content()[i].Value == f.Name
		if isMatchingField {
			requireMatchFieldValue := f.Value != nil
			if !requireMatchFieldValue || rn.Content()[i+1].Value == f.Value.YNode().Value {
				return NewRNode(rn.Content()[i+1]), nil
			}
		}
	}

	if f.Create != nil {
		return rn.Pipe(SetField(f.Name, f.Create))
	}

	return nil, nil
}

// Lookup returns a PathGetter to lookup a field by its path.
func Lookup(path ...string) PathGetter {
	return PathGetter{Path: path}
}

// LookupCreate returns a PathGetter to lookup a field by its path and create it if it doesn't already
// exist.
func LookupCreate(kind yaml.Kind, path ...string) PathGetter {
	return PathGetter{Path: path, Create: kind}
}

// ConventionalContainerPaths is a list of paths at which containers typically appear in workload APIs.
// It is intended for use with LookupFirstMatch.
var ConventionalContainerPaths = [][]string{
	// e.g. Deployment, ReplicaSet, DaemonSet, Job, StatefulSet
	{"spec", "template", "spec", "containers"},
	// e.g. CronJob
	{"spec", "jobTemplate", "spec", "template", "spec", "containers"},
	// e.g. Pod
	{"spec", "containers"},
	// e.g. PodTemplate
	{"template", "spec", "containers"},
}

// LookupFirstMatch returns a Filter for locating a value that may exist at one of several possible paths.
// For example, it can be used with ConventionalContainerPaths to find the containers field in a standard workload resource.
// If more than one of the paths exists in the resource, the first will be returned. If none exist,
// nil will be returned. If an error is encountered during lookup, it will be returned.
func LookupFirstMatch(paths [][]string) Filter {
	return FilterFunc(func(object *RNode) (*RNode, error) {
		var result *RNode
		var err error
		for _, path := range paths {
			result, err = object.Pipe(PathGetter{Path: path})
			if err != nil {
				return nil, errors.Wrap(err)
			}
			if result != nil {
				return result, nil
			}
		}
		return nil, nil
	})
}

// PathGetter returns the RNode under Path.
type PathGetter struct {
	Kind string `yaml:"kind,omitempty"`

	// Path is a slice of parts leading to the RNode to lookup.
	// Each path part may be one of:
	// * FieldMatcher -- e.g. "spec"
	// * Map Key -- e.g. "app.k8s.io/version"
	// * List Entry -- e.g. "[name=nginx]" or "[=-jar]" or "0" or "-"
	//
	// Map Keys and Fields are equivalent.
	// See FieldMatcher for more on Fields and Map Keys.
	//
	// List Entries can be specified as map entry to match [fieldName=fieldValue]
	// or a positional index like 0 to get the element. - (unquoted hyphen) is
	// special and means the last element.
	//
	// See Elem for more on List Entries.
	//
	// Examples:
	// * spec.template.spec.container with matching name: [name=nginx]
	// * spec.template.spec.container.argument matching a value: [=-jar]
	Path []string `yaml:"path,omitempty"`

	// Create will cause missing path parts to be created as they are walked.
	//
	// * The leaf Node (final path) will be created with a Kind matching Create
	// * Intermediary Nodes will be created as either a MappingNodes or
	//   SequenceNodes as appropriate for each's Path location.
	// * If a list item is specified by a index (an offset or "-"), this item will
	//   not be created even Create is set.
	Create yaml.Kind `yaml:"create,omitempty"`

	// Style is the style to apply to created value Nodes.
	// Created key Nodes keep an unspecified Style.
	Style yaml.Style `yaml:"style,omitempty"`
}

func (l PathGetter) Filter(rn *RNode) (*RNode, error) {
	var err error
	fieldPath := append([]string{}, rn.FieldPath()...)
	match := rn

	// iterate over path until encountering an error or missing value
	l.Path = cleanPath(l.Path)
	for i := range l.Path {
		var part, nextPart string
		part = l.Path[i]
		if len(l.Path) > i+1 {
			nextPart = l.Path[i+1]
		}
		var fltr Filter
		fltr, err = l.getFilter(part, nextPart, &fieldPath)
		if err != nil {
			return nil, err
		}
		match, err = match.Pipe(fltr)
		if IsMissingOrError(match, err) {
			return nil, err
		}
		match.AppendToFieldPath(fieldPath...)
	}
	return match, nil
}

func (l PathGetter) getFilter(part, nextPart string, fieldPath *[]string) (Filter, error) {
	idx, err := strconv.Atoi(part)
	switch {
	case err == nil:
		// part is a number
		if idx < 0 {
			return nil, fmt.Errorf("array index %d cannot be negative", idx)
		}
		return GetElementByIndex(idx), nil
	case part == "-":
		// part is a hyphen
		return GetElementByIndex(-1), nil
	case IsListIndex(part):
		// part is surrounded by brackets
		return l.elemFilter(part)
	default:
		// mapping node
		*fieldPath = append(*fieldPath, part)
		return l.fieldFilter(part, l.getKind(nextPart))
	}
}

func (l PathGetter) elemFilter(part string) (Filter, error) {
	var match *RNode
	name, value, err := SplitIndexNameValue(part)
	if err != nil {
		return nil, errors.Wrap(err)
	}
	if !IsCreate(l.Create) {
		return MatchElement(name, value), nil
	}

	var elem *RNode
	primitiveElement := len(name) == 0
	if primitiveElement {
		// append a ScalarNode
		elem = NewScalarRNode(value)
		elem.YNode().Style = l.Style
		match = elem
	} else {
		// append a MappingNode
		match = NewRNode(&yaml.Node{Kind: yaml.ScalarNode, Value: value, Style: l.Style})
		elem = NewRNode(&yaml.Node{
			Kind:    yaml.MappingNode,
			Content: []*yaml.Node{{Kind: yaml.ScalarNode, Value: name}, match.YNode()},
			Style:   l.Style,
		})
	}
	// Append the Node
	return ElementMatcher{Keys: []string{name}, Values: []string{value}, Create: elem}, nil
}

func (l PathGetter) fieldFilter(
	name string, kind yaml.Kind) (Filter, error) {
	if !IsCreate(l.Create) {
		return Get(name), nil
	}
	return FieldMatcher{Name: name, Create: &RNode{value: &yaml.Node{Kind: kind, Style: l.Style}}}, nil
}

func (l PathGetter) getKind(nextPart string) yaml.Kind {
	if IsListIndex(nextPart) {
		// if nextPart is of the form [a=b], then it is an index into a Sequence
		// so the current part must be a SequenceNode
		return yaml.SequenceNode
	}
	if nextPart == "" {
		// final name in the path, use the l.Create defined Kind
		return l.Create
	}

	// non-sequence intermediate Node
	return yaml.MappingNode
}

func SetField(name string, value *RNode) FieldSetter {
	return FieldSetter{Name: name, Value: value}
}

func Set(value *RNode) FieldSetter {
	return FieldSetter{Value: value}
}

// FieldSetter sets a field or map entry to a value.
type FieldSetter struct {
	Kind string `yaml:"kind,omitempty"`

	// Name is the name of the field or key to lookup in a MappingNode.
	// If Name is unspecified, and the input is a ScalarNode, FieldSetter will set the
	// value on the ScalarNode.
	Name string `yaml:"name,omitempty"`

	// Comments for the field
	Comments Comments `yaml:"comments,omitempty"`

	// Value is the value to set.
	// Optional if Kind is set.
	Value *RNode `yaml:"value,omitempty"`

	StringValue string `yaml:"stringValue,omitempty"`

	// OverrideStyle can be set to override the style of the existing node
	// when setting it.  Otherwise, if an existing node is found, the style is
	// retained.
	OverrideStyle bool `yaml:"overrideStyle,omitempty"`
}

func (s FieldSetter) Filter(rn *RNode) (*RNode, error) {
	if s.StringValue != "" && s.Value == nil {
		s.Value = NewScalarRNode(s.StringValue)
	}

	if s.Name == "" {
		if err := ErrorIfInvalid(rn, yaml.ScalarNode); err != nil {
			return rn, err
		}
		if IsMissingOrNull(s.Value) {
			return rn, nil
		}
		// only apply the style if there is not an existing style
		// or we want to override it
		if !s.OverrideStyle || s.Value.YNode().Style == 0 {
			// keep the original style if it exists
			s.Value.YNode().Style = rn.YNode().Style
		}
		rn.SetYNode(s.Value.YNode())
		return rn, nil
	}

	// Clear the field if it is empty, or explicitly null
	if s.Value == nil || s.Value.IsTaggedNull() {
		return rn.Pipe(Clear(s.Name))
	}

	field, err := rn.Pipe(FieldMatcher{Name: s.Name})
	if err != nil {
		return nil, err
	}
	if field != nil {
		// only apply the style if there is not an existing style
		// or we want to override it
		if !s.OverrideStyle || field.YNode().Style == 0 {
			// keep the original style if it exists
			s.Value.YNode().Style = field.YNode().Style
		}
		// need to def ref the Node since field is ephemeral
		field.SetYNode(s.Value.YNode())
		return field, nil
	}

	// create the field
	rn.YNode().Content = append(
		rn.YNode().Content,
		&yaml.Node{
			Kind:        yaml.ScalarNode,
			Value:       s.Name,
			HeadComment: s.Comments.HeadComment,
			LineComment: s.Comments.LineComment,
			FootComment: s.Comments.FootComment,
		},
		s.Value.YNode())
	return s.Value, nil
}

// Tee calls the provided Filters, and returns its argument rather than the result
// of the filters.
// May be used to fork sub-filters from a call.
// e.g. locate field, set value; locate another field, set another value
func Tee(filters ...Filter) Filter {
	return TeePiper{Filters: filters}
}

// TeePiper Calls a slice of Filters and returns its input.
// May be used to fork sub-filters from a call.
// e.g. locate field, set value; locate another field, set another value
type TeePiper struct {
	Kind string `yaml:"kind,omitempty"`

	// Filters are the set of Filters run by TeePiper.
	Filters []Filter `yaml:"filters,omitempty"`
}

func (t TeePiper) Filter(rn *RNode) (*RNode, error) {
	_, err := rn.Pipe(t.Filters...)
	return rn, err
}

// IsCreate returns true if kind is specified
func IsCreate(kind yaml.Kind) bool {
	return kind != 0
}

// IsMissingOrError returns true if rn is NOT found or err is non-nil
func IsMissingOrError(rn *RNode, err error) bool {
	return rn == nil || err != nil
}

// IsFoundOrError returns true if rn is found or err is non-nil
func IsFoundOrError(rn *RNode, err error) bool {
	return rn != nil || err != nil
}

func ErrorIfAnyInvalidAndNonNull(kind yaml.Kind, rn ...*RNode) error {
	for i := range rn {
		if IsMissingOrNull(rn[i]) {
			continue
		}
		if err := ErrorIfInvalid(rn[i], kind); err != nil {
			return err
		}
	}
	return nil
}

var nodeTypeIndex = map[yaml.Kind]string{
	yaml.SequenceNode: "SequenceNode",
	yaml.MappingNode:  "MappingNode",
	yaml.ScalarNode:   "ScalarNode",
	yaml.DocumentNode: "DocumentNode",
	yaml.AliasNode:    "AliasNode",
}

func ErrorIfInvalid(rn *RNode, kind yaml.Kind) error {
	if IsMissingOrNull(rn) {
		// node has no type, pass validation
		return nil
	}

	if rn.YNode().Kind != kind {
		s, _ := rn.String()
		return errors.Errorf(
			"wrong Node Kind for %s expected: %v was %v: value: {%s}",
			strings.Join(rn.FieldPath(), "."),
			nodeTypeIndex[kind], nodeTypeIndex[rn.YNode().Kind], strings.TrimSpace(s))
	}

	if kind == yaml.MappingNode {
		if len(rn.YNode().Content)%2 != 0 {
			return errors.Errorf(
				"yaml MappingNodes must have even length contents: %v", spew.Sdump(rn))
		}
	}

	return nil
}

// IsListIndex returns true if p is an index into a Val.
// e.g. [fieldName=fieldValue]
// e.g. [=primitiveValue]
func IsListIndex(p string) bool {
	return strings.HasPrefix(p, "[") && strings.HasSuffix(p, "]")
}

// SplitIndexNameValue splits a lookup part Val index into the field name
// and field value to match.
// e.g. splits [name=nginx] into (name, nginx)
// e.g. splits [=-jar] into ("", -jar)
func SplitIndexNameValue(p string) (string, string, error) {
	elem := strings.TrimSuffix(p, "]")
	elem = strings.TrimPrefix(elem, "[")
	parts := strings.SplitN(elem, "=", 2)
	if len(parts) == 1 {
		return "", "", fmt.Errorf("list path element must contain fieldName=fieldValue for element to match")
	}
	return parts[0], parts[1], nil
}

// IncrementFieldIndex increments i to point to the next field name element in
// a slice of Contents.
func IncrementFieldIndex(i int) int {
	return i + 2
}
