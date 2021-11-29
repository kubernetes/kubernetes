// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"regexp"
	"strconv"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/internal/forked/github.com/go-yaml/yaml"
	"sigs.k8s.io/kustomize/kyaml/sliceutil"
	"sigs.k8s.io/kustomize/kyaml/yaml/internal/k8sgen/pkg/labels"
)

// MakeNullNode returns an RNode that represents an empty document.
func MakeNullNode() *RNode {
	return NewRNode(&Node{Tag: NodeTagNull})
}

// IsMissingOrNull is true if the RNode is nil or explicitly tagged null.
// TODO: make this a method on RNode.
func IsMissingOrNull(node *RNode) bool {
	return node.IsNil() || node.YNode().Tag == NodeTagNull
}

// IsEmptyMap returns true if the RNode is an empty node or an empty map.
// TODO: make this a method on RNode.
func IsEmptyMap(node *RNode) bool {
	return IsMissingOrNull(node) || IsYNodeEmptyMap(node.YNode())
}

// GetValue returns underlying yaml.Node Value field
func GetValue(node *RNode) string {
	if IsMissingOrNull(node) {
		return ""
	}
	return node.YNode().Value
}

// Parse parses a yaml string into an *RNode.
// To parse multiple resources, consider a kio.ByteReader
func Parse(value string) (*RNode, error) {
	return Parser{Value: value}.Filter(nil)
}

// ReadFile parses a single Resource from a yaml file.
// To parse multiple resources, consider a kio.ByteReader
func ReadFile(path string) (*RNode, error) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(string(b))
}

// WriteFile writes a single Resource to a yaml file
func WriteFile(node *RNode, path string) error {
	out, err := node.String()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, []byte(out), 0600)
}

// UpdateFile reads the file at path, applies the filter to it, and write the result back.
// path must contain a exactly 1 resource (YAML).
func UpdateFile(filter Filter, path string) error {
	// Read the yaml
	y, err := ReadFile(path)
	if err != nil {
		return err
	}

	// Update the yaml
	if err := y.PipeE(filter); err != nil {
		return err
	}

	// Write the yaml
	return WriteFile(y, path)
}

// MustParse parses a yaml string into an *RNode and panics if there is an error
func MustParse(value string) *RNode {
	v, err := Parser{Value: value}.Filter(nil)
	if err != nil {
		panic(err)
	}
	return v
}

// NewScalarRNode returns a new Scalar *RNode containing the provided scalar value.
func NewScalarRNode(value string) *RNode {
	return &RNode{
		value: &yaml.Node{
			Kind:  yaml.ScalarNode,
			Value: value,
		}}
}

// NewStringRNode returns a new Scalar *RNode containing the provided string.
// If the string is non-utf8, it will be base64 encoded, and the tag
// will indicate binary data.
func NewStringRNode(value string) *RNode {
	n := yaml.Node{Kind: yaml.ScalarNode}
	n.SetString(value)
	return NewRNode(&n)
}

// NewListRNode returns a new List *RNode containing the provided scalar values.
func NewListRNode(values ...string) *RNode {
	seq := &RNode{value: &yaml.Node{Kind: yaml.SequenceNode}}
	for _, v := range values {
		seq.value.Content = append(seq.value.Content, &yaml.Node{
			Kind:  yaml.ScalarNode,
			Value: v,
		})
	}
	return seq
}

// NewMapRNode returns a new Map *RNode containing the provided values
func NewMapRNode(values *map[string]string) *RNode {
	m := &RNode{value: &yaml.Node{
		Kind: yaml.MappingNode,
	}}
	if values == nil {
		return m
	}

	for k, v := range *values {
		m.value.Content = append(m.value.Content, &yaml.Node{
			Kind:  yaml.ScalarNode,
			Value: k,
		}, &yaml.Node{
			Kind:  yaml.ScalarNode,
			Value: v,
		})
	}

	return m
}

// SyncMapNodesOrder sorts the map node keys in 'to' node to match the order of
// map node keys in 'from' node, additional keys are moved to the end
func SyncMapNodesOrder(from, to *RNode) {
	to.Copy()
	res := &RNode{value: &yaml.Node{
		Kind:        to.YNode().Kind,
		Style:       to.YNode().Style,
		Tag:         to.YNode().Tag,
		Anchor:      to.YNode().Anchor,
		Alias:       to.YNode().Alias,
		HeadComment: to.YNode().HeadComment,
		LineComment: to.YNode().LineComment,
		FootComment: to.YNode().FootComment,
		Line:        to.YNode().Line,
		Column:      to.YNode().Column,
	}}

	fromFieldNames, err := from.Fields()
	if err != nil {
		return
	}

	toFieldNames, err := to.Fields()
	if err != nil {
		return
	}

	for _, fieldName := range fromFieldNames {
		if !sliceutil.Contains(toFieldNames, fieldName) {
			continue
		}
		// append the common nodes in the order defined in 'from' node
		res.value.Content = append(res.value.Content, to.Field(fieldName).Key.YNode(), to.Field(fieldName).Value.YNode())
		toFieldNames = sliceutil.Remove(toFieldNames, fieldName)
	}

	for _, fieldName := range toFieldNames {
		// append the residual nodes which are not present in 'from' node
		res.value.Content = append(res.value.Content, to.Field(fieldName).Key.YNode(), to.Field(fieldName).Value.YNode())
	}

	to.SetYNode(res.YNode())
}

// NewRNode returns a new RNode pointer containing the provided Node.
func NewRNode(value *yaml.Node) *RNode {
	return &RNode{value: value}
}

// RNode provides functions for manipulating Kubernetes Resources
// Objects unmarshalled into *yaml.Nodes
type RNode struct {
	// fieldPath contains the path from the root of the KubernetesObject to
	// this field.
	// Only field names are captured in the path.
	// e.g. a image field in a Deployment would be
	// 'spec.template.spec.containers.image'
	fieldPath []string

	// FieldValue contains the value.
	// FieldValue is always set:
	// field: field value
	// list entry: list entry value
	// object root: object root
	value *yaml.Node

	Match []string
}

// Copy returns a distinct copy.
func (rn *RNode) Copy() *RNode {
	if rn == nil {
		return nil
	}
	result := *rn
	result.value = CopyYNode(rn.value)
	return &result
}

var ErrMissingMetadata = fmt.Errorf("missing Resource metadata")

// IsNil is true if the node is nil, or its underlying YNode is nil.
func (rn *RNode) IsNil() bool {
	return rn == nil || rn.YNode() == nil
}

// IsTaggedNull is true if a non-nil node is explicitly tagged Null.
func (rn *RNode) IsTaggedNull() bool {
	return !rn.IsNil() && IsYNodeTaggedNull(rn.YNode())
}

// IsNilOrEmpty is true if the node is nil,
// has no YNode, or has YNode that appears empty.
func (rn *RNode) IsNilOrEmpty() bool {
	return rn.IsNil() ||
		IsYNodeTaggedNull(rn.YNode()) ||
		IsYNodeEmptyMap(rn.YNode()) ||
		IsYNodeEmptySeq(rn.YNode()) ||
		IsYNodeZero(rn.YNode())
}

// GetMeta returns the ResourceMeta for an RNode
func (rn *RNode) GetMeta() (ResourceMeta, error) {
	if IsMissingOrNull(rn) {
		return ResourceMeta{}, nil
	}
	missingMeta := true
	n := rn
	if n.YNode().Kind == DocumentNode {
		// get the content is this is the document node
		n = NewRNode(n.Content()[0])
	}

	// don't decode into the struct directly or it will fail on UTF-8 issues
	// which appear in comments
	m := ResourceMeta{}

	// TODO: consider optimizing this parsing
	if f := n.Field(APIVersionField); !f.IsNilOrEmpty() {
		m.APIVersion = GetValue(f.Value)
		missingMeta = false
	}
	if f := n.Field(KindField); !f.IsNilOrEmpty() {
		m.Kind = GetValue(f.Value)
		missingMeta = false
	}

	mf := n.Field(MetadataField)
	if mf.IsNilOrEmpty() {
		if missingMeta {
			return m, ErrMissingMetadata
		}
		return m, nil
	}
	meta := mf.Value

	if f := meta.Field(NameField); !f.IsNilOrEmpty() {
		m.Name = f.Value.YNode().Value
		missingMeta = false
	}
	if f := meta.Field(NamespaceField); !f.IsNilOrEmpty() {
		m.Namespace = GetValue(f.Value)
		missingMeta = false
	}

	if f := meta.Field(LabelsField); !f.IsNilOrEmpty() {
		m.Labels = map[string]string{}
		_ = f.Value.VisitFields(func(node *MapNode) error {
			m.Labels[GetValue(node.Key)] = GetValue(node.Value)
			return nil
		})
		missingMeta = false
	}
	if f := meta.Field(AnnotationsField); !f.IsNilOrEmpty() {
		m.Annotations = map[string]string{}
		_ = f.Value.VisitFields(func(node *MapNode) error {
			m.Annotations[GetValue(node.Key)] = GetValue(node.Value)
			return nil
		})
		missingMeta = false
	}

	if missingMeta {
		return m, ErrMissingMetadata
	}
	return m, nil
}

// Pipe sequentially invokes each Filter, and passes the result to the next
// Filter.
//
// Analogous to http://www.linfo.org/pipes.html
//
// * rn is provided as input to the first Filter.
// * if any Filter returns an error, immediately return the error
// * if any Filter returns a nil RNode, immediately return nil, nil
// * if all Filters succeed with non-empty results, return the final result
func (rn *RNode) Pipe(functions ...Filter) (*RNode, error) {
	// check if rn is nil to make chaining Pipe calls easier
	if rn == nil {
		return nil, nil
	}

	var v *RNode
	var err error
	if rn.value != nil && rn.value.Kind == yaml.DocumentNode {
		// the first node may be a DocumentNode containing a single MappingNode
		v = &RNode{value: rn.value.Content[0]}
	} else {
		v = rn
	}

	// return each fn in sequence until encountering an error or missing value
	for _, c := range functions {
		v, err = c.Filter(v)
		if err != nil || v == nil {
			return v, errors.Wrap(err)
		}
	}
	return v, err
}

// PipeE runs Pipe, dropping the *RNode return value.
// Useful for directly returning the Pipe error value from functions.
func (rn *RNode) PipeE(functions ...Filter) error {
	_, err := rn.Pipe(functions...)
	return errors.Wrap(err)
}

// Document returns the Node for the value.
func (rn *RNode) Document() *yaml.Node {
	return rn.value
}

// YNode returns the yaml.Node value.  If the yaml.Node value is a DocumentNode,
// YNode will return the DocumentNode Content entry instead of the DocumentNode.
func (rn *RNode) YNode() *yaml.Node {
	if rn == nil || rn.value == nil {
		return nil
	}
	if rn.value.Kind == yaml.DocumentNode {
		return rn.value.Content[0]
	}
	return rn.value
}

// SetYNode sets the yaml.Node value on an RNode.
func (rn *RNode) SetYNode(node *yaml.Node) {
	if rn.value == nil || node == nil {
		rn.value = node
		return
	}
	*rn.value = *node
}

// GetKind returns the kind, if it exists, else empty string.
func (rn *RNode) GetKind() string {
	if node := rn.getMapFieldValue(KindField); node != nil {
		return node.Value
	}
	return ""
}

// SetKind sets the kind.
func (rn *RNode) SetKind(k string) {
	rn.SetMapField(NewScalarRNode(k), KindField)
}

// GetApiVersion returns the apiversion, if it exists, else empty string.
func (rn *RNode) GetApiVersion() string {
	if node := rn.getMapFieldValue(APIVersionField); node != nil {
		return node.Value
	}
	return ""
}

// SetApiVersion sets the apiVersion.
func (rn *RNode) SetApiVersion(av string) {
	rn.SetMapField(NewScalarRNode(av), APIVersionField)
}

// getMapFieldValue returns the value (*yaml.Node) of a mapping field.
// The value might be nil.  Also, the function returns nil, not an error,
// if this node is not a mapping node, or if this node does not have the
// given field, so this function cannot be used to make distinctions
// between these cases.
func (rn *RNode) getMapFieldValue(field string) *yaml.Node {
	for i := 0; i < len(rn.Content()); i = IncrementFieldIndex(i) {
		if rn.Content()[i].Value == field {
			return rn.Content()[i+1]
		}
	}
	return nil
}

// GetName returns the name, or empty string if
// field not found.  The setter is more restrictive.
func (rn *RNode) GetName() string {
	return rn.getMetaStringField(NameField)
}

// getMetaStringField returns the value of a string field in metadata.
func (rn *RNode) getMetaStringField(fName string) string {
	md := rn.getMetaData()
	if md == nil {
		return ""
	}
	f := md.Field(fName)
	if f.IsNilOrEmpty() {
		return ""
	}
	return GetValue(f.Value)
}

// getMetaData returns the RNode holding the value of the metadata field.
// Return nil if field not found (no error).
func (rn *RNode) getMetaData() *RNode {
	if IsMissingOrNull(rn) {
		return nil
	}
	var n *RNode
	if rn.YNode().Kind == DocumentNode {
		// get the content if this is the document node
		n = NewRNode(rn.Content()[0])
	} else {
		n = rn
	}
	mf := n.Field(MetadataField)
	if mf.IsNilOrEmpty() {
		return nil
	}
	return mf.Value
}

// SetName sets the metadata name field.
func (rn *RNode) SetName(name string) error {
	return rn.SetMapField(NewScalarRNode(name), MetadataField, NameField)
}

// GetNamespace gets the metadata namespace field, or empty string if
// field not found.  The setter is more restrictive.
func (rn *RNode) GetNamespace() string {
	return rn.getMetaStringField(NamespaceField)
}

// SetNamespace tries to set the metadata namespace field.  If the argument
// is empty, the field is dropped.
func (rn *RNode) SetNamespace(ns string) error {
	meta, err := rn.Pipe(Lookup(MetadataField))
	if err != nil {
		return err
	}
	if ns == "" {
		if rn == nil {
			return nil
		}
		return meta.PipeE(Clear(NamespaceField))
	}
	return rn.SetMapField(
		NewScalarRNode(ns), MetadataField, NamespaceField)
}

// GetAnnotations gets the metadata annotations field.
// If the field is missing, returns an empty map.
// Use another method to check for missing metadata.
func (rn *RNode) GetAnnotations() map[string]string {
	meta := rn.getMetaData()
	if meta == nil {
		return make(map[string]string)
	}
	return rn.getMapFromMeta(meta, AnnotationsField)
}

// SetAnnotations tries to set the metadata annotations field.
func (rn *RNode) SetAnnotations(m map[string]string) error {
	return rn.setMapInMetadata(m, AnnotationsField)
}

// GetLabels gets the metadata labels field.
// If the field is missing, returns an empty map.
// Use another method to check for missing metadata.
func (rn *RNode) GetLabels() map[string]string {
	meta := rn.getMetaData()
	if meta == nil {
		return make(map[string]string)
	}
	return rn.getMapFromMeta(meta, LabelsField)
}

// getMapFromMeta returns map, sometimes empty, from metadata.
func (rn *RNode) getMapFromMeta(meta *RNode, fName string) map[string]string {
	result := make(map[string]string)
	if f := meta.Field(fName); !f.IsNilOrEmpty() {
		_ = f.Value.VisitFields(func(node *MapNode) error {
			result[GetValue(node.Key)] = GetValue(node.Value)
			return nil
		})
	}
	return result
}

// SetLabels sets the metadata labels field.
func (rn *RNode) SetLabels(m map[string]string) error {
	return rn.setMapInMetadata(m, LabelsField)
}

// This established proper quoting on string values, and sorts by key.
func (rn *RNode) setMapInMetadata(m map[string]string, field string) error {
	meta, err := rn.Pipe(LookupCreate(MappingNode, MetadataField))
	if err != nil {
		return err
	}
	if err = meta.PipeE(Clear(field)); err != nil {
		return err
	}
	if len(m) == 0 {
		return nil
	}
	mapNode, err := meta.Pipe(LookupCreate(MappingNode, field))
	if err != nil {
		return err
	}
	for _, k := range SortedMapKeys(m) {
		if _, err := mapNode.Pipe(
			SetField(k, NewStringRNode(m[k]))); err != nil {
			return err
		}
	}
	return nil
}

func (rn *RNode) SetMapField(value *RNode, path ...string) error {
	return rn.PipeE(
		LookupCreate(yaml.MappingNode, path[0:len(path)-1]...),
		SetField(path[len(path)-1], value),
	)
}

func (rn *RNode) GetDataMap() map[string]string {
	n, err := rn.Pipe(Lookup(DataField))
	if err != nil {
		return nil
	}
	result := map[string]string{}
	_ = n.VisitFields(func(node *MapNode) error {
		result[GetValue(node.Key)] = GetValue(node.Value)
		return nil
	})
	return result
}

func (rn *RNode) GetBinaryDataMap() map[string]string {
	n, err := rn.Pipe(Lookup(BinaryDataField))
	if err != nil {
		return nil
	}
	result := map[string]string{}
	_ = n.VisitFields(func(node *MapNode) error {
		result[GetValue(node.Key)] = GetValue(node.Value)
		return nil
	})
	return result
}

// GetValidatedDataMap retrieves the data map and returns an error if the data
// map contains entries which are not included in the expectedKeys set.
func (rn *RNode) GetValidatedDataMap(expectedKeys []string) (map[string]string, error) {
	dataMap := rn.GetDataMap()
	err := rn.validateDataMap(dataMap, expectedKeys)
	return dataMap, err
}

func (rn *RNode) validateDataMap(dataMap map[string]string, expectedKeys []string) error {
	if dataMap == nil {
		return fmt.Errorf("The datamap is unassigned")
	}
	for key := range dataMap {
		found := false
		for _, expected := range expectedKeys {
			if expected == key {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("an unexpected key (%v) was found", key)
		}
	}
	return nil
}

func (rn *RNode) SetDataMap(m map[string]string) {
	if rn == nil {
		log.Fatal("cannot set data map on nil Rnode")
	}
	if err := rn.PipeE(Clear(DataField)); err != nil {
		log.Fatal(err)
	}
	if len(m) == 0 {
		return
	}
	if err := rn.LoadMapIntoConfigMapData(m); err != nil {
		log.Fatal(err)
	}
}

func (rn *RNode) SetBinaryDataMap(m map[string]string) {
	if rn == nil {
		log.Fatal("cannot set binaryData map on nil Rnode")
	}
	if err := rn.PipeE(Clear(BinaryDataField)); err != nil {
		log.Fatal(err)
	}
	if len(m) == 0 {
		return
	}
	if err := rn.LoadMapIntoConfigMapBinaryData(m); err != nil {
		log.Fatal(err)
	}
}

// AppendToFieldPath appends a field name to the FieldPath.
func (rn *RNode) AppendToFieldPath(parts ...string) {
	rn.fieldPath = append(rn.fieldPath, parts...)
}

// FieldPath returns the field path from the Resource root node, to rn.
// Does not include list indexes.
func (rn *RNode) FieldPath() []string {
	return rn.fieldPath
}

// String returns string representation of the RNode
func (rn *RNode) String() (string, error) {
	if rn == nil {
		return "", nil
	}
	return String(rn.value)
}

// MustString returns string representation of the RNode or panics if there is an error
func (rn *RNode) MustString() string {
	s, err := rn.String()
	if err != nil {
		panic(err)
	}
	return s
}

// Content returns Node Content field.
func (rn *RNode) Content() []*yaml.Node {
	if rn == nil {
		return nil
	}
	return rn.YNode().Content
}

// Fields returns the list of field names for a MappingNode.
// Returns an error for non-MappingNodes.
func (rn *RNode) Fields() ([]string, error) {
	if err := ErrorIfInvalid(rn, yaml.MappingNode); err != nil {
		return nil, errors.Wrap(err)
	}
	var fields []string
	for i := 0; i < len(rn.Content()); i += 2 {
		fields = append(fields, rn.Content()[i].Value)
	}
	return fields, nil
}

// FieldRNodes returns the list of field key RNodes for a MappingNode.
// Returns an error for non-MappingNodes.
func (rn *RNode) FieldRNodes() ([]*RNode, error) {
	if err := ErrorIfInvalid(rn, yaml.MappingNode); err != nil {
		return nil, errors.Wrap(err)
	}
	var fields []*RNode
	for i := 0; i < len(rn.Content()); i += 2 {
		yNode := rn.Content()[i]
		// for each key node in the input mapping node contents create equivalent rNode
		rNode := &RNode{}
		rNode.SetYNode(yNode)
		fields = append(fields, rNode)
	}
	return fields, nil
}

// Field returns a fieldName, fieldValue pair for MappingNodes.
// Returns nil for non-MappingNodes.
func (rn *RNode) Field(field string) *MapNode {
	if rn.YNode().Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i < len(rn.Content()); i = IncrementFieldIndex(i) {
		isMatchingField := rn.Content()[i].Value == field
		if isMatchingField {
			return &MapNode{Key: NewRNode(rn.Content()[i]), Value: NewRNode(rn.Content()[i+1])}
		}
	}
	return nil
}

// VisitFields calls fn for each field in the RNode.
// Returns an error for non-MappingNodes.
func (rn *RNode) VisitFields(fn func(node *MapNode) error) error {
	// get the list of srcFieldNames
	srcFieldNames, err := rn.Fields()
	if err != nil {
		return errors.Wrap(err)
	}

	// visit each field
	for _, fieldName := range srcFieldNames {
		if err := fn(rn.Field(fieldName)); err != nil {
			return errors.Wrap(err)
		}
	}
	return nil
}

// Elements returns the list of elements in the RNode.
// Returns an error for non-SequenceNodes.
func (rn *RNode) Elements() ([]*RNode, error) {
	if err := ErrorIfInvalid(rn, yaml.SequenceNode); err != nil {
		return nil, errors.Wrap(err)
	}
	var elements []*RNode
	for i := 0; i < len(rn.Content()); i++ {
		elements = append(elements, NewRNode(rn.Content()[i]))
	}
	return elements, nil
}

// ElementValues returns a list of all observed values for a given field name
// in a list of elements.
// Returns error for non-SequenceNodes.
func (rn *RNode) ElementValues(key string) ([]string, error) {
	if err := ErrorIfInvalid(rn, yaml.SequenceNode); err != nil {
		return nil, errors.Wrap(err)
	}
	var elements []string
	for i := 0; i < len(rn.Content()); i++ {
		field := NewRNode(rn.Content()[i]).Field(key)
		if !field.IsNilOrEmpty() {
			elements = append(elements, field.Value.YNode().Value)
		}
	}
	return elements, nil
}

// ElementValuesList returns a list of lists, where each list is a set of
// values corresponding to each key in keys.
// Returns error for non-SequenceNodes.
func (rn *RNode) ElementValuesList(keys []string) ([][]string, error) {
	if err := ErrorIfInvalid(rn, yaml.SequenceNode); err != nil {
		return nil, errors.Wrap(err)
	}
	elements := make([][]string, len(rn.Content()))

	for i := 0; i < len(rn.Content()); i++ {
		for _, key := range keys {
			field := NewRNode(rn.Content()[i]).Field(key)
			if field.IsNilOrEmpty() {
				elements[i] = append(elements[i], "")
			} else {
				elements[i] = append(elements[i], field.Value.YNode().Value)
			}
		}
	}
	return elements, nil
}

// Element returns the element in the list which contains the field matching the value.
// Returns nil for non-SequenceNodes or if no Element matches.
func (rn *RNode) Element(key, value string) *RNode {
	if rn.YNode().Kind != yaml.SequenceNode {
		return nil
	}
	elem, err := rn.Pipe(MatchElement(key, value))
	if err != nil {
		return nil
	}
	return elem
}

// ElementList returns the element in the list in which all fields keys[i] matches all
// corresponding values[i].
// Returns nil for non-SequenceNodes or if no Element matches.
func (rn *RNode) ElementList(keys []string, values []string) *RNode {
	if rn.YNode().Kind != yaml.SequenceNode {
		return nil
	}
	elem, err := rn.Pipe(MatchElementList(keys, values))
	if err != nil {
		return nil
	}
	return elem
}

// VisitElements calls fn for each element in a SequenceNode.
// Returns an error for non-SequenceNodes
func (rn *RNode) VisitElements(fn func(node *RNode) error) error {
	elements, err := rn.Elements()
	if err != nil {
		return errors.Wrap(err)
	}

	for i := range elements {
		if err := fn(elements[i]); err != nil {
			return errors.Wrap(err)
		}
	}
	return nil
}

// AssociativeSequenceKeys is a map of paths to sequences that have associative keys.
// The order sets the precedence of the merge keys -- if multiple keys are present
// in Resources in a list, then the FIRST key which ALL elements in the list have is used as the
// associative key for merging that list.
// Only infer name as a merge key.
var AssociativeSequenceKeys = []string{"name"}

// IsAssociative returns true if the RNode contains an AssociativeSequenceKey as a field.
func (rn *RNode) IsAssociative() bool {
	return rn.GetAssociativeKey() != ""
}

// GetAssociativeKey returns the AssociativeSequenceKey used to merge the elements in the
// SequenceNode, or "" if the  list is not associative.
func (rn *RNode) GetAssociativeKey() string {
	// look for any associative keys in the first element
	for _, key := range AssociativeSequenceKeys {
		if checkKey(key, rn.Content()) {
			return key
		}
	}

	// element doesn't have an associative keys
	return ""
}

// MarshalJSON creates a byte slice from the RNode.
func (rn *RNode) MarshalJSON() ([]byte, error) {
	s, err := rn.String()
	if err != nil {
		return nil, err
	}

	if rn.YNode().Kind == SequenceNode {
		var a []interface{}
		if err := Unmarshal([]byte(s), &a); err != nil {
			return nil, err
		}
		return json.Marshal(a)
	}

	m := map[string]interface{}{}
	if err := Unmarshal([]byte(s), &m); err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

// UnmarshalJSON overwrites this RNode with data from []byte.
func (rn *RNode) UnmarshalJSON(b []byte) error {
	m := map[string]interface{}{}
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	r, err := FromMap(m)
	if err != nil {
		return err
	}
	rn.value = r.value
	return nil
}

// DeAnchor inflates all YAML aliases with their anchor values.
// All YAML anchor data is permanently removed (feel free to call Copy first).
func (rn *RNode) DeAnchor() (err error) {
	rn.value, err = deAnchor(rn.value)
	return
}

// deAnchor removes all AliasNodes from the yaml.Node's tree, replacing
// them with what they point to.  All Anchor fields (these are used to mark
// anchor definitions) are cleared.
func deAnchor(yn *yaml.Node) (res *yaml.Node, err error) {
	if yn == nil {
		return nil, nil
	}
	if yn.Anchor != "" {
		// This node defines an anchor. Clear the field so that it
		// doesn't show up when marshalling.
		if yn.Kind == yaml.AliasNode {
			// Maybe this is OK, but for now treating it as a bug.
			return nil, fmt.Errorf(
				"anchor %q defined using alias %v", yn.Anchor, yn.Alias)
		}
		yn.Anchor = ""
	}
	switch yn.Kind {
	case yaml.ScalarNode:
		return yn, nil
	case yaml.AliasNode:
		return deAnchor(yn.Alias)
	case yaml.DocumentNode, yaml.MappingNode, yaml.SequenceNode:
		for i := range yn.Content {
			yn.Content[i], err = deAnchor(yn.Content[i])
			if err != nil {
				return nil, err
			}
		}
		return yn, nil
	default:
		return nil, fmt.Errorf("cannot deAnchor kind %q", yn.Kind)
	}
}

// GetValidatedMetadata returns metadata after subjecting it to some tests.
func (rn *RNode) GetValidatedMetadata() (ResourceMeta, error) {
	m, err := rn.GetMeta()
	if err != nil {
		return m, err
	}
	if m.Kind == "" {
		return m, fmt.Errorf("missing kind in object %v", m)
	}
	if strings.HasSuffix(m.Kind, "List") {
		// A list doesn't require a name.
		return m, nil
	}
	if m.NameMeta.Name == "" {
		return m, fmt.Errorf("missing metadata.name in object %v", m)
	}
	return m, nil
}

// MatchesAnnotationSelector returns true on a selector match to annotations.
func (rn *RNode) MatchesAnnotationSelector(selector string) (bool, error) {
	s, err := labels.Parse(selector)
	if err != nil {
		return false, err
	}
	return s.Matches(labels.Set(rn.GetAnnotations())), nil
}

// MatchesLabelSelector returns true on a selector match to labels.
func (rn *RNode) MatchesLabelSelector(selector string) (bool, error) {
	s, err := labels.Parse(selector)
	if err != nil {
		return false, err
	}
	return s.Matches(labels.Set(rn.GetLabels())), nil
}

// HasNilEntryInList returns true if the RNode contains a list which has
// a nil item, along with the path to the missing item.
// TODO(broken): This doesn't do what it claims to do.
// (see TODO in unit test and pr 1513).
func (rn *RNode) HasNilEntryInList() (bool, string) {
	return hasNilEntryInList(rn.value)
}

func hasNilEntryInList(in interface{}) (bool, string) {
	switch v := in.(type) {
	case map[string]interface{}:
		for key, s := range v {
			if result, path := hasNilEntryInList(s); result {
				return result, key + "/" + path
			}
		}
	case []interface{}:
		for index, s := range v {
			if s == nil {
				return true, ""
			}
			if result, path := hasNilEntryInList(s); result {
				return result, "[" + strconv.Itoa(index) + "]/" + path
			}
		}
	}
	return false, ""
}

func FromMap(m map[string]interface{}) (*RNode, error) {
	c, err := Marshal(m)
	if err != nil {
		return nil, err
	}
	return Parse(string(c))
}

func (rn *RNode) Map() (map[string]interface{}, error) {
	if rn == nil || rn.value == nil {
		return make(map[string]interface{}), nil
	}
	var result map[string]interface{}
	if err := rn.value.Decode(&result); err != nil {
		// Should not be able to create an RNode that cannot be decoded;
		// this is an unrecoverable error.
		str, _ := rn.String()
		return nil, fmt.Errorf("received error %w for the following resource:\n%s", err, str)
	}
	return result, nil
}

// ConvertJSONToYamlNode parses input json string and returns equivalent yaml node
func ConvertJSONToYamlNode(jsonStr string) (*RNode, error) {
	var body map[string]interface{}
	err := json.Unmarshal([]byte(jsonStr), &body)
	if err != nil {
		return nil, err
	}
	yml, err := yaml.Marshal(body)
	if err != nil {
		return nil, err
	}
	node, err := Parse(string(yml))
	if err != nil {
		return nil, err
	}
	return node, nil
}

// checkKey returns true if all elems have the key
func checkKey(key string, elems []*Node) bool {
	count := 0
	for i := range elems {
		elem := NewRNode(elems[i])
		if elem.Field(key) != nil {
			count++
		}
	}
	return count == len(elems)
}

// Deprecated: use pipes instead.
// GetSlice returns the contents of the slice field at the given path.
func (rn *RNode) GetSlice(path string) ([]interface{}, error) {
	value, err := rn.GetFieldValue(path)
	if err != nil {
		return nil, err
	}
	if sliceValue, ok := value.([]interface{}); ok {
		return sliceValue, nil
	}
	return nil, fmt.Errorf("node %s is not a slice", path)
}

// Deprecated: use pipes instead.
// GetString returns the contents of the string field at the given path.
func (rn *RNode) GetString(path string) (string, error) {
	value, err := rn.GetFieldValue(path)
	if err != nil {
		return "", err
	}
	if v, ok := value.(string); ok {
		return v, nil
	}
	return "", fmt.Errorf("node %s is not a string: %v", path, value)
}

// Deprecated: use slash paths instead.
// GetFieldValue finds period delimited fields.
// TODO: When doing kustomize var replacement, which is likely a
// a primary use of this function and the reason it returns interface{}
// rather than string, we do conversion from Nodes to Go types and back
// to nodes.  We should figure out how to do replacement using raw nodes,
// assuming we keep the var feature in kustomize.
// The other end of this is: refvar.go:updateNodeValue.
func (rn *RNode) GetFieldValue(path string) (interface{}, error) {
	fields := convertSliceIndex(strings.Split(path, "."))
	rn, err := rn.Pipe(Lookup(fields...))
	if err != nil {
		return nil, err
	}
	if rn == nil {
		return nil, NoFieldError{path}
	}
	yn := rn.YNode()

	// If this is an alias node, resolve it
	if yn.Kind == yaml.AliasNode {
		yn = yn.Alias
	}

	// Return value as map for DocumentNode and MappingNode kinds
	if yn.Kind == yaml.DocumentNode || yn.Kind == yaml.MappingNode {
		var result map[string]interface{}
		if err := yn.Decode(&result); err != nil {
			return nil, err
		}
		return result, err
	}

	// Return value as slice for SequenceNode kind
	if yn.Kind == yaml.SequenceNode {
		var result []interface{}
		if err := yn.Decode(&result); err != nil {
			return nil, err
		}
		return result, nil
	}
	if yn.Kind != yaml.ScalarNode {
		return nil, fmt.Errorf("expected ScalarNode, got Kind=%d", yn.Kind)
	}

	switch yn.Tag {
	case NodeTagString:
		return yn.Value, nil
	case NodeTagInt:
		return strconv.Atoi(yn.Value)
	case NodeTagFloat:
		return strconv.ParseFloat(yn.Value, 64)
	case NodeTagBool:
		return strconv.ParseBool(yn.Value)
	default:
		// Possibly this should be an error or log.
		return yn.Value, nil
	}
}

// convertSliceIndex traverses the items in `fields` and find
// if there is a slice index in the item and change it to a
// valid Lookup field path. For example, 'ports[0]' will be
// converted to 'ports' and '0'.
func convertSliceIndex(fields []string) []string {
	var res []string
	for _, s := range fields {
		if !strings.HasSuffix(s, "]") {
			res = append(res, s)
			continue
		}
		re := regexp.MustCompile(`^(.*)\[(\d+)\]$`)
		groups := re.FindStringSubmatch(s)
		if len(groups) == 0 {
			// no match, add to result
			res = append(res, s)
			continue
		}
		if groups[1] != "" {
			res = append(res, groups[1])
		}
		res = append(res, groups[2])
	}
	return res
}

type NoFieldError struct {
	Field string
}

func (e NoFieldError) Error() string {
	return fmt.Sprintf("no field named '%s'", e.Field)
}
