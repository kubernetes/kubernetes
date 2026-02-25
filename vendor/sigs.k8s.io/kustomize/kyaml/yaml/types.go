// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import (
	"bytes"
	"strings"

	yaml "go.yaml.in/yaml/v3"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/sets"
)

// CopyYNode returns a distinct copy of its argument.
// Use https://github.com/jinzhu/copier instead?
func CopyYNode(n *yaml.Node) *yaml.Node {
	if n == nil {
		return nil
	}
	c := *n
	if len(n.Content) > 0 {
		// Using Go 'copy' here doesn't yield independent slices.
		c.Content = make([]*Node, len(n.Content))
		for i, item := range n.Content {
			c.Content[i] = CopyYNode(item)
		}
	}
	return &c
}

// IsYNodeTaggedNull returns true if the node is explicitly tagged Null.
func IsYNodeTaggedNull(n *yaml.Node) bool {
	return n != nil && n.Tag == NodeTagNull
}

// IsYNodeEmptyMap is true if the Node is a non-nil empty map.
func IsYNodeEmptyMap(n *yaml.Node) bool {
	return n != nil && n.Kind == yaml.MappingNode && len(n.Content) == 0
}

// IsYNodeEmptySeq is true if the Node is a non-nil empty sequence.
func IsYNodeEmptySeq(n *yaml.Node) bool {
	return n != nil && n.Kind == yaml.SequenceNode && len(n.Content) == 0
}

// IsYNodeNilOrEmpty is true if the Node is nil or appears empty.
func IsYNodeNilOrEmpty(n *yaml.Node) bool {
	return n == nil ||
		IsYNodeTaggedNull(n) ||
		IsYNodeEmptyMap(n) ||
		IsYNodeEmptySeq(n) ||
		IsYNodeZero(n)
}

// IsYNodeEmptyDoc is true if the node is a Document with no content.
// E.g.: "---\n---"
func IsYNodeEmptyDoc(n *yaml.Node) bool {
	return n.Kind == yaml.DocumentNode && n.Content[0].Tag == NodeTagNull
}

func IsYNodeString(n *yaml.Node) bool {
	return n.Kind == yaml.ScalarNode &&
		(n.Tag == NodeTagString || n.Tag == NodeTagEmpty)
}

// IsYNodeZero is true if all the public fields in the Node are empty.
// Which means it's not initialized and should be omitted when marshal.
// The Node itself has a method IsZero but it is not released
// in yaml.v3. https://pkg.go.dev/gopkg.in/yaml.v3#Node.IsZero
func IsYNodeZero(n *yaml.Node) bool {
	// TODO: Change this to use IsZero when it's avaialable.
	return n != nil && n.Kind == 0 && n.Style == 0 && n.Tag == "" && n.Value == "" &&
		n.Anchor == "" && n.Alias == nil && n.Content == nil &&
		n.HeadComment == "" && n.LineComment == "" && n.FootComment == "" &&
		n.Line == 0 && n.Column == 0
}

// Parser parses values into configuration.
type Parser struct {
	Kind  string `yaml:"kind,omitempty"`
	Value string `yaml:"value,omitempty"`
}

func (p Parser) Filter(_ *RNode) (*RNode, error) {
	d := yaml.NewDecoder(bytes.NewBuffer([]byte(p.Value)))
	o := &RNode{value: &yaml.Node{}}
	return o, d.Decode(o.value)
}

// TODO(pwittrock): test this
func GetStyle(styles ...string) Style {
	var style Style
	for _, s := range styles {
		switch s {
		case "TaggedStyle":
			style |= TaggedStyle
		case "DoubleQuotedStyle":
			style |= DoubleQuotedStyle
		case "SingleQuotedStyle":
			style |= SingleQuotedStyle
		case "LiteralStyle":
			style |= LiteralStyle
		case "FoldedStyle":
			style |= FoldedStyle
		case "FlowStyle":
			style |= FlowStyle
		}
	}
	return style
}

// Filter defines a function to manipulate an individual RNode such as by changing
// its values, or returning a field.
//
// When possible, Filters should be serializable to yaml so that they can be described
// declaratively as data.
//
// Analogous to http://www.linfo.org/filters.html
type Filter interface {
	Filter(object *RNode) (*RNode, error)
}

type FilterFunc func(object *RNode) (*RNode, error)

func (f FilterFunc) Filter(object *RNode) (*RNode, error) {
	return f(object)
}

// TypeMeta partially copies apimachinery/pkg/apis/meta/v1.TypeMeta
// No need for a direct dependence; the fields are stable.
type TypeMeta struct {
	// APIVersion is the apiVersion field of a Resource
	APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	// Kind is the kind field of a Resource
	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`
}

// NameMeta contains name information.
type NameMeta struct {
	// Name is the metadata.name field of a Resource
	Name string `json:"name,omitempty" yaml:"name,omitempty"`
	// Namespace is the metadata.namespace field of a Resource
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
}

// ResourceMeta contains the metadata for a both Resource Type and Resource.
type ResourceMeta struct {
	TypeMeta `json:",inline" yaml:",inline"`
	// ObjectMeta is the metadata field of a Resource
	ObjectMeta `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// ObjectMeta contains metadata about a Resource
type ObjectMeta struct {
	NameMeta `json:",inline" yaml:",inline"`
	// Labels is the metadata.labels field of a Resource
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
	// Annotations is the metadata.annotations field of a Resource.
	Annotations map[string]string `json:"annotations,omitempty" yaml:"annotations,omitempty"`
}

// GetIdentifier returns a ResourceIdentifier that includes
// the information needed to uniquely identify a resource in a cluster.
func (m *ResourceMeta) GetIdentifier() ResourceIdentifier {
	return ResourceIdentifier{
		TypeMeta: m.TypeMeta,
		NameMeta: m.NameMeta,
	}
}

// ResourceIdentifier contains the information needed to uniquely
// identify a resource in a cluster.
type ResourceIdentifier struct {
	TypeMeta `json:",inline" yaml:",inline"`
	NameMeta `json:",inline" yaml:",inline"`
}

// Comments struct is comment yaml comment types
type Comments struct {
	LineComment string `yaml:"lineComment,omitempty"`
	HeadComment string `yaml:"headComment,omitempty"`
	FootComment string `yaml:"footComment,omitempty"`
}

func (r *ResourceIdentifier) GetName() string {
	return r.Name
}

func (r *ResourceIdentifier) GetNamespace() string {
	return r.Namespace
}

func (r *ResourceIdentifier) GetAPIVersion() string {
	return r.APIVersion
}

func (r *ResourceIdentifier) GetKind() string {
	return r.Kind
}

const (
	Trim = "Trim"
	Flow = "Flow"
)

// String returns a string value for a Node, applying the supplied formatting options
func String(node *yaml.Node, opts ...string) (string, error) {
	if node == nil {
		return "", nil
	}
	optsSet := sets.String{}
	optsSet.Insert(opts...)
	if optsSet.Has(Flow) {
		oldStyle := node.Style
		defer func() {
			node.Style = oldStyle
		}()
		node.Style = yaml.FlowStyle
	}

	b := &bytes.Buffer{}
	e := NewEncoder(b)
	err := e.Encode(node)
	errClose := e.Close()
	if err == nil {
		err = errClose
	}
	val := b.String()
	if optsSet.Has(Trim) {
		val = strings.TrimSpace(val)
	}
	return val, errors.Wrap(err)
}

// MergeOptionsListIncreaseDirection is the type of list growth in merge
type MergeOptionsListIncreaseDirection int

const (
	MergeOptionsListAppend MergeOptionsListIncreaseDirection = iota
	MergeOptionsListPrepend
)

// MergeOptions is a struct which contains the options for merge
type MergeOptions struct {
	// ListIncreaseDirection indicates should merge function prepend the items from
	// source list to destination or append.
	ListIncreaseDirection MergeOptionsListIncreaseDirection
}

// Since ObjectMeta and TypeMeta are stable, we manually create DeepCopy funcs for ResourceMeta and ObjectMeta.
// For TypeMeta and NameMeta no DeepCopy funcs are required, as they only contain basic types.

// DeepCopyInto copies the receiver, writing into out. in must be non-nil.
func (in *ObjectMeta) DeepCopyInto(out *ObjectMeta) {
	*out = *in
	out.NameMeta = in.NameMeta
	if in.Labels != nil {
		in, out := &in.Labels, &out.Labels
		*out = make(map[string]string, len(*in))
		for key, val := range *in {
			(*out)[key] = val
		}
	}
	if in.Annotations != nil {
		in, out := &in.Annotations, &out.Annotations
		*out = make(map[string]string, len(*in))
		for key, val := range *in {
			(*out)[key] = val
		}
	}
}

// DeepCopy copies the receiver, creating a new ObjectMeta.
func (in *ObjectMeta) DeepCopy() *ObjectMeta {
	if in == nil {
		return nil
	}
	out := new(ObjectMeta)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto copies the receiver, writing into out. in must be non-nil.
func (in *ResourceMeta) DeepCopyInto(out *ResourceMeta) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
}

// DeepCopy copies the receiver, creating a new ResourceMeta.
func (in *ResourceMeta) DeepCopy() *ResourceMeta {
	if in == nil {
		return nil
	}
	out := new(ResourceMeta)
	in.DeepCopyInto(out)
	return out
}
