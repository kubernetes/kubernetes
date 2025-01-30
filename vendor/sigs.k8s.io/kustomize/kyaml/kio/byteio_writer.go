// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kio

import (
	"encoding/json"
	"io"
	"path/filepath"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/kio/kioutil"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// ByteWriter writes ResourceNodes to bytes. Generally YAML encoding will be used but in the special
// case of writing a single, bare yaml.RNode that has a kioutil.PathAnnotation indicating that the
// target is a JSON file JSON encoding is used. See shouldJSONEncodeSingleBareNode below for more
// information.
type ByteWriter struct {
	// Writer is where ResourceNodes are encoded.
	Writer io.Writer

	// KeepReaderAnnotations if set will keep the Reader specific annotations when writing
	// the Resources, otherwise they will be cleared.
	KeepReaderAnnotations bool

	// ClearAnnotations is a list of annotations to clear when writing the Resources.
	ClearAnnotations []string

	// Style is a style that is set on the Resource Node Document.
	Style yaml.Style

	// FunctionConfig is the function config for an ResourceList.  If non-nil
	// wrap the results in an ResourceList.
	FunctionConfig *yaml.RNode

	Results *yaml.RNode

	// WrappingKind if set will cause ByteWriter to wrap the Resources in
	// an 'items' field in this kind.  e.g. if WrappingKind is 'List',
	// ByteWriter will wrap the Resources in a List .items field.
	WrappingKind string

	// WrappingAPIVersion is the apiVersion for WrappingKind
	WrappingAPIVersion string

	// Sort if set, will cause ByteWriter to sort the nodes before writing them.
	Sort bool
}

var _ Writer = ByteWriter{}

func (w ByteWriter) Write(inputNodes []*yaml.RNode) error {
	// Copy the nodes to prevent writer from mutating the original nodes.
	nodes := copyRNodes(inputNodes)
	if w.Sort {
		if err := kioutil.SortNodes(nodes); err != nil {
			return errors.Wrap(err)
		}
	}

	// Even though we use the this value further down we must check this before removing annotations
	jsonEncodeSingleBareNode := w.shouldJSONEncodeSingleBareNode(nodes)

	// store seqindent annotation value for each node in order to set the encoder indentation
	var seqIndentsForNodes []string
	for i := range nodes {
		seqIndentsForNodes = append(seqIndentsForNodes, nodes[i].GetAnnotations()[kioutil.SeqIndentAnnotation])
	}

	for i := range nodes {
		// clean resources by removing annotations set by the Reader
		if !w.KeepReaderAnnotations {
			_, err := nodes[i].Pipe(yaml.ClearAnnotation(kioutil.IndexAnnotation))
			if err != nil {
				return errors.Wrap(err)
			}
			_, err = nodes[i].Pipe(yaml.ClearAnnotation(kioutil.LegacyIndexAnnotation))
			if err != nil {
				return errors.Wrap(err)
			}

			_, err = nodes[i].Pipe(yaml.ClearAnnotation(kioutil.SeqIndentAnnotation))
			if err != nil {
				return errors.Wrap(err)
			}
		}
		for _, a := range w.ClearAnnotations {
			_, err := nodes[i].Pipe(yaml.ClearAnnotation(a))
			if err != nil {
				return errors.Wrap(err)
			}
		}

		if err := yaml.ClearEmptyAnnotations(nodes[i]); err != nil {
			return err
		}

		if w.Style != 0 {
			nodes[i].YNode().Style = w.Style
		}
	}

	if jsonEncodeSingleBareNode {
		encoder := json.NewEncoder(w.Writer)
		encoder.SetIndent("", "  ")
		return errors.Wrap(encoder.Encode(nodes[0]))
	}

	encoder := yaml.NewEncoder(w.Writer)
	defer encoder.Close()
	// don't wrap the elements
	if w.WrappingKind == "" {
		for i := range nodes {
			if seqIndentsForNodes[i] == string(yaml.WideSequenceStyle) {
				encoder.DefaultSeqIndent()
			} else {
				encoder.CompactSeqIndent()
			}
			if err := encoder.Encode(upWrapBareSequenceNode(nodes[i].Document())); err != nil {
				return errors.Wrap(err)
			}
		}
		return nil
	}
	// wrap the elements in a list
	items := &yaml.Node{Kind: yaml.SequenceNode}
	list := &yaml.Node{
		Kind:  yaml.MappingNode,
		Style: w.Style,
		Content: []*yaml.Node{
			{Kind: yaml.ScalarNode, Value: "apiVersion"},
			{Kind: yaml.ScalarNode, Value: w.WrappingAPIVersion},
			{Kind: yaml.ScalarNode, Value: "kind"},
			{Kind: yaml.ScalarNode, Value: w.WrappingKind},
			{Kind: yaml.ScalarNode, Value: "items"}, items,
		}}
	if w.FunctionConfig != nil {
		list.Content = append(list.Content,
			&yaml.Node{Kind: yaml.ScalarNode, Value: "functionConfig"},
			w.FunctionConfig.YNode())
	}
	if w.Results != nil {
		list.Content = append(list.Content,
			&yaml.Node{Kind: yaml.ScalarNode, Value: "results"},
			w.Results.YNode())
	}
	doc := &yaml.Node{
		Kind:    yaml.DocumentNode,
		Content: []*yaml.Node{list}}
	for i := range nodes {
		items.Content = append(items.Content, nodes[i].YNode())
	}
	return encoder.Encode(doc)
}

func copyRNodes(in []*yaml.RNode) []*yaml.RNode {
	out := make([]*yaml.RNode, len(in))
	for i := range in {
		out[i] = in[i].Copy()
	}
	return out
}

// shouldJSONEncodeSingleBareNode determines if nodes contain a single node that should not be
// wrapped and has a JSON file extension, which in turn means that the node should be JSON encoded.
// Note 1: this must be checked before any annotations to avoid losing information about the target
//         filename extension.
// Note 2: JSON encoding should only be used for single, unwrapped nodes because multiple unwrapped
//         nodes cannot be represented in JSON (no multi doc support). Furthermore, the typical use
//         cases for wrapping nodes would likely not include later writing the whole wrapper to a
//         .json file, i.e. there is no point risking any edge case information loss e.g. comments
//         disappearing, that could come from JSON encoding the whole wrapper just to ensure that
//         one (or all nodes) can be read as JSON.
func (w ByteWriter) shouldJSONEncodeSingleBareNode(nodes []*yaml.RNode) bool {
	if w.WrappingKind == "" && len(nodes) == 1 {
		if path, _, _ := kioutil.GetFileAnnotations(nodes[0]); path != "" {
			filename := filepath.Base(path)
			for _, glob := range JSONMatch {
				if match, _ := filepath.Match(glob, filename); match {
					return true
				}
			}
		}
	}
	return false
}

// upWrapBareSequenceNode unwraps the bare sequence nodes wrapped by yaml.BareSeqNodeWrappingKey
func upWrapBareSequenceNode(node *yaml.Node) *yaml.Node {
	rNode := yaml.NewRNode(node)
	seqNode, err := rNode.Pipe(yaml.Lookup(yaml.BareSeqNodeWrappingKey))
	if err == nil && !seqNode.IsNilOrEmpty() {
		return seqNode.YNode()
	}
	return node
}
