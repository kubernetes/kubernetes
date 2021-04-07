// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package wrappy

import (
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"

	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// WNode implements ifc.Kunstructured using yaml.RNode.
//
// It exists only to help manage a switch from
// kunstruct.UnstructAdapter to yaml.RNode as the core
// representation of KRM objects in kustomize.
//
// It's got a silly name because we don't want it around for long,
// and want its use to be obvious.
type WNode struct {
	node *yaml.RNode
}

var _ ifc.Kunstructured = (*WNode)(nil)

func NewWNode() *WNode {
	return FromRNode(yaml.NewRNode(nil))
}

func FromMap(m map[string]interface{}) (*WNode, error) {
	n, err := yaml.FromMap(m)
	if err != nil {
		return nil, err
	}
	return FromRNode(n), nil
}

func FromRNode(node *yaml.RNode) *WNode {
	return &WNode{node: node}
}

func (wn *WNode) AsRNode() *yaml.RNode {
	return wn.node
}

func (wn *WNode) demandMetaData(label string) yaml.ResourceMeta {
	meta, err := wn.node.GetMeta()
	if err != nil {
		// Log and die since interface doesn't allow error.
		log.Fatalf("for %s', expected valid resource: %v", label, err)
	}
	return meta
}

// Copy implements ifc.Kunstructured.
func (wn *WNode) Copy() ifc.Kunstructured {
	return &WNode{node: wn.node.Copy()}
}

// GetAnnotations implements ifc.Kunstructured.
func (wn *WNode) GetAnnotations() map[string]string {
	return wn.demandMetaData("GetAnnotations").Annotations
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

// GetFieldValue implements ifc.Kunstructured.
func (wn *WNode) GetFieldValue(path string) (interface{}, error) {
	fields := convertSliceIndex(strings.Split(path, "."))
	rn, err := wn.node.Pipe(yaml.Lookup(fields...))
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

	// TODO: When doing kustomize var replacement, which is likely a
	// a primary use of this function and the reason it returns interface{}
	// rather than string, we do conversion from Nodes to Go types and back
	// to nodes.  We should figure out how to do replacement using raw nodes,
	// assuming we keep the var feature in kustomize.
	// The other end of this is: refvar.go:updateNodeValue.
	switch yn.Tag {
	case yaml.NodeTagString:
		return yn.Value, nil
	case yaml.NodeTagInt:
		return strconv.Atoi(yn.Value)
	case yaml.NodeTagFloat:
		return strconv.ParseFloat(yn.Value, 64)
	case yaml.NodeTagBool:
		return strconv.ParseBool(yn.Value)
	default:
		// Possibly this should be an error or log.
		return yn.Value, nil
	}
}

// GetGvk implements ifc.Kunstructured.
func (wn *WNode) GetGvk() resid.Gvk {
	meta := wn.demandMetaData("GetGvk")
	g, v := resid.ParseGroupVersion(meta.APIVersion)
	return resid.Gvk{Group: g, Version: v, Kind: meta.Kind}
}

// GetDataMap implements ifc.Kunstructured.
func (wn *WNode) GetDataMap() map[string]string {
	return wn.node.GetDataMap()
}

// SetDataMap implements ifc.Kunstructured.
func (wn *WNode) SetDataMap(m map[string]string) {
	wn.node.SetDataMap(m)
}

// GetBinaryDataMap implements ifc.Kunstructured.
func (wn *WNode) GetBinaryDataMap() map[string]string {
	return wn.node.GetBinaryDataMap()
}

// SetBinaryDataMap implements ifc.Kunstructured.
func (wn *WNode) SetBinaryDataMap(m map[string]string) {
	wn.node.SetBinaryDataMap(m)
}

// GetKind implements ifc.Kunstructured.
func (wn *WNode) GetKind() string {
	return wn.demandMetaData("GetKind").Kind
}

// GetLabels implements ifc.Kunstructured.
func (wn *WNode) GetLabels() map[string]string {
	return wn.demandMetaData("GetLabels").Labels
}

// GetName implements ifc.Kunstructured.
func (wn *WNode) GetName() string {
	return wn.demandMetaData("GetName").Name
}

// GetSlice implements ifc.Kunstructured.
func (wn *WNode) GetSlice(path string) ([]interface{}, error) {
	value, err := wn.GetFieldValue(path)
	if err != nil {
		return nil, err
	}
	if sliceValue, ok := value.([]interface{}); ok {
		return sliceValue, nil
	}
	return nil, fmt.Errorf("node %s is not a slice", path)
}

// GetSlice implements ifc.Kunstructured.
func (wn *WNode) GetString(path string) (string, error) {
	value, err := wn.GetFieldValue(path)
	if err != nil {
		return "", err
	}
	if v, ok := value.(string); ok {
		return v, nil
	}
	return "", fmt.Errorf("node %s is not a string: %v", path, value)
}

// Map implements ifc.Kunstructured.
func (wn *WNode) Map() (map[string]interface{}, error) {
	return wn.node.Map()
}

// MarshalJSON implements ifc.Kunstructured.
func (wn *WNode) MarshalJSON() ([]byte, error) {
	return wn.node.MarshalJSON()
}

// MatchesAnnotationSelector implements ifc.Kunstructured.
func (wn *WNode) MatchesAnnotationSelector(selector string) (bool, error) {
	return wn.node.MatchesAnnotationSelector(selector)
}

// MatchesLabelSelector implements ifc.Kunstructured.
func (wn *WNode) MatchesLabelSelector(selector string) (bool, error) {
	return wn.node.MatchesLabelSelector(selector)
}

// SetAnnotations implements ifc.Kunstructured.
func (wn *WNode) SetAnnotations(annotations map[string]string) {
	if err := wn.node.SetAnnotations(annotations); err != nil {
		log.Fatal(err) // interface doesn't allow error.
	}
}

// SetGvk implements ifc.Kunstructured.
func (wn *WNode) SetGvk(gvk resid.Gvk) {
	wn.setMapField(yaml.NewScalarRNode(gvk.Kind), yaml.KindField)
	wn.setMapField(yaml.NewScalarRNode(gvk.ApiVersion()), yaml.APIVersionField)
}

// SetLabels implements ifc.Kunstructured.
func (wn *WNode) SetLabels(labels map[string]string) {
	if err := wn.node.SetLabels(labels); err != nil {
		log.Fatal(err) // interface doesn't allow error.
	}
}

// SetName implements ifc.Kunstructured.
func (wn *WNode) SetName(name string) {
	wn.setMapField(yaml.NewScalarRNode(name), yaml.MetadataField, yaml.NameField)
}

// SetNamespace implements ifc.Kunstructured.
func (wn *WNode) SetNamespace(ns string) {
	if err := wn.node.SetNamespace(ns); err != nil {
		log.Fatal(err) // interface doesn't allow error.
	}
}

func (wn *WNode) setMapField(value *yaml.RNode, path ...string) {
	if err := wn.node.SetMapField(value, path...); err != nil {
		// Log and die since interface doesn't allow error.
		log.Fatalf("failed to set field %v: %v", path, err)
	}
}

// UnmarshalJSON implements ifc.Kunstructured.
func (wn *WNode) UnmarshalJSON(data []byte) error {
	return wn.node.UnmarshalJSON(data)
}

type NoFieldError struct {
	Field string
}

func (e NoFieldError) Error() string {
	return fmt.Sprintf("no field named '%s'", e.Field)
}
