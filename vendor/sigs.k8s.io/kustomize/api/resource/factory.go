// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resource

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/generators"
	"sigs.k8s.io/kustomize/api/internal/kusterr"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/resid"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Factory makes instances of Resource.
type Factory struct {
	hasher ifc.KustHasher

	// When set to true, IncludeLocalConfigs indicates
	// that Factory should include resources with the
	// annotation 'config.kubernetes.io/local-config'.
	// By default these resources are ignored.
	IncludeLocalConfigs bool
}

// NewFactory makes an instance of Factory.
func NewFactory(h ifc.KustHasher) *Factory {
	return &Factory{hasher: h}
}

// Hasher returns an ifc.KustHasher
func (rf *Factory) Hasher() ifc.KustHasher {
	return rf.hasher
}

// FromMap returns a new instance of Resource.
func (rf *Factory) FromMap(m map[string]interface{}) *Resource {
	return rf.FromMapAndOption(m, nil)
}

// FromMapWithName returns a new instance with the given "original" name.
func (rf *Factory) FromMapWithName(n string, m map[string]interface{}) *Resource {
	return rf.FromMapWithNamespaceAndName(resid.DefaultNamespace, n, m)
}

// FromMapWithNamespaceAndName returns a new instance with the given "original" namespace.
func (rf *Factory) FromMapWithNamespaceAndName(ns string, n string, m map[string]interface{}) *Resource {
	r := rf.FromMapAndOption(m, nil)
	return r.setPreviousId(ns, n, r.GetKind())
}

// FromMapAndOption returns a new instance of Resource with given options.
func (rf *Factory) FromMapAndOption(
	m map[string]interface{}, args *types.GeneratorArgs) *Resource {
	n, err := yaml.FromMap(m)
	if err != nil {
		// TODO: return err instead of log.
		log.Fatal(err)
	}
	return rf.makeOne(n, types.NewGenArgs(args))
}

// makeOne returns a new instance of Resource.
func (rf *Factory) makeOne(rn *yaml.RNode, o *types.GenArgs) *Resource {
	if rn == nil {
		log.Fatal("RNode must not be null")
	}
	if o == nil {
		o = types.NewGenArgs(nil)
	}
	return &Resource{RNode: *rn, options: o}
}

// SliceFromPatches returns a slice of resources given a patch path
// slice from a kustomization file.
func (rf *Factory) SliceFromPatches(
	ldr ifc.Loader, paths []types.PatchStrategicMerge) ([]*Resource, error) {
	var result []*Resource
	for _, path := range paths {
		content, err := ldr.Load(string(path))
		if err != nil {
			return nil, err
		}
		res, err := rf.SliceFromBytes(content)
		if err != nil {
			return nil, kusterr.Handler(err, string(path))
		}
		result = append(result, res...)
	}
	return result, nil
}

// FromBytes unmarshalls bytes into one Resource.
func (rf *Factory) FromBytes(in []byte) (*Resource, error) {
	result, err := rf.SliceFromBytes(in)
	if err != nil {
		return nil, err
	}
	if len(result) != 1 {
		return nil, fmt.Errorf(
			"expected 1 resource, found %d in %v", len(result), in)
	}
	return result[0], nil
}

// SliceFromBytes unmarshals bytes into a Resource slice.
func (rf *Factory) SliceFromBytes(in []byte) ([]*Resource, error) {
	nodes, err := rf.RNodesFromBytes(in)
	if err != nil {
		return nil, err
	}
	return rf.resourcesFromRNodes(nodes), nil
}

// ResourcesFromRNodes converts RNodes to Resources.
func (rf *Factory) ResourcesFromRNodes(
	nodes []*yaml.RNode) (result []*Resource, err error) {
	nodes, err = rf.dropBadNodes(nodes)
	if err != nil {
		return nil, err
	}
	return rf.resourcesFromRNodes(nodes), nil
}

// resourcesFromRNode assumes all nodes are good.
func (rf *Factory) resourcesFromRNodes(
	nodes []*yaml.RNode) (result []*Resource) {
	for _, n := range nodes {
		result = append(result, rf.makeOne(n, nil))
	}
	return
}

func (rf *Factory) RNodesFromBytes(b []byte) (result []*yaml.RNode, err error) {
	nodes, err := kio.FromBytes(b)
	if err != nil {
		return nil, err
	}
	nodes, err = rf.dropBadNodes(nodes)
	if err != nil {
		return nil, err
	}
	for len(nodes) > 0 {
		n0 := nodes[0]
		nodes = nodes[1:]
		kind := n0.GetKind()
		if !strings.HasSuffix(kind, "List") {
			result = append(result, n0)
			continue
		}
		// Convert a FooList into a slice of Foo.
		var m map[string]interface{}
		m, err = n0.Map()
		if err != nil {
			return nil, err
		}
		items, ok := m["items"]
		if !ok {
			// treat as an empty list
			continue
		}
		slice, ok := items.([]interface{})
		if !ok {
			if items == nil {
				// an empty list
				continue
			}
			return nil, fmt.Errorf(
				"expected array in %s/items, but found %T", kind, items)
		}
		innerNodes, err := rf.convertObjectSliceToNodeSlice(slice)
		if err != nil {
			return nil, err
		}
		nodes = append(nodes, innerNodes...)
	}
	return result, nil
}

// convertObjectSlice converts a list of objects to a list of RNode.
func (rf *Factory) convertObjectSliceToNodeSlice(
	objects []interface{}) (result []*yaml.RNode, err error) {
	var bytes []byte
	var nodes []*yaml.RNode
	for _, obj := range objects {
		bytes, err = json.Marshal(obj)
		if err != nil {
			return
		}
		nodes, err = kio.FromBytes(bytes)
		if err != nil {
			return
		}
		nodes, err = rf.dropBadNodes(nodes)
		if err != nil {
			return
		}
		result = append(result, nodes...)
	}
	return
}

// dropBadNodes may drop some nodes from its input argument.
func (rf *Factory) dropBadNodes(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	var result []*yaml.RNode
	for _, n := range nodes {
		ignore, err := rf.shouldIgnore(n)
		if err != nil {
			return nil, err
		}
		if !ignore {
			result = append(result, n)
		}
	}
	return result, nil
}

// shouldIgnore returns true if there's some reason to ignore the node.
func (rf *Factory) shouldIgnore(n *yaml.RNode) (bool, error) {
	if n.IsNilOrEmpty() {
		return true, nil
	}
	if !rf.IncludeLocalConfigs {
		md, err := n.GetValidatedMetadata()
		if err != nil {
			return true, err
		}
		_, ignore := md.ObjectMeta.Annotations[konfig.IgnoredByKustomizeAnnotation]
		if ignore {
			return true, nil
		}
	}
	if foundNil, path := n.HasNilEntryInList(); foundNil {
		return true, fmt.Errorf("empty item at %v in object %v", path, n)
	}
	return false, nil
}

// SliceFromBytesWithNames unmarshals bytes into a Resource slice with specified original
// name.
func (rf *Factory) SliceFromBytesWithNames(names []string, in []byte) ([]*Resource, error) {
	result, err := rf.SliceFromBytes(in)
	if err != nil {
		return nil, err
	}
	if len(names) != len(result) {
		return nil, fmt.Errorf("number of names doesn't match number of resources")
	}
	for i, res := range result {
		res.setPreviousId(resid.DefaultNamespace, names[i], res.GetKind())
	}
	return result, nil
}

// MakeConfigMap makes an instance of Resource for ConfigMap
func (rf *Factory) MakeConfigMap(kvLdr ifc.KvLoader, args *types.ConfigMapArgs) (*Resource, error) {
	rn, err := generators.MakeConfigMap(kvLdr, args)
	if err != nil {
		return nil, err
	}
	return rf.makeOne(rn, types.NewGenArgs(&args.GeneratorArgs)), nil
}

// MakeSecret makes an instance of Resource for Secret
func (rf *Factory) MakeSecret(kvLdr ifc.KvLoader, args *types.SecretArgs) (*Resource, error) {
	rn, err := generators.MakeSecret(kvLdr, args)
	if err != nil {
		return nil, err
	}
	return rf.makeOne(rn, types.NewGenArgs(&args.GeneratorArgs)), nil
}
