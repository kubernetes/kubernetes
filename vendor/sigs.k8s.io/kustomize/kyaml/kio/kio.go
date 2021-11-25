// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package kio contains low-level libraries for reading, modifying and writing
// Resource Configuration and packages.
package kio

import (
	"fmt"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/kio/kioutil"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Reader reads ResourceNodes. Analogous to io.Reader.
type Reader interface {
	Read() ([]*yaml.RNode, error)
}

// ResourceNodeSlice is a collection of ResourceNodes.
// While ResourceNodeSlice has no inherent constraints on ordering or uniqueness, specific
// Readers, Filters or Writers may have constraints.
type ResourceNodeSlice []*yaml.RNode

var _ Reader = ResourceNodeSlice{}

func (o ResourceNodeSlice) Read() ([]*yaml.RNode, error) {
	return o, nil
}

// Writer writes ResourceNodes. Analogous to io.Writer.
type Writer interface {
	Write([]*yaml.RNode) error
}

// WriterFunc implements a Writer as a function.
type WriterFunc func([]*yaml.RNode) error

func (fn WriterFunc) Write(o []*yaml.RNode) error {
	return fn(o)
}

// ReaderWriter implements both Reader and Writer interfaces
type ReaderWriter interface {
	Reader
	Writer
}

// Filter modifies a collection of Resource Configuration by returning the modified slice.
// When possible, Filters should be serializable to yaml so that they can be described
// as either data or code.
//
// Analogous to http://www.linfo.org/filters.html
type Filter interface {
	Filter([]*yaml.RNode) ([]*yaml.RNode, error)
}

// FilterFunc implements a Filter as a function.
type FilterFunc func([]*yaml.RNode) ([]*yaml.RNode, error)

func (fn FilterFunc) Filter(o []*yaml.RNode) ([]*yaml.RNode, error) {
	return fn(o)
}

// Pipeline reads Resource Configuration from a set of Inputs, applies some
// transformation filters, and writes the results to a set of Outputs.
//
// Analogous to http://www.linfo.org/pipes.html
type Pipeline struct {
	// Inputs provide sources for Resource Configuration to be read.
	Inputs []Reader `yaml:"inputs,omitempty"`

	// Filters are transformations applied to the Resource Configuration.
	// They are applied in the order they are specified.
	// Analogous to http://www.linfo.org/filters.html
	Filters []Filter `yaml:"filters,omitempty"`

	// Outputs are where the transformed Resource Configuration is written.
	Outputs []Writer `yaml:"outputs,omitempty"`

	// ContinueOnEmptyResult configures what happens when a filter in the pipeline
	// returns an empty result.
	// If it is false (default), subsequent filters will be skipped and the result
	// will be returned immediately. This is useful as an optimization when you
	// know that subsequent filters will not alter the empty result.
	// If it is true, the empty result will be provided as input to the next
	// filter in the list. This is useful when subsequent functions in the
	// pipeline may generate new resources.
	ContinueOnEmptyResult bool `yaml:"continueOnEmptyResult,omitempty"`
}

// Execute executes each step in the sequence, returning immediately after encountering
// any error as part of the Pipeline.
func (p Pipeline) Execute() error {
	return p.ExecuteWithCallback(nil)
}

// PipelineExecuteCallbackFunc defines a callback function that will be called each time a step in the pipeline succeeds.
type PipelineExecuteCallbackFunc = func(op Filter)

// ExecuteWithCallback executes each step in the sequence, returning immediately after encountering
// any error as part of the Pipeline. The callback will be called each time a step succeeds.
func (p Pipeline) ExecuteWithCallback(callback PipelineExecuteCallbackFunc) error {
	var result []*yaml.RNode

	// read from the inputs
	for _, i := range p.Inputs {
		nodes, err := i.Read()
		if err != nil {
			return errors.Wrap(err)
		}
		result = append(result, nodes...)
	}

	// apply operations
	var err error
	for i := range p.Filters {
		// Not all RNodes passed through kio.Pipeline have metadata nor should
		// they all be required to.
		var nodeAnnos map[string]map[string]string
		nodeAnnos, err = storeInternalAnnotations(result)
		if err != nil && err != yaml.ErrMissingMetadata {
			return err
		}

		op := p.Filters[i]
		if callback != nil {
			callback(op)
		}
		result, err = op.Filter(result)
		// TODO (issue 2872): This len(result) == 0 should be removed and empty result list should be
		// handled by outputs. However currently some writer like LocalPackageReadWriter
		// will clear the output directory and which will cause unpredictable results
		if len(result) == 0 && !p.ContinueOnEmptyResult || err != nil {
			return errors.Wrap(err)
		}

		// If either the internal annotations for path, index, and id OR the legacy
		// annotations for path, index, and id are changed, we have to update the other.
		err = reconcileInternalAnnotations(result, nodeAnnos)
		if err != nil && err != yaml.ErrMissingMetadata {
			return err
		}
	}

	// write to the outputs
	for _, o := range p.Outputs {
		if err := o.Write(result); err != nil {
			return errors.Wrap(err)
		}
	}
	return nil
}

// FilterAll runs the yaml.Filter against all inputs
func FilterAll(filter yaml.Filter) Filter {
	return FilterFunc(func(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
		for i := range nodes {
			_, err := filter.Filter(nodes[i])
			if err != nil {
				return nil, errors.Wrap(err)
			}
		}
		return nodes, nil
	})
}

// Store the original path, index, and id annotations so that we can reconcile
// it later. This is necessary because currently both internal-prefixed annotations
// and legacy annotations are currently supported, and a change to one must be
// reflected in the other.
func storeInternalAnnotations(result []*yaml.RNode) (map[string]map[string]string, error) {
	nodeAnnosMap := make(map[string]map[string]string)

	for i := range result {
		if err := kioutil.CopyLegacyAnnotations(result[i]); err != nil {
			return nil, err
		}
		meta, err := result[i].GetMeta()
		if err != nil {
			return nil, err
		}
		if err := checkMismatchedAnnos(meta); err != nil {
			return nil, err
		}

		path := meta.Annotations[kioutil.PathAnnotation]
		index := meta.Annotations[kioutil.IndexAnnotation]
		id := meta.Annotations[kioutil.IdAnnotation]

		if _, ok := nodeAnnosMap[path]; !ok {
			nodeAnnosMap[path] = make(map[string]string)
		}
		nodeAnnosMap[path][index] = id
	}
	return nodeAnnosMap, nil
}

func checkMismatchedAnnos(meta yaml.ResourceMeta) error {
	path := meta.Annotations[kioutil.PathAnnotation]
	index := meta.Annotations[kioutil.IndexAnnotation]
	id := meta.Annotations[kioutil.IdAnnotation]

	legacyPath := meta.Annotations[kioutil.LegacyPathAnnotation]
	legacyIndex := meta.Annotations[kioutil.LegacyIndexAnnotation]
	legacyId := meta.Annotations[kioutil.LegacyIdAnnotation]

	// if prior to running the functions, the legacy and internal annotations differ,
	// throw an error as we cannot infer the user's intent.
	if path != legacyPath {
		return fmt.Errorf("resource input to function has mismatched legacy and internal path annotations")
	}
	if index != legacyIndex {
		return fmt.Errorf("resource input to function has mismatched legacy and internal index annotations")
	}
	if id != legacyId {
		return fmt.Errorf("resource input to function has mismatched legacy and internal id annotations")
	}
	return nil
}

type nodeAnnotations struct {
	path  string
	index string
	id    string
}

func reconcileInternalAnnotations(result []*yaml.RNode, nodeAnnosMap map[string]map[string]string) error {
	for _, node := range result {
		meta, err := node.GetMeta()
		if err != nil {
			return err
		}
		// if only one annotation is set, set the other.
		err = missingInternalOrLegacyAnnotations(node, meta)
		if err != nil {
			return err
		}
		// we must check to see if the function changed either the new internal annotations
		// or the old legacy annotations. If one is changed, the change must be reflected
		// in the other.
		err = checkAnnotationsAltered(node, meta, nodeAnnosMap)
		if err != nil {
			return err
		}
		// if the annotations are still somehow out of sync, throw an error
		meta, err = node.GetMeta()
		if err != nil {
			return err
		}
		err = checkMismatchedAnnos(meta)
		if err != nil {
			return err
		}
	}
	return nil
}

func missingInternalOrLegacyAnnotations(rn *yaml.RNode, meta yaml.ResourceMeta) error {
	if err := missingInternalOrLegacyAnnotation(rn, meta, kioutil.PathAnnotation, kioutil.LegacyPathAnnotation); err != nil {
		return err
	}
	if err := missingInternalOrLegacyAnnotation(rn, meta, kioutil.IndexAnnotation, kioutil.LegacyIndexAnnotation); err != nil {
		return err
	}
	if err := missingInternalOrLegacyAnnotation(rn, meta, kioutil.IdAnnotation, kioutil.LegacyIdAnnotation); err != nil {
		return err
	}
	return nil
}

func missingInternalOrLegacyAnnotation(rn *yaml.RNode, meta yaml.ResourceMeta, newKey string, legacyKey string) error {
	value := meta.Annotations[newKey]
	legacyValue := meta.Annotations[legacyKey]

	if value == "" && legacyValue == "" {
		// do nothing
		return nil
	}

	if value == "" {
		// new key is not set, copy from legacy key
		if err := rn.PipeE(yaml.SetAnnotation(newKey, legacyValue)); err != nil {
			return err
		}
	} else if legacyValue == "" {
		// legacy key is not set, copy from new key
		if err := rn.PipeE(yaml.SetAnnotation(legacyKey, value)); err != nil {
			return err
		}
	}
	return nil
}

func checkAnnotationsAltered(rn *yaml.RNode, meta yaml.ResourceMeta, nodeAnnosMap map[string]map[string]string) error {
	// get the resource's current path, index, and ids from the new annotations
	internal := nodeAnnotations{
		path:  meta.Annotations[kioutil.PathAnnotation],
		index: meta.Annotations[kioutil.IndexAnnotation],
		id:    meta.Annotations[kioutil.IdAnnotation],
	}

	// get the resource's current path, index, and ids from the legacy annotations
	legacy := nodeAnnotations{
		path:  meta.Annotations[kioutil.LegacyPathAnnotation],
		index: meta.Annotations[kioutil.LegacyIndexAnnotation],
		id:    meta.Annotations[kioutil.LegacyIdAnnotation],
	}

	if internal.path == legacy.path &&
		internal.index == legacy.index &&
		internal.id == legacy.id {
		// none of the annotations differ, so no reconciliation is needed
		return nil
	}

	// nodeAnnosMap is a map of structure path -> index -> id that stores
	// all of the resources' path/index/id annotations prior to the functions
	// being run. We use that to check whether the legacy or new internal
	// annotations have been changed, and make sure the change is reflected
	// in the other.

	// first, check if the internal annotations are found in nodeAnnosMap
	if indexIdMap, ok := nodeAnnosMap[internal.path]; ok {
		if id, ok := indexIdMap[internal.index]; ok {
			if id == internal.id {
				// the internal annotations of the resource match the ones stored in
				// nodeAnnosMap, so we should copy the legacy annotations to the
				// internal ones
				if err := updateAnnotations(rn, meta,
					[]string{
						kioutil.PathAnnotation,
						kioutil.IndexAnnotation,
						kioutil.IdAnnotation,
					},
					[]string{
						legacy.path,
						legacy.index,
						legacy.id,
					}); err != nil {
					return err
				}
			}
		}
	}

	// check the opposite, to see if the legacy annotations are in nodeAnnosMap
	if indexIdMap, ok := nodeAnnosMap[legacy.path]; ok {
		if id, ok := indexIdMap[legacy.index]; ok {
			if id == legacy.id {
				// the legacy annotations of the resource match the ones stored in
				// nodeAnnosMap, so we should copy the internal annotations to the
				// legacy ones
				if err := updateAnnotations(rn, meta,
					[]string{
						kioutil.LegacyPathAnnotation,
						kioutil.LegacyIndexAnnotation,
						kioutil.LegacyIdAnnotation,
					},
					[]string{
						internal.path,
						internal.index,
						internal.id,
					}); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func updateAnnotations(rn *yaml.RNode, meta yaml.ResourceMeta, keys []string, values []string) error {
	if len(keys) != len(values) {
		return fmt.Errorf("keys is not same length as values")
	}
	for i := range keys {
		_, ok := meta.Annotations[keys[i]]
		if values[i] == "" && !ok {
			// don't set "" if annotation is not already there
			continue
		}
		if err := rn.PipeE(yaml.SetAnnotation(keys[i], values[i])); err != nil {
			return err
		}

	}
	return nil
}
