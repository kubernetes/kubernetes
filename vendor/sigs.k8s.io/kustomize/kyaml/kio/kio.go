// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package kio contains low-level libraries for reading, modifying and writing
// Resource Configuration and packages.
package kio

import (
	"fmt"
	"strconv"

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

// TrackableFilter is an extension of Filter which is also capable of tracking
// which fields were mutated by the filter.
type TrackableFilter interface {
	Filter
	WithMutationTracker(func(key, value, tag string, node *yaml.RNode))
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
	for i := range p.Filters {
		// Not all RNodes passed through kio.Pipeline have metadata nor should
		// they all be required to.
		nodeAnnos, err := PreprocessResourcesForInternalAnnotationMigration(result)
		if err != nil {
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
		err = ReconcileInternalAnnotations(result, nodeAnnos)
		if err != nil {
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

// PreprocessResourcesForInternalAnnotationMigration returns a mapping from id to all
// internal annotations, so that we can use it to reconcile the annotations
// later. This is necessary because currently both internal-prefixed annotations
// and legacy annotations are currently supported, and a change to one must be
// reflected in the other if needed.
func PreprocessResourcesForInternalAnnotationMigration(result []*yaml.RNode) (map[string]map[string]string, error) {
	idToAnnosMap := make(map[string]map[string]string)
	for i := range result {
		idStr := strconv.Itoa(i)
		err := result[i].PipeE(yaml.SetAnnotation(kioutil.InternalAnnotationsMigrationResourceIDAnnotation, idStr))
		if err != nil {
			return nil, err
		}
		idToAnnosMap[idStr] = kioutil.GetInternalAnnotations(result[i])
		if err = kioutil.CopyLegacyAnnotations(result[i]); err != nil {
			return nil, err
		}
		meta, _ := result[i].GetMeta()
		if err = checkMismatchedAnnos(meta.Annotations); err != nil {
			return nil, err
		}
	}
	return idToAnnosMap, nil
}

func checkMismatchedAnnos(annotations map[string]string) error {
	path := annotations[kioutil.PathAnnotation]
	index := annotations[kioutil.IndexAnnotation]
	id := annotations[kioutil.IdAnnotation]

	legacyPath := annotations[kioutil.LegacyPathAnnotation]
	legacyIndex := annotations[kioutil.LegacyIndexAnnotation]
	legacyId := annotations[kioutil.LegacyIdAnnotation]

	// if prior to running the functions, the legacy and internal annotations differ,
	// throw an error as we cannot infer the user's intent.
	if path != "" && legacyPath != "" && path != legacyPath {
		return fmt.Errorf("resource input to function has mismatched legacy and internal path annotations")
	}
	if index != "" && legacyIndex != "" && index != legacyIndex {
		return fmt.Errorf("resource input to function has mismatched legacy and internal index annotations")
	}
	if id != "" && legacyId != "" && id != legacyId {
		return fmt.Errorf("resource input to function has mismatched legacy and internal id annotations")
	}
	return nil
}

type nodeAnnotations struct {
	path  string
	index string
	id    string
}

// ReconcileInternalAnnotations reconciles the annotation format for path, index and id annotations.
// It will ensure the output annotation format matches the format in the input. e.g. if the input
// format uses the legacy format and the output will be converted to the legacy format if it's not.
func ReconcileInternalAnnotations(result []*yaml.RNode, nodeAnnosMap map[string]map[string]string) error {
	useInternal, useLegacy, err := determineAnnotationsFormat(nodeAnnosMap)
	if err != nil {
		return err
	}

	for i := range result {
		// if only one annotation is set, set the other.
		err = missingInternalOrLegacyAnnotations(result[i])
		if err != nil {
			return err
		}
		// we must check to see if the function changed either the new internal annotations
		// or the old legacy annotations. If one is changed, the change must be reflected
		// in the other.
		err = checkAnnotationsAltered(result[i], nodeAnnosMap)
		if err != nil {
			return err
		}
		// We invoke determineAnnotationsFormat to find out if the original annotations
		// use the internal or (and) the legacy format. We format the resources to
		// make them consistent with original format.
		err = formatInternalAnnotations(result[i], useInternal, useLegacy)
		if err != nil {
			return err
		}
		// if the annotations are still somehow out of sync, throw an error
		meta, _ := result[i].GetMeta()
		err = checkMismatchedAnnos(meta.Annotations)
		if err != nil {
			return err
		}

		if _, err = result[i].Pipe(yaml.ClearAnnotation(kioutil.InternalAnnotationsMigrationResourceIDAnnotation)); err != nil {
			return err
		}
	}
	return nil
}

// determineAnnotationsFormat determines if the resources are using one of the internal and legacy annotation format or both of them.
func determineAnnotationsFormat(nodeAnnosMap map[string]map[string]string) (useInternal, useLegacy bool, err error) {
	if len(nodeAnnosMap) == 0 {
		return true, true, nil
	}

	var internal, legacy *bool
	for _, annos := range nodeAnnosMap {
		_, foundPath := annos[kioutil.PathAnnotation]
		_, foundIndex := annos[kioutil.IndexAnnotation]
		_, foundId := annos[kioutil.IdAnnotation]
		_, foundLegacyPath := annos[kioutil.LegacyPathAnnotation]
		_, foundLegacyIndex := annos[kioutil.LegacyIndexAnnotation]
		_, foundLegacyId := annos[kioutil.LegacyIdAnnotation]

		if !(foundPath || foundIndex || foundId || foundLegacyPath || foundLegacyIndex || foundLegacyId) {
			continue
		}

		foundOneOf := foundPath || foundIndex || foundId
		if internal == nil {
			f := foundOneOf
			internal = &f
		}
		if (foundOneOf && !*internal) || (!foundOneOf && *internal) {
			err = fmt.Errorf("the annotation formatting in the input resources is not consistent")
			return
		}

		foundOneOf = foundLegacyPath || foundLegacyIndex || foundLegacyId
		if legacy == nil {
			f := foundOneOf
			legacy = &f
		}
		if (foundOneOf && !*legacy) || (!foundOneOf && *legacy) {
			err = fmt.Errorf("the annotation formatting in the input resources is not consistent")
			return
		}
	}
	if internal != nil {
		useInternal = *internal
	}
	if legacy != nil {
		useLegacy = *legacy
	}
	return
}

func missingInternalOrLegacyAnnotations(rn *yaml.RNode) error {
	if err := missingInternalOrLegacyAnnotation(rn, kioutil.PathAnnotation, kioutil.LegacyPathAnnotation); err != nil {
		return err
	}
	if err := missingInternalOrLegacyAnnotation(rn, kioutil.IndexAnnotation, kioutil.LegacyIndexAnnotation); err != nil {
		return err
	}
	if err := missingInternalOrLegacyAnnotation(rn, kioutil.IdAnnotation, kioutil.LegacyIdAnnotation); err != nil {
		return err
	}
	return nil
}

func missingInternalOrLegacyAnnotation(rn *yaml.RNode, newKey string, legacyKey string) error {
	meta, _ := rn.GetMeta()
	annotations := meta.Annotations
	value := annotations[newKey]
	legacyValue := annotations[legacyKey]

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

func checkAnnotationsAltered(rn *yaml.RNode, nodeAnnosMap map[string]map[string]string) error {
	meta, _ := rn.GetMeta()
	annotations := meta.Annotations
	// get the resource's current path, index, and ids from the new annotations
	internal := nodeAnnotations{
		path:  annotations[kioutil.PathAnnotation],
		index: annotations[kioutil.IndexAnnotation],
		id:    annotations[kioutil.IdAnnotation],
	}

	// get the resource's current path, index, and ids from the legacy annotations
	legacy := nodeAnnotations{
		path:  annotations[kioutil.LegacyPathAnnotation],
		index: annotations[kioutil.LegacyIndexAnnotation],
		id:    annotations[kioutil.LegacyIdAnnotation],
	}

	rid := annotations[kioutil.InternalAnnotationsMigrationResourceIDAnnotation]
	originalAnnotations, found := nodeAnnosMap[rid]
	if !found {
		return nil
	}
	originalPath, found := originalAnnotations[kioutil.PathAnnotation]
	if !found {
		originalPath = originalAnnotations[kioutil.LegacyPathAnnotation]
	}
	if originalPath != "" {
		if originalPath != internal.path && originalPath != legacy.path && internal.path != legacy.path {
			return fmt.Errorf("resource input to function has mismatched legacy and internal path annotations")
		} else if originalPath != internal.path {
			if _, err := rn.Pipe(yaml.SetAnnotation(kioutil.LegacyPathAnnotation, internal.path)); err != nil {
				return err
			}
		} else if originalPath != legacy.path {
			if _, err := rn.Pipe(yaml.SetAnnotation(kioutil.PathAnnotation, legacy.path)); err != nil {
				return err
			}
		}
	}

	originalIndex, found := originalAnnotations[kioutil.IndexAnnotation]
	if !found {
		originalIndex = originalAnnotations[kioutil.LegacyIndexAnnotation]
	}
	if originalIndex != "" {
		if originalIndex != internal.index && originalIndex != legacy.index && internal.index != legacy.index {
			return fmt.Errorf("resource input to function has mismatched legacy and internal index annotations")
		} else if originalIndex != internal.index {
			if _, err := rn.Pipe(yaml.SetAnnotation(kioutil.LegacyIndexAnnotation, internal.index)); err != nil {
				return err
			}
		} else if originalIndex != legacy.index {
			if _, err := rn.Pipe(yaml.SetAnnotation(kioutil.IndexAnnotation, legacy.index)); err != nil {
				return err
			}
		}
	}
	return nil
}

func formatInternalAnnotations(rn *yaml.RNode, useInternal, useLegacy bool) error {
	if !useInternal {
		if err := rn.PipeE(yaml.ClearAnnotation(kioutil.IdAnnotation)); err != nil {
			return err
		}
		if err := rn.PipeE(yaml.ClearAnnotation(kioutil.PathAnnotation)); err != nil {
			return err
		}
		if err := rn.PipeE(yaml.ClearAnnotation(kioutil.IndexAnnotation)); err != nil {
			return err
		}
	}
	if !useLegacy {
		if err := rn.PipeE(yaml.ClearAnnotation(kioutil.LegacyIdAnnotation)); err != nil {
			return err
		}
		if err := rn.PipeE(yaml.ClearAnnotation(kioutil.LegacyPathAnnotation)); err != nil {
			return err
		}
		if err := rn.PipeE(yaml.ClearAnnotation(kioutil.LegacyIndexAnnotation)); err != nil {
			return err
		}
	}
	return nil
}
