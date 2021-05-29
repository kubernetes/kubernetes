// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package kio contains low-level libraries for reading, modifying and writing
// Resource Configuration and packages.
package kio

import (
	"sigs.k8s.io/kustomize/kyaml/errors"
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
