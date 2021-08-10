// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package runtimeutil

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"path"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/comments"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/kio/kioutil"

	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// FunctionFilter wraps another filter to be invoked in the context of a function.
// FunctionFilter manages scoping the function, deferring failures, and saving results
// to files.
type FunctionFilter struct {
	// Run implements the function.
	Run func(reader io.Reader, writer io.Writer) error

	// FunctionConfig is passed to the function through ResourceList.functionConfig.
	FunctionConfig *yaml.RNode `yaml:"functionConfig,omitempty"`

	// GlobalScope explicitly scopes the function to all input resources rather than only those
	// resources scoped to it by path.
	GlobalScope bool

	// ResultsFile is the file to write function ResourceList.results to.
	// If unset, results will not be written.
	ResultsFile string

	// DeferFailure will cause the Filter to return a nil error even if Run returns an error.
	// The Run error will be available through GetExit().
	DeferFailure bool

	// results saves the results emitted from Run
	Results *yaml.RNode

	// exit saves the error returned from Run
	exit error

	ids map[string]*yaml.RNode
}

// GetExit returns the error from Run
func (c FunctionFilter) GetExit() error {
	return c.exit
}

// functionsDirectoryName is keyword directory name for functions scoped 1 directory higher
const functionsDirectoryName = "functions"

// getFunctionScope returns the path of the directory containing the function config,
// or its parent directory if the base directory is named "functions"
func (c *FunctionFilter) getFunctionScope() (string, error) {
	m, err := c.FunctionConfig.GetMeta()
	if err != nil {
		return "", errors.Wrap(err)
	}
	p, found := m.Annotations[kioutil.PathAnnotation]
	if !found {
		return "", nil
	}

	functionDir := path.Clean(path.Dir(p))

	if path.Base(functionDir) == functionsDirectoryName {
		// the scope of functions in a directory called "functions" is 1 level higher
		// this is similar to how the golang "internal" directory scoping works
		functionDir = path.Dir(functionDir)
	}
	return functionDir, nil
}

// scope partitions the input nodes into 2 slices.  The first slice contains only Resources
// which are scoped under dir, and the second slice contains the Resources which are not.
func (c *FunctionFilter) scope(dir string, nodes []*yaml.RNode) ([]*yaml.RNode, []*yaml.RNode, error) {
	// scope container filtered Resources to Resources under that directory
	var input, saved []*yaml.RNode
	if c.GlobalScope {
		return nodes, nil, nil
	}

	// global function
	if dir == "" || dir == "." {
		return nodes, nil, nil
	}

	// identify Resources read from directories under the function configuration
	for i := range nodes {
		m, err := nodes[i].GetMeta()
		if err != nil {
			return nil, nil, err
		}
		p, found := m.Annotations[kioutil.PathAnnotation]
		if !found {
			// this Resource isn't scoped under the function -- don't know where it came from
			// consider it out of scope
			saved = append(saved, nodes[i])
			continue
		}

		resourceDir := path.Clean(path.Dir(p))
		if path.Base(resourceDir) == functionsDirectoryName {
			// Functions in the `functions` directory are scoped to
			// themselves, and should see themselves as input
			resourceDir = path.Dir(resourceDir)
		}
		if !strings.HasPrefix(resourceDir, dir) {
			// this Resource doesn't fall under the function scope if it
			// isn't in a subdirectory of where the function lives
			saved = append(saved, nodes[i])
			continue
		}

		// this input is scoped under the function
		input = append(input, nodes[i])
	}

	return input, saved, nil
}

func (c *FunctionFilter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	in := &bytes.Buffer{}
	out := &bytes.Buffer{}

	// only process Resources scoped to this function, save the others
	functionDir, err := c.getFunctionScope()
	if err != nil {
		return nil, err
	}
	input, saved, err := c.scope(functionDir, nodes)
	if err != nil {
		return nil, err
	}

	// set ids on each input so it is possible to copy comments from inputs back to outputs
	if err := c.setIds(input); err != nil {
		return nil, err
	}

	// write the input
	err = kio.ByteWriter{
		WrappingAPIVersion:    kio.ResourceListAPIVersion,
		WrappingKind:          kio.ResourceListKind,
		Writer:                in,
		KeepReaderAnnotations: true,
		FunctionConfig:        c.FunctionConfig}.Write(input)
	if err != nil {
		return nil, err
	}

	// capture the command stdout for the return value
	r := &kio.ByteReader{Reader: out}

	// don't exit immediately if the function fails -- write out the validation
	c.exit = c.Run(in, out)

	output, err := r.Read()
	if err != nil {
		return nil, err
	}

	// copy the comments from the inputs to the outputs
	if err := c.setComments(output); err != nil {
		return nil, err
	}

	if err := c.doResults(r); err != nil {
		return nil, err
	}

	if c.exit != nil && !c.DeferFailure {
		return append(output, saved...), c.exit
	}

	// annotate any generated Resources with a path and index if they don't already have one
	if err := kioutil.DefaultPathAnnotation(functionDir, output); err != nil {
		return nil, err
	}

	// emit both the Resources output from the function, and the out-of-scope Resources
	// which were not provided to the function
	return append(output, saved...), nil
}

const idAnnotation = "config.k8s.io/id"

func (c *FunctionFilter) setIds(nodes []*yaml.RNode) error {
	// set the id on each node to map inputs to outputs
	var id int
	c.ids = map[string]*yaml.RNode{}
	for i := range nodes {
		id++
		idStr := fmt.Sprintf("%v", id)
		err := nodes[i].PipeE(yaml.SetAnnotation(idAnnotation, idStr))
		if err != nil {
			return errors.Wrap(err)
		}
		c.ids[idStr] = nodes[i]
	}
	return nil
}

func (c *FunctionFilter) setComments(nodes []*yaml.RNode) error {
	for i := range nodes {
		node := nodes[i]
		anID, err := node.Pipe(yaml.GetAnnotation(idAnnotation))
		if err != nil {
			return errors.Wrap(err)
		}
		if anID == nil {
			continue
		}

		var in *yaml.RNode
		var found bool
		if in, found = c.ids[anID.YNode().Value]; !found {
			continue
		}
		if err := comments.CopyComments(in, node); err != nil {
			return errors.Wrap(err)
		}
		if err := node.PipeE(yaml.ClearAnnotation(idAnnotation)); err != nil {
			return errors.Wrap(err)
		}
	}
	return nil
}

func (c *FunctionFilter) doResults(r *kio.ByteReader) error {
	// Write the results to a file if configured to do so
	if c.ResultsFile != "" && r.Results != nil {
		results, err := r.Results.String()
		if err != nil {
			return err
		}
		err = ioutil.WriteFile(c.ResultsFile, []byte(results), 0600)
		if err != nil {
			return err
		}
	}

	if r.Results != nil {
		c.Results = r.Results
	}
	return nil
}
