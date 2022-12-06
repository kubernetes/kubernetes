// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package starlark

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"

	"go.starlark.net/starlark"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/fn/runtime/runtimeutil"
	"sigs.k8s.io/kustomize/kyaml/internal/forked/github.com/qri-io/starlib/util"
	"sigs.k8s.io/kustomize/kyaml/kio/filters"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Filter transforms a set of resources through the provided program
type Filter struct {
	Name string

	// Program is a starlark script which will be run against the resources
	Program string

	// URL is the url of a starlark program to fetch and run
	URL string

	// Path is the path to a starlark program to read and run
	Path string

	runtimeutil.FunctionFilter
}

func (sf *Filter) String() string {
	return fmt.Sprintf(
		"name: %v path: %v url: %v program: %v", sf.Name, sf.Path, sf.URL, sf.Program)
}

func (sf *Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	if err := sf.setup(); err != nil {
		return nil, err
	}
	sf.FunctionFilter.Run = sf.Run

	return sf.FunctionFilter.Filter(nodes)
}

func (sf *Filter) setup() error {
	if (sf.URL != "" && sf.Path != "") ||
		(sf.URL != "" && sf.Program != "") ||
		(sf.Path != "" && sf.Program != "") {
		return errors.Errorf("Filter Path, Program and URL are mutually exclusive")
	}

	// read the program from a file
	if sf.Path != "" {
		b, err := ioutil.ReadFile(sf.Path)
		if err != nil {
			return err
		}
		sf.Program = string(b)
	}

	// read the program from a URL
	if sf.URL != "" {
		err := func() error {
			resp, err := http.Get(sf.URL)
			if err != nil {
				return err
			}
			defer resp.Body.Close()
			b, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				return err
			}
			sf.Program = string(b)
			return nil
		}()
		if err != nil {
			return err
		}
	}

	return nil
}

func (sf *Filter) Run(reader io.Reader, writer io.Writer) error {
	// retain map of inputs to outputs by id so if the name is changed by the
	// starlark program, we are able to match the same resources
	value, err := sf.readResourceList(reader)
	if err != nil {
		return errors.Wrap(err)
	}

	// run the starlark as program as transformation function
	thread := &starlark.Thread{Name: sf.Name}

	ctx := &Context{resourceList: value}
	pd, err := ctx.predeclared()
	if err != nil {
		return errors.Wrap(err)
	}
	_, err = starlark.ExecFile(thread, sf.Name, sf.Program, pd)
	if err != nil {
		return errors.Wrap(err)
	}

	return sf.writeResourceList(value, writer)
}

// inputToResourceList transforms input into a starlark.Value
func (sf *Filter) readResourceList(reader io.Reader) (starlark.Value, error) {
	// read and parse the inputs
	rl := bytes.Buffer{}
	_, err := rl.ReadFrom(reader)
	if err != nil {
		return nil, errors.Wrap(err)
	}
	rn, err := yaml.Parse(rl.String())
	if err != nil {
		return nil, errors.Wrap(err)
	}

	// convert to a starlark value
	b, err := yaml.Marshal(rn.Document()) // convert to bytes
	if err != nil {
		return nil, errors.Wrap(err)
	}
	var in map[string]interface{}
	err = yaml.Unmarshal(b, &in) // convert to map[string]interface{}
	if err != nil {
		return nil, errors.Wrap(err)
	}
	return util.Marshal(in) // convert to starlark value
}

// resourceListToOutput converts the output of the starlark program to the filter output
func (sf *Filter) writeResourceList(value starlark.Value, writer io.Writer) error {
	// convert the modified resourceList back into a slice of RNodes
	// by first converting to a map[string]interface{}
	out, err := util.Unmarshal(value)
	if err != nil {
		return errors.Wrap(err)
	}
	b, err := yaml.Marshal(out)
	if err != nil {
		return errors.Wrap(err)
	}

	rl, err := yaml.Parse(string(b))
	if err != nil {
		return errors.Wrap(err)
	}

	// preserve the comments from the input
	items, err := rl.Pipe(yaml.Lookup("items"))
	if err != nil {
		return errors.Wrap(err)
	}
	err = items.VisitElements(func(node *yaml.RNode) error {
		// starlark will serialize the resources sorting the fields alphabetically,
		// format them to have a better ordering
		_, err := filters.FormatFilter{}.Filter([]*yaml.RNode{node})
		return err
	})
	if err != nil {
		return errors.Wrap(err)
	}

	s, err := rl.String()
	if err != nil {
		return errors.Wrap(err)
	}

	_, err = writer.Write([]byte(s))
	return err
}
