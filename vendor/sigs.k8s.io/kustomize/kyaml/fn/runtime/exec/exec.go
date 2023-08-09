// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package exec

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/fn/runtime/runtimeutil"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type Filter struct {
	// Path is the path to the executable to run
	Path string `yaml:"path,omitempty"`

	// Args are the arguments to the executable
	Args []string `yaml:"args,omitempty"`

	// WorkingDir is the working directory that the executable
	// should run in
	WorkingDir string

	runtimeutil.FunctionFilter
}

func (c *Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	c.FunctionFilter.Run = c.Run
	return c.FunctionFilter.Filter(nodes)
}

func (c *Filter) Run(reader io.Reader, writer io.Writer) error {
	cmd := exec.Command(c.Path, c.Args...) // nolint:gosec
	cmd.Stdin = reader
	cmd.Stdout = writer
	cmd.Stderr = os.Stderr
	if c.WorkingDir == "" {
		return errors.Errorf("no working directory set for exec function")
	}
	if !filepath.IsAbs(c.WorkingDir) {
		return errors.Errorf(
			"relative working directory %s not allowed", c.WorkingDir)
	}
	if c.WorkingDir == "/" {
		return errors.Errorf(
			"root working directory '/' not allowed")
	}
	cmd.Dir = c.WorkingDir
	return cmd.Run()
}
