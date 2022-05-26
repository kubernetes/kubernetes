/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package git

import (
	"fmt"

	"github.com/pkg/errors"

	"sigs.k8s.io/zeitgeist/internal/command"
)

// DescribeOptions is the type for the argument passed to repo.Describe
type DescribeOptions struct {
	revision string
	abbrev   int16
	always   bool
	dirty    bool
	tags     bool
}

// NewDescribeOptions creates new repository describe options
func NewDescribeOptions() *DescribeOptions {
	return &DescribeOptions{
		revision: "",
		abbrev:   -1,
		always:   false,
		dirty:    false,
		tags:     false,
	}
}

// WithRevision sets the revision in the DescribeOptions
func (d *DescribeOptions) WithRevision(rev string) *DescribeOptions {
	d.revision = rev
	return d
}

// WithAbbrev sets the --abbrev=<parameter> in the DescribeOptions
func (d *DescribeOptions) WithAbbrev(abbrev uint8) *DescribeOptions {
	d.abbrev = int16(abbrev)
	return d
}

// WithAlways sets always to true in the DescribeOptions
func (d *DescribeOptions) WithAlways() *DescribeOptions {
	d.always = true
	return d
}

// WithDirty sets dirty to true in the DescribeOptions
func (d *DescribeOptions) WithDirty() *DescribeOptions {
	d.dirty = true
	return d
}

// WithTags sets tags to true in the DescribeOptions
func (d *DescribeOptions) WithTags() *DescribeOptions {
	d.tags = true
	return d
}

// toArgs converts DescribeOptions to string arguments
func (d *DescribeOptions) toArgs() (args []string) {
	if d.tags {
		args = append(args, "--tags")
	}
	if d.dirty {
		args = append(args, "--dirty")
	}
	if d.always {
		args = append(args, "--always")
	}
	if d.abbrev >= 0 {
		args = append(args, fmt.Sprintf("--abbrev=%d", d.abbrev))
	}
	if d.revision != "" {
		args = append(args, d.revision)
	}
	return args
}

// Describe runs `git describe` with the provided arguments
func (r *Repo) Describe(opts *DescribeOptions) (string, error) {
	if opts == nil {
		return "", errors.New("provided describe options are nil")
	}
	output, err := command.NewWithWorkDir(
		r.Dir(), gitExecutable, append([]string{"describe"}, opts.toArgs()...)...,
	).RunSilentSuccessOutput()
	if err != nil {
		return "", err
	}
	return output.OutputTrimNL(), nil
}
