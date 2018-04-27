/*
Copyright 2018 The Kubernetes Authors.

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

package cmd

import (
	"io"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

type FileNameFlags struct {
	Usage string

	Filenames *[]string
	Recursive *bool
}

func (o *FileNameFlags) ToOptions() resource.FilenameOptions {
	options := resource.FilenameOptions{}

	if o.Recursive != nil {
		options.Recursive = *o.Recursive
	}
	if o.Filenames != nil {
		options.Filenames = *o.Filenames
	}

	return options
}

func (o *FileNameFlags) AddFlags(cmd *cobra.Command) {
	if o.Recursive != nil {
		cmd.Flags().BoolVarP(o.Recursive, "recursive", "R", *o.Recursive, "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory.")
	}
	if o.Filenames != nil {
		kubectl.AddJsonFilenameFlag(cmd, o.Filenames, "Filename, directory, or URL to files "+o.Usage)
	}
}

// PrintFlags composes common printer flag structs
// used for commands requiring deletion logic.
type DeleteFlags struct {
	FileNameFlags *FileNameFlags

	All            *bool
	Cascade        *bool
	Force          *bool
	GracePeriod    *int
	IgnoreNotFound *bool
	Now            *bool
	Timeout        *time.Duration
	Output         *string
}

func (f *DeleteFlags) ToOptions(out, errOut io.Writer) *DeleteOptions {
	options := &DeleteOptions{
		Out:    out,
		ErrOut: errOut,
	}

	// add filename options
	if f.FileNameFlags != nil {
		options.FilenameOptions = f.FileNameFlags.ToOptions()
	}

	// add output format
	if f.Output != nil {
		options.Output = *f.Output
	}

	if f.All != nil {
		options.DeleteAll = *f.All
	}
	if f.Cascade != nil {
		options.Cascade = *f.Cascade
	}
	if f.Force != nil {
		options.ForceDeletion = *f.Force
	}
	if f.GracePeriod != nil {
		options.GracePeriod = *f.GracePeriod
	}
	if f.IgnoreNotFound != nil {
		options.IgnoreNotFound = *f.IgnoreNotFound
	}
	if f.Now != nil {
		options.DeleteNow = *f.Now
	}
	if f.Timeout != nil {
		options.Timeout = *f.Timeout
	}

	return options
}

func (f *DeleteFlags) AddFlags(cmd *cobra.Command) {
	f.FileNameFlags.AddFlags(cmd)

	if f.All != nil {
		cmd.Flags().BoolVar(f.All, "all", *f.All, "Delete all resources, including uninitialized ones, in the namespace of the specified resource types.")
	}
	if f.Force != nil {
		cmd.Flags().BoolVar(f.Force, "force", *f.Force, "Only used when grace-period=0. If true, immediately remove resources from API and bypass graceful deletion. Note that immediate deletion of some resources may result in inconsistency or data loss and requires confirmation.")
	}
	if f.Cascade != nil {
		cmd.Flags().BoolVar(f.Cascade, "cascade", *f.Cascade, "If true, cascade the deletion of the resources managed by this resource (e.g. Pods created by a ReplicationController).  Default true.")
	}
	if f.Now != nil {
		cmd.Flags().BoolVar(f.Now, "now", *f.Now, "If true, resources are signaled for immediate shutdown (same as --grace-period=1).")
	}
	if f.GracePeriod != nil {
		cmd.Flags().IntVar(f.GracePeriod, "grace-period", *f.GracePeriod, "Period of time in seconds given to the resource to terminate gracefully. Ignored if negative. Set to 1 for immediate shutdown. Can only be set to 0 when --force is true (force deletion).")
	}
	if f.Timeout != nil {
		cmd.Flags().DurationVar(f.Timeout, "timeout", *f.Timeout, "The length of time to wait before giving up on a delete, zero means determine a timeout from the size of the object")
	}
	if f.IgnoreNotFound != nil {
		cmd.Flags().BoolVar(f.IgnoreNotFound, "ignore-not-found", *f.IgnoreNotFound, "Treat \"resource not found\" as a successful delete. Defaults to \"true\" when --all is specified.")
	}

	if f.Output != nil {
		cmd.Flags().StringVarP(f.Output, "output", "o", *f.Output, "Output mode. Use \"-o name\" for shorter output (resource/name).")
	}

}

// NewDeleteCommandFlags provides default flags and values for use with the "delete" command
func NewDeleteCommandFlags(usage string) *DeleteFlags {
	cascade := true
	gracePeriod := -1

	// setup command defaults
	all := false
	force := false
	ignoreNotFound := false
	now := false
	output := ""
	timeout := time.Duration(0)

	filenames := []string{}
	recursive := false

	return &DeleteFlags{
		FileNameFlags: &FileNameFlags{Usage: usage, Filenames: &filenames, Recursive: &recursive},

		Cascade:     &cascade,
		GracePeriod: &gracePeriod,

		All:            &all,
		Force:          &force,
		IgnoreNotFound: &ignoreNotFound,
		Now:            &now,
		Timeout:        &timeout,
		Output:         &output,
	}
}

// NewDeleteFlags provides default flags and values for use in commands outside of "delete"
func NewDeleteFlags(usage string) *DeleteFlags {
	cascade := true
	gracePeriod := -1

	force := false
	timeout := time.Duration(0)

	filenames := []string{}
	recursive := false

	return &DeleteFlags{
		FileNameFlags: &FileNameFlags{Usage: usage, Filenames: &filenames, Recursive: &recursive},

		Cascade:     &cascade,
		GracePeriod: &gracePeriod,

		// add non-defaults
		Force:   &force,
		Timeout: &timeout,
	}
}
