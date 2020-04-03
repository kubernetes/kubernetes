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

package delete

import (
	"time"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/dynamic"
)

// DeleteFlags composes common printer flag structs
// used for commands requiring deletion logic.
type DeleteFlags struct {
	FileNameFlags *genericclioptions.FileNameFlags
	LabelSelector *string
	FieldSelector *string

	All            *bool
	AllNamespaces  *bool
	Cascade        *bool
	Force          *bool
	GracePeriod    *int
	IgnoreNotFound *bool
	Now            *bool
	Timeout        *time.Duration
	Wait           *bool
	Output         *string
	Raw            *string
}

func (f *DeleteFlags) ToOptions(dynamicClient dynamic.Interface, streams genericclioptions.IOStreams) *DeleteOptions {
	options := &DeleteOptions{
		DynamicClient: dynamicClient,
		IOStreams:     streams,
	}

	// add filename options
	if f.FileNameFlags != nil {
		options.FilenameOptions = f.FileNameFlags.ToOptions()
	}
	if f.LabelSelector != nil {
		options.LabelSelector = *f.LabelSelector
	}
	if f.FieldSelector != nil {
		options.FieldSelector = *f.FieldSelector
	}

	// add output format
	if f.Output != nil {
		options.Output = *f.Output
	}

	if f.All != nil {
		options.DeleteAll = *f.All
	}
	if f.AllNamespaces != nil {
		options.DeleteAllNamespaces = *f.AllNamespaces
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
	if f.Wait != nil {
		options.WaitForDeletion = *f.Wait
	}
	if f.Raw != nil {
		options.Raw = *f.Raw
	}

	return options
}

func (f *DeleteFlags) AddFlags(cmd *cobra.Command) {
	f.FileNameFlags.AddFlags(cmd.Flags())
	if f.LabelSelector != nil {
		cmd.Flags().StringVarP(f.LabelSelector, "selector", "l", *f.LabelSelector, "Selector (label query) to filter on, not including uninitialized ones.")
	}
	if f.FieldSelector != nil {
		cmd.Flags().StringVarP(f.FieldSelector, "field-selector", "", *f.FieldSelector, "Selector (field query) to filter on, supports '=', '==', and '!='.(e.g. --field-selector key1=value1,key2=value2). The server only supports a limited number of field queries per type.")
	}
	if f.All != nil {
		cmd.Flags().BoolVar(f.All, "all", *f.All, "Delete all resources, including uninitialized ones, in the namespace of the specified resource types.")
	}
	if f.AllNamespaces != nil {
		cmd.Flags().BoolVarP(f.AllNamespaces, "all-namespaces", "A", *f.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	}
	if f.Force != nil {
		cmd.Flags().BoolVar(f.Force, "force", *f.Force, "If true, immediately remove resources from API and bypass graceful deletion. Note that immediate deletion of some resources may result in inconsistency or data loss and requires confirmation.")
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
	if f.Wait != nil {
		cmd.Flags().BoolVar(f.Wait, "wait", *f.Wait, "If true, wait for resources to be gone before returning. This waits for finalizers.")
	}
	if f.Output != nil {
		cmd.Flags().StringVarP(f.Output, "output", "o", *f.Output, "Output mode. Use \"-o name\" for shorter output (resource/name).")
	}
	if f.Raw != nil {
		cmd.Flags().StringVar(f.Raw, "raw", *f.Raw, "Raw URI to DELETE to the server.  Uses the transport specified by the kubeconfig file.")
	}
}

// NewDeleteCommandFlags provides default flags and values for use with the "delete" command
func NewDeleteCommandFlags(usage string) *DeleteFlags {
	cascade := true
	gracePeriod := -1

	// setup command defaults
	all := false
	allNamespaces := false
	force := false
	ignoreNotFound := false
	now := false
	output := ""
	labelSelector := ""
	fieldSelector := ""
	timeout := time.Duration(0)
	wait := true
	raw := ""

	filenames := []string{}
	recursive := false
	kustomize := ""

	return &DeleteFlags{
		// Not using helpers.go since it provides function to add '-k' for FileNameOptions, but not FileNameFlags
		FileNameFlags: &genericclioptions.FileNameFlags{Usage: usage, Filenames: &filenames, Kustomize: &kustomize, Recursive: &recursive},
		LabelSelector: &labelSelector,
		FieldSelector: &fieldSelector,

		Cascade:     &cascade,
		GracePeriod: &gracePeriod,

		All:            &all,
		AllNamespaces:  &allNamespaces,
		Force:          &force,
		IgnoreNotFound: &ignoreNotFound,
		Now:            &now,
		Timeout:        &timeout,
		Wait:           &wait,
		Output:         &output,
		Raw:            &raw,
	}
}

// NewDeleteFlags provides default flags and values for use in commands outside of "delete"
func NewDeleteFlags(usage string) *DeleteFlags {
	cascade := true
	gracePeriod := -1

	force := false
	timeout := time.Duration(0)
	wait := false

	filenames := []string{}
	kustomize := ""
	recursive := false

	return &DeleteFlags{
		FileNameFlags: &genericclioptions.FileNameFlags{Usage: usage, Filenames: &filenames, Kustomize: &kustomize, Recursive: &recursive},

		Cascade:     &cascade,
		GracePeriod: &gracePeriod,

		// add non-defaults
		Force:   &force,
		Timeout: &timeout,
		Wait:    &wait,
	}
}
