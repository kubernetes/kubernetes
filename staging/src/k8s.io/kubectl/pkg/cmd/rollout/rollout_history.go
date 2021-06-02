/*
Copyright 2016 The Kubernetes Authors.

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

package rollout

import (
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	historyLong = templates.LongDesc(i18n.T(`
		View previous rollout revisions and configurations.`))

	historyExample = templates.Examples(`
		# View the rollout history of a deployment
		kubectl rollout history deployment/abc

		# View the details of daemonset revision 3
		kubectl rollout history daemonset/abc --revision=3`)
)

// RolloutHistoryOptions holds the options for 'rollout history' sub command
type RolloutHistoryOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	Revision int64

	Builder          func() *resource.Builder
	Resources        []string
	Namespace        string
	EnforceNamespace bool

	HistoryViewer    polymorphichelpers.HistoryViewerFunc
	RESTClientGetter genericclioptions.RESTClientGetter

	resource.FilenameOptions
	genericclioptions.IOStreams
}

// NewRolloutHistoryOptions returns an initialized RolloutHistoryOptions instance
func NewRolloutHistoryOptions(streams genericclioptions.IOStreams) *RolloutHistoryOptions {
	return &RolloutHistoryOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme),
		IOStreams:  streams,
	}
}

// NewCmdRolloutHistory returns a Command instance for RolloutHistory sub command
func NewCmdRolloutHistory(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewRolloutHistoryOptions(streams)

	validArgs := []string{"deployment", "daemonset", "statefulset"}

	cmd := &cobra.Command{
		Use:                   "history (TYPE NAME | TYPE/NAME) [flags]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("View rollout history"),
		Long:                  historyLong,
		Example:               historyExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
		ValidArgs: validArgs,
	}

	cmd.Flags().Int64Var(&o.Revision, "revision", o.Revision, "See the details, including podTemplate of the revision specified")

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)

	o.PrintFlags.AddFlags(cmd)

	return cmd
}

// Complete completes al the required options
func (o *RolloutHistoryOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Resources = args

	var err error
	if o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace(); err != nil {
		return err
	}

	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		return o.PrintFlags.ToPrinter()
	}

	o.HistoryViewer = polymorphichelpers.HistoryViewerFn
	o.RESTClientGetter = f
	o.Builder = f.NewBuilder

	return nil
}

// Validate makes sure all the provided values for command-line options are valid
func (o *RolloutHistoryOptions) Validate() error {
	if len(o.Resources) == 0 && cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		return fmt.Errorf("required resource not specified")
	}
	if o.Revision < 0 {
		return fmt.Errorf("revision must be a positive integer: %v", o.Revision)
	}

	return nil
}

// Run performs the execution of 'rollout history' sub command
func (o *RolloutHistoryOptions) Run() error {

	r := o.Builder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(true, o.Resources...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		mapping := info.ResourceMapping()
		historyViewer, err := o.HistoryViewer(o.RESTClientGetter, mapping)
		if err != nil {
			return err
		}
		historyInfo, err := historyViewer.ViewHistory(info.Namespace, info.Name, o.Revision)
		if err != nil {
			return err
		}

		withRevision := ""
		if o.Revision > 0 {
			withRevision = fmt.Sprintf("with revision #%d", o.Revision)
		}

		printer, err := o.ToPrinter(fmt.Sprintf("%s\n%s", withRevision, historyInfo))
		if err != nil {
			return err
		}

		return printer.PrintObj(info.Object, o.Out)
	})
}
