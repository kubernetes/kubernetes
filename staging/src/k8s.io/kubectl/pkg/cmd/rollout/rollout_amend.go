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
	amendLong = templates.LongDesc(i18n.T(`
		Amend previous rollout revision change cause.`))

	amendExample = templates.Examples(`
		# Amend the revision's change cause of a deployment
		kubectl rollout amend deployment/abc "new change cause"

		# Amend the revision 3's change cause of daemonset
		kubectl rollout amend daemonset/abc --revision=3 "new change cause"`)
)

// RolloutAmendOptions holds the options for 'rollout amend' sub command
type RolloutAmendOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	Revision int64

	Builder          func() *resource.Builder
	Resources        []string
	Namespace        string
	EnforceNamespace bool

	AmendViewer    polymorphichelpers.AmendViewerFunc
	RESTClientGetter genericclioptions.RESTClientGetter

	resource.FilenameOptions
	genericclioptions.IOStreams
}

// NewRolloutAmendOptions returns an initialized RolloutAmendOptions instance
func NewRolloutAmendOptions(streams genericclioptions.IOStreams) *RolloutAmendOptions {
	return &RolloutAmendOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme),
		IOStreams:  streams,
	}
}

// NewCmdRolloutAmend returns a Command instance for RolloutAmend sub command
func NewCmdRolloutAmend(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewRolloutAmendOptions(streams)

	validArgs := []string{"deployment", "daemonset", "statefulset"}

	cmd := &cobra.Command{
		Use:                   "amend (TYPE NAME | TYPE/NAME) [flags]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Amend revision change cause"),
		Long:                  amendLong,
		Example:               amendExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
		ValidArgs: validArgs,
	}

	cmd.Flags().Int64Var(&o.Revision, "revision", o.Revision, "Amend specified revision change cause, by default change latest revision's change cause.")

	usage := "identifying the resource to amend from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)

	o.PrintFlags.AddFlags(cmd)

	return cmd
}

// Complete completes al the required options
func (o *RolloutAmendOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Resources = args

	var err error
	if o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace(); err != nil {
		return err
	}

	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		return o.PrintFlags.ToPrinter()
	}

	o.AmendViewer = polymorphichelpers.AmendViewerFn
	o.RESTClientGetter = f
	o.Builder = f.NewBuilder

	return nil
}

// Validate makes sure all the provided values for command-line options are valid
func (o *RolloutAmendOptions) Validate() error {
	if len(o.Resources) == 0 && cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		return fmt.Errorf("required resource not specified")
	}
	if o.Revision <= 0 {
		return fmt.Errorf("revision must be a positive integer: %v", o.Revision)
	}

	return nil
}

// Run performs the execution of 'rollout amend' sub command
func (o *RolloutAmendOptions) Run() error {

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
		amendViewer, err := o.AmendViewer(o.RESTClientGetter, mapping)
		if err != nil {
			return err
		}
		amendInfo, err := amendViewer.ViewAmend(info.Namespace, info.Name, o.Revision)
		if err != nil {
			return err
		}

		withRevision = fmt.Sprintf("with revision #%d", o.Revision)

		printer, err := o.ToPrinter(fmt.Sprintf("%s\n%s", withRevision, amendInfo))
		if err != nil {
			return err
		}

		return printer.PrintObj(info.Object, o.Out)
	})
}
