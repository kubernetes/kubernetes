/*
Copyright 2015 The Kubernetes Authors.

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

package create

import (
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/generate"
	generateversioned "k8s.io/kubectl/pkg/generate/versioned"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	namespaceLong = templates.LongDesc(i18n.T(`
		Create a namespace with the specified name.`))

	namespaceExample = templates.Examples(i18n.T(`
	  # Create a new namespace named my-namespace
	  kubectl create namespace my-namespace`))
)

// NamespaceOpts is the options for 'create namespace' sub command
type NamespaceOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateNamespace is a macro command to create a new namespace
func NewCmdCreateNamespace(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &NamespaceOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use:                   "namespace NAME [--dry-run]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"ns"},
		Short:                 i18n.T("Create a namespace with the specified name"),
		Long:                  namespaceLong,
		Example:               namespaceExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, generateversioned.NamespaceV1GeneratorName)

	return cmd
}

// Complete completes all the required options
func (o *NamespaceOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	var generator generate.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case generateversioned.NamespaceV1GeneratorName:
		generator = &generateversioned.NamespaceGeneratorV1{Name: name}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// Run calls the CreateSubcommandOptions.Run in NamespaceOpts instance
func (o *NamespaceOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
