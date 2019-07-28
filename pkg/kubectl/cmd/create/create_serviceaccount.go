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
	serviceAccountLong = templates.LongDesc(i18n.T(`
		Create a service account with the specified name.`))

	serviceAccountExample = templates.Examples(i18n.T(`
	  # Create a new service account named my-service-account
	  kubectl create serviceaccount my-service-account`))
)

// ServiceAccountOpts holds the options for 'create serviceaccount' sub command
type ServiceAccountOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateServiceAccount is a macro command to create a new service account
func NewCmdCreateServiceAccount(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &ServiceAccountOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use:                   "serviceaccount NAME [--dry-run]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"sa"},
		Short:                 i18n.T("Create a service account with the specified name"),
		Long:                  serviceAccountLong,
		Example:               serviceAccountExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, generateversioned.ServiceAccountV1GeneratorName)
	return cmd
}

// Complete completes all the required options
func (o *ServiceAccountOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	var generator generate.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case generateversioned.ServiceAccountV1GeneratorName:
		generator = &generateversioned.ServiceAccountGeneratorV1{Name: name}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// Run calls the CreateSubcommandOptions.Run in ServiceAccountOpts instance
func (o *ServiceAccountOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
