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
	quotaLong = templates.LongDesc(i18n.T(`
		Create a resourcequota with the specified name, hard limits and optional scopes`))

	quotaExample = templates.Examples(i18n.T(`
		# Create a new resourcequota named my-quota
		kubectl create quota my-quota --hard=cpu=1,memory=1G,pods=2,services=3,replicationcontrollers=2,resourcequotas=1,secrets=5,persistentvolumeclaims=10

		# Create a new resourcequota named best-effort
		kubectl create quota best-effort --hard=pods=100 --scopes=BestEffort`))
)

// QuotaOpts holds the command-line options for 'create quota' sub command
type QuotaOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateQuota is a macro command to create a new quota
func NewCmdCreateQuota(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &QuotaOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use:                   "quota NAME [--hard=key1=value1,key2=value2] [--scopes=Scope1,Scope2] [--dry-run=bool]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"resourcequota"},
		Short:                 i18n.T("Create a quota with the specified name."),
		Long:                  quotaLong,
		Example:               quotaExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, generateversioned.ResourceQuotaV1GeneratorName)
	cmd.Flags().String("hard", "", i18n.T("A comma-delimited set of resource=quantity pairs that define a hard limit."))
	cmd.Flags().String("scopes", "", i18n.T("A comma-delimited set of quota scopes that must all match each object tracked by the quota."))
	return cmd
}

// Complete completes all the required options
func (o *QuotaOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	var generator generate.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case generateversioned.ResourceQuotaV1GeneratorName:
		generator = &generateversioned.ResourceQuotaGeneratorV1{
			Name:   name,
			Hard:   cmdutil.GetFlagString(cmd, "hard"),
			Scopes: cmdutil.GetFlagString(cmd, "scopes"),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// Run calls the CreateSubcommandOptions.Run in QuotaOpts instance
func (o *QuotaOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
