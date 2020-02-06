/*
Copyright 2017 The Kubernetes Authors.

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

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/generate"
	generateversioned "k8s.io/kubectl/pkg/generate/versioned"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	pcLong = templates.LongDesc(i18n.T(`
		Create a priorityclass with the specified name, value, globalDefault and description`))

	pcExample = templates.Examples(i18n.T(`
		# Create a priorityclass named high-priority
		kubectl create priorityclass high-priority --value=1000 --description="high priority"

		# Create a priorityclass named default-priority that considered as the global default priority
		kubectl create priorityclass default-priority --value=1000 --global-default=true --description="default priority"
		
		# Create a priorityclass named high-priority that can not preempt pods with lower priority
		kubectl create priorityclass high-priority --value=1000 --description="high priority" --preemption-policy="Never"`))
)

// PriorityClassOpts holds the options for 'create priorityclass' sub command
type PriorityClassOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreatePriorityClass is a macro command to create a new priorityClass.
func NewCmdCreatePriorityClass(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &PriorityClassOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use:                   "priorityclass NAME --value=VALUE --global-default=BOOL [--dry-run]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"pc"},
		Short:                 i18n.T("Create a priorityclass with the specified name."),
		Long:                  pcLong,
		Example:               pcExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, generateversioned.PriorityClassV1GeneratorName)

	cmd.Flags().Int32("value", 0, i18n.T("the value of this priority class."))
	cmd.Flags().Bool("global-default", false, i18n.T("global-default specifies whether this PriorityClass should be considered as the default priority."))
	cmd.Flags().String("description", "", i18n.T("description is an arbitrary string that usually provides guidelines on when this priority class should be used."))
	cmd.Flags().String("preemption-policy", "", i18n.T("preemption-policy is the policy for preempting pods with lower priority."))
	return cmd
}

// Complete completes all the required options
func (o *PriorityClassOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	var generator generate.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case generateversioned.PriorityClassV1GeneratorName:
		generator = &generateversioned.PriorityClassV1Generator{
			Name:             name,
			Value:            cmdutil.GetFlagInt32(cmd, "value"),
			GlobalDefault:    cmdutil.GetFlagBool(cmd, "global-default"),
			Description:      cmdutil.GetFlagString(cmd, "description"),
			PreemptionPolicy: apiv1.PreemptionPolicy(cmdutil.GetFlagString(cmd, "preemption-policy")),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// Run calls the CreateSubcommandOptions.Run in the PriorityClassOpts instance
func (o *PriorityClassOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
