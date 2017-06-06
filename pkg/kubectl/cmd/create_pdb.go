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

package cmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	pdbLong = templates.LongDesc(i18n.T(`
		Create a pod disruption budget with the specified name, selector, and desired minimum available pods`))

	pdbExample = templates.Examples(i18n.T(`
		# Create a pod disruption budget named my-pdb that will select all pods with the app=rails label
		# and require at least one of them being available at any point in time.
		kubectl create poddisruptionbudget my-pdb --selector=app=rails --min-available=1

		# Create a pod disruption budget named my-pdb that will select all pods with the app=nginx label
		# and require at least half of the pods selected to be available at any point in time.
		kubectl create pdb my-pdb --selector=app=nginx --min-available=50%`))
)

// NewCmdCreatePodDisruptionBudget is a macro command to create a new pod disruption budget.
func NewCmdCreatePodDisruptionBudget(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "poddisruptionbudget NAME --selector=SELECTOR --min-available=N [--dry-run]",
		Aliases: []string{"pdb"},
		Short:   i18n.T("Create a pod disruption budget with the specified name."),
		Long:    pdbLong,
		Example: pdbExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreatePodDisruptionBudget(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.PodDisruptionBudgetV2GeneratorName)

	cmd.Flags().String("min-available", "", i18n.T("The minimum number or percentage of available pods this budget requires."))
	cmd.Flags().String("max-unavailable", "", i18n.T("The maximum number or percentage of unavailable pods this budget requires."))
	cmd.Flags().String("selector", "", i18n.T("A label selector to use for this budget. Only equality-based selector requirements are supported."))
	return cmd
}

// CreatePodDisruptionBudget implements the behavior to run the create pdb command.
func CreatePodDisruptionBudget(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.PodDisruptionBudgetV1GeneratorName:
		generator = &kubectl.PodDisruptionBudgetV1Generator{
			Name:         name,
			MinAvailable: cmdutil.GetFlagString(cmd, "min-available"),
			Selector:     cmdutil.GetFlagString(cmd, "selector"),
		}
	case cmdutil.PodDisruptionBudgetV2GeneratorName:
		generator = &kubectl.PodDisruptionBudgetV2Generator{
			Name:           name,
			MinAvailable:   cmdutil.GetFlagString(cmd, "min-available"),
			MaxUnavailable: cmdutil.GetFlagString(cmd, "max-unavailable"),
			Selector:       cmdutil.GetFlagString(cmd, "selector"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetFlagBool(cmd, "dry-run"),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
