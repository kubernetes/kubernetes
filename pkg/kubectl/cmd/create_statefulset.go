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

package cmd

import (
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	statefulSetLong = templates.LongDesc(i18n.T(`
	Create a StatefulSet with the specified name.`))

	statefulSetExample = templates.Examples(i18n.T(`
	# Create a new StatefulSet named my-sts that runs the busybox image.
	kubectl create statefulset my-sts --image=busybox`))
)

// NewCmdCreateStatefulSet is a command to create a new StatefulSet.
func NewCmdCreateStatefulSet(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "statefulset NAME --image=image [--dry-run]",
		Aliases: []string{"sts"},
		Short:   i18n.T("Create a StatefulSet with the specified name."),
		Long:    statefulSetLong,
		Example: statefulSetExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(CreateStatefulSet(f, cmdOut, cmd, args))
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.StatefulSetAppsV1beta2GeneratorName)
	cmd.Flags().StringSlice("image", []string{}, "Image name to run.")
	cmd.MarkFlagRequired("image")
	return cmd
}

func CreateStatefulSet(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.StatefulSetAppsV1beta2GeneratorName:
		generator = &kubectl.StatefulSetGeneratorV1Beta2{
			BaseStatefulSetGenerator: kubectl.BaseStatefulSetGenerator{
				Name:   name,
				Images: cmdutil.GetFlagStringSlice(cmd, "image"),
			},
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
