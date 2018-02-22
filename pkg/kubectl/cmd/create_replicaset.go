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
	replicaSetLong = templates.LongDesc(i18n.T(`
	Create a ReplicaSet with the specified name.`))

	replicaSetExample = templates.Examples(i18n.T(`
	# Create a new ReplicaSet named my-rs that runs the busybox image.
	kubectl create rs my-rs --image=busybox`))
)

// NewCmdCreateReplicaSet is a command to create a new ReplicaSet.
func NewCmdCreateReplicaSet(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "replicaset NAME --image=image [--dry-run]",
		Aliases: []string{"rs"},
		Short:   i18n.T("Create a ReplicaSet with the specified name."),
		Long:    replicaSetLong,
		Example: replicaSetExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(CreateReplicaSet(f, cmdOut, cmd, args))
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ReplicaSetAppsV1beta2GeneratorName)
	cmdutil.AddImageFlag(cmd)
	cmd.MarkFlagRequired("image")
	cmdutil.AddReplicasFlag(cmd, 1)
	return cmd
}

func CreateReplicaSet(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.ReplicaSetAppsV1beta2GeneratorName:
		generator = &kubectl.ReplicaSetGeneratorAppsV1Beta2{
			BaseReplicaSetGenerator: kubectl.BaseReplicaSetGenerator{
				Replicas: cmdutil.GetFlagInt32(cmd, "replicas"),
				Name:     name,
				Images:   cmdutil.GetFlagStringSlice(cmd, "image"),
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
