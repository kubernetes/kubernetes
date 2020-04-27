/*
Copyright 2020 The Kubernetes Authors.

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
	podLong = templates.LongDesc(i18n.T(`
	Create a pod with the specified name.`))

	podExample = templates.Examples(i18n.T(`
	# Create a new pod named my-dep that runs the busybox image.
	kubectl create pod my-dep --image=busybox`))
)

// PodOpts is the options for 'create 'pod' sub command
type PodOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreatePod is a macro command to create a new pod.
// This command is better known to users as `kubectl create pod`.
// Note that this command overlaps significantly with the `kubectl run` command.
func NewCmdCreatePod(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &PodOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use:                   "pod NAME --image=image [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a pod with the specified name."),
		Long:                  podLong,
		Example:               podExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, generateversioned.RunPodV1GeneratorName)
	cmd.Flags().StringSlice("image", []string{}, "Image name to run.")
	cmd.MarkFlagRequired("image")
	return cmd
}

// Complete completes all the required options
func (o *PodOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	imageNames := cmdutil.GetFlagStringSlice(cmd, "image")

	var generator generate.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case generateversioned.RunPodV1GeneratorName:
		generator = &generateversioned.BasicPod{
			Name:    name,
			PodSpec: generateversioned.PodSpec{Images: imageNames},
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// Run calls the CreateSubcommandOptions.Run in NamespaceOpts instance
func (o *PodOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
