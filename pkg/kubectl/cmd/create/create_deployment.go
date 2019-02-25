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
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/generate"
	generateversioned "k8s.io/kubernetes/pkg/kubectl/generate/versioned"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
)

var (
	deploymentLong = templates.LongDesc(i18n.T(`
	Create a deployment with the specified name.`))

	deploymentExample = templates.Examples(i18n.T(`
	# Create a new deployment named my-dep that runs the busybox image.
	kubectl create deployment my-dep --image=busybox`))
)

// DeploymentOpts is returned by NewCmdCreateDeployment
type DeploymentOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateDeployment is a macro command to create a new deployment.
// This command is better known to users as `kubectl create deployment`.
// Note that this command overlaps significantly with the `kubectl run` command.
func NewCmdCreateDeployment(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &DeploymentOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use:                   "deployment NAME --image=image [--dry-run]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"deploy"},
		Short:                 i18n.T("Create a deployment with the specified name."),
		Long:                  deploymentLong,
		Example:               deploymentExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, "")
	cmd.Flags().StringSlice("image", []string{}, "Image name to run.")
	cmd.MarkFlagRequired("image")
	return cmd
}

// generatorFromName returns the appropriate StructuredGenerator based on the
// generatorName. If the generatorName is unrecognized, then return (nil,
// false).
func generatorFromName(
	generatorName string,
	imageNames []string,
	deploymentName string,
) (generate.StructuredGenerator, bool) {

	switch generatorName {
	case generateversioned.DeploymentBasicAppsV1GeneratorName:
		generator := &generateversioned.DeploymentBasicAppsGeneratorV1{
			BaseDeploymentGenerator: generateversioned.BaseDeploymentGenerator{
				Name:   deploymentName,
				Images: imageNames,
			},
		}
		return generator, true

	case generateversioned.DeploymentBasicAppsV1Beta1GeneratorName:
		generator := &generateversioned.DeploymentBasicAppsGeneratorV1Beta1{
			BaseDeploymentGenerator: generateversioned.BaseDeploymentGenerator{
				Name:   deploymentName,
				Images: imageNames,
			},
		}
		return generator, true

	case generateversioned.DeploymentBasicV1Beta1GeneratorName:
		generator := &generateversioned.DeploymentBasicGeneratorV1{
			BaseDeploymentGenerator: generateversioned.BaseDeploymentGenerator{
				Name:   deploymentName,
				Images: imageNames,
			},
		}
		return generator, true
	}

	return nil, false
}

// Complete completes all the options
func (o *DeploymentOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}

	generatorName := cmdutil.GetFlagString(cmd, "generator")

	if len(generatorName) == 0 {
		generatorName = generateversioned.DeploymentBasicAppsV1GeneratorName
		generatorNameTemp, err := generateversioned.FallbackGeneratorNameIfNecessary(generatorName, clientset.Discovery(), o.CreateSubcommandOptions.ErrOut)
		if err != nil {
			return err
		}
		if generatorNameTemp != generatorName {
			cmdutil.Warning(o.CreateSubcommandOptions.ErrOut, generatorName, generatorNameTemp)
		} else {
			generatorName = generatorNameTemp
		}
	}

	imageNames := cmdutil.GetFlagStringSlice(cmd, "image")
	generator, ok := generatorFromName(generatorName, imageNames, name)
	if !ok {
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// Run performs the execution of 'create deployment' sub command
func (o *DeploymentOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
