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
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	deploymentLong = templates.LongDesc(i18n.T(`
	Create a deployment with the specified name.`))

	deploymentExample = templates.Examples(i18n.T(`
	# Create a new deployment named my-dep that runs the busybox image.
	kubectl create deployment my-dep --image=busybox`))
)

// NewCmdCreateDeployment is a macro command to create a new deployment.
// This command is better known to users as `kubectl create deployment`.
// Note that this command overlaps significantly with the `kubectl run` command.
func NewCmdCreateDeployment(f cmdutil.Factory, cmdOut, cmdErr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "deployment NAME --image=image [--dry-run]",
		Aliases: []string{"deploy"},
		Short:   i18n.T("Create a deployment with the specified name."),
		Long:    deploymentLong,
		Example: deploymentExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := createDeployment(f, cmdOut, cmdErr, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.DeploymentBasicV1Beta1GeneratorName)
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
) (kubectl.StructuredGenerator, bool) {

	switch generatorName {
	case cmdutil.DeploymentBasicAppsV1Beta1GeneratorName:
		generator := &kubectl.DeploymentBasicAppsGeneratorV1{
			BaseDeploymentGenerator: kubectl.BaseDeploymentGenerator{
				Name:   deploymentName,
				Images: imageNames,
			},
		}
		return generator, true

	case cmdutil.DeploymentBasicV1Beta1GeneratorName:
		generator := &kubectl.DeploymentBasicGeneratorV1{
			BaseDeploymentGenerator: kubectl.BaseDeploymentGenerator{
				Name:   deploymentName,
				Images: imageNames,
			},
		}
		return generator, true
	}

	return nil, false
}

// createDeployment
// 1. Reads user config values from Cobra.
// 2. Sets up the correct Generator object.
// 3. Calls RunCreateSubcommand.
func createDeployment(f cmdutil.Factory, cmdOut, cmdErr io.Writer,
	cmd *cobra.Command, args []string) error {

	deploymentName, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}
	resourcesList, err := clientset.Discovery().ServerResources()
	// ServerResources ignores errors for old servers do not expose discovery
	if err != nil {
		return fmt.Errorf("failed to discover supported resources: %v", err)
	}

	generatorName := cmdutil.GetFlagString(cmd, "generator")

	// It is possible we have to modify the user-provided generator name if
	// the server does not have support for the requested generator.
	generatorName = cmdutil.FallbackGeneratorNameIfNecessary(generatorName, resourcesList, cmdErr)

	imageNames := cmdutil.GetFlagStringSlice(cmd, "image")
	generator, ok := generatorFromName(generatorName, imageNames, deploymentName)
	if !ok {
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                deploymentName,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
