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

	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	deploymentLong = templates.LongDesc(`
    Create a deployment with the specified name.`)

	deploymentExample = templates.Examples(`
    # Create a new deployment named my-dep that runs the busybox image.
    kubectl create deployment my-dep --image=busybox`)
)

// NewCmdCreateDeployment is a macro command to create a new deployment
func NewCmdCreateDeployment(f cmdutil.Factory, cmdOut, cmdErr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "deployment NAME --image=image [--dry-run]",
		Aliases: []string{"deploy"},
		Short:   i18n.T("Create a deployment with the specified name."),
		Long:    deploymentLong,
		Example: deploymentExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateDeployment(f, cmdOut, cmdErr, cmd, args)
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

// CreateDeployment implements the behavior to run the create deployment command
func CreateDeployment(f cmdutil.Factory, cmdOut, cmdErr io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
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
	// fallback to the old generator if server does not support apps/v1beta1 deployments
	if generatorName == cmdutil.DeploymentBasicAppsV1Beta1GeneratorName &&
		!contains(resourcesList, appsv1beta1.SchemeGroupVersion.WithResource("deployments")) {
		fmt.Fprintf(cmdErr, "WARNING: New deployments generator specified (%s), but apps/v1beta1.Deployments are not available, falling back to the old one (%s).\n",
			cmdutil.DeploymentBasicAppsV1Beta1GeneratorName, cmdutil.DeploymentBasicV1Beta1GeneratorName)
		generatorName = cmdutil.DeploymentBasicV1Beta1GeneratorName
	}
	var generator kubectl.StructuredGenerator
	switch generatorName {
	case cmdutil.DeploymentBasicAppsV1Beta1GeneratorName:
		generator = &kubectl.DeploymentBasicAppsGeneratorV1{Name: name, Images: cmdutil.GetFlagStringSlice(cmd, "image")}
	case cmdutil.DeploymentBasicV1Beta1GeneratorName:
		generator = &kubectl.DeploymentBasicGeneratorV1{Name: name, Images: cmdutil.GetFlagStringSlice(cmd, "image")}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
