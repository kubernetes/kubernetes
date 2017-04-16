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
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	ingressLong = templates.LongDesc(i18n.T(`
    Create an ingress with the specified name.`))

	ingressExample = templates.Examples(i18n.T(`
    # Create a new ingress named my-ingress supporting TLS with Let's Encrypt
    kubectl create ingress my-ingress --host=host.example.com --tls-acme

    # Create a new ingress named my-ingress exposing the service my-service
    kubectl create ingress my-ingress --host=host.example.com --service-name=my-service`))
)

// NewCmdCreateIngress is a macro command for creating an ingress
func NewCmdCreateIngress(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "ingress NAME --host=host.example.com [--tls-acme] [--dry-run]",
		Short:   i18n.T("Create an Ingress with the specified name."),
		Long:    ingressLong,
		Example: ingressExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateIngress(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.IngressV1Beta1GeneratorName)
	cmd.Flags().StringSlice("host", []string{}, "Host name for the ingress record")
	cmd.MarkFlagRequired("host")
	cmd.Flags().Bool("tls-acme", false, "Enables ACME (Let's Encrypt) support for automatic TLS")
	cmd.Flags().String("service-name", "", "Name of backend service; defaults to same as ingress name.")
	cmd.Flags().String("service-port", "80", "Port of backend service; defaults to port 80.")
	return cmd
}

// CreateIngress is the implementation of the create ingress command
func CreateIngress(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.IngressV1Beta1GeneratorName:
		generator = &kubectl.IngressV1Beta1{
			Name:        name,
			Host:        cmdutil.GetFlagStringSlice(cmd, "host"),
			ServiceName: cmdutil.GetFlagString(cmd, "service-name"),
			ServicePort: intstr.Parse(cmdutil.GetFlagString(cmd, "service-port")),
			TLSAcme:     cmdutil.GetFlagBool(cmd, "tls-acme"),
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
