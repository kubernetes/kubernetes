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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// NewCmdCreateService is a macro command to create a new namespace
func NewCmdCreateService(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "service",
		Short: "Create a service using specified subcommand.",
		Long:  "Create a service using specified subcommand.",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}
	cmd.AddCommand(NewCmdCreateServiceClusterIP(f, cmdOut))
	cmd.AddCommand(NewCmdCreateServiceNodePort(f, cmdOut))
	cmd.AddCommand(NewCmdCreateServiceLoadBalancer(f, cmdOut))

	return cmd
}

var (
	serviceClusterIPLong = dedent.Dedent(`
                Create a clusterIP service with the specified name.`)

	serviceClusterIPExample = dedent.Dedent(`
                # Create a new clusterIP service named my-cs
                kubectl create service clusterip my-cs --tcp=5678:8080

                # Create a new clusterIP service named my-cs (in headless mode)
                kubectl create service clusterip my-cs --clusterip="None"`)
)

func addPortFlags(cmd *cobra.Command) {
	cmd.Flags().StringSlice("tcp", []string{}, "Port pairs can be specified as '<port>:<targetPort>'.")
}

// NewCmdCreateServiceClusterIP is a command to create generic secrets from files, directories, or literal values
func NewCmdCreateServiceClusterIP(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "clusterip NAME [--tcp=<port>:<targetPort>] [--dry-run]",
		Short:   "Create a clusterIP service.",
		Long:    serviceClusterIPLong,
		Example: serviceClusterIPExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateServiceClusterIP(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ServiceClusterIPGeneratorV1Name)
	addPortFlags(cmd)
	cmd.Flags().String("clusterip", "", "Assign your own ClusterIP or set to 'None' for a 'headless' service (no loadbalancing).")
	return cmd
}

// CreateServiceClusterIP implements the behavior to run the create namespace command
func CreateServiceClusterIP(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.ServiceClusterIPGeneratorV1Name:
		generator = &kubectl.ServiceCommonGeneratorV1{
			Name:      name,
			TCP:       cmdutil.GetFlagStringSlice(cmd, "tcp"),
			Type:      api.ServiceTypeClusterIP,
			ClusterIP: cmdutil.GetFlagString(cmd, "clusterip"),
		}
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

var (
	serviceNodePortLong = dedent.Dedent(`
                Create a nodeport service with the specified name.`)

	serviceNodePortExample = dedent.Dedent(`
                # Create a new nodeport service named my-ns
                kubectl create service nodeport my-ns --tcp=5678:8080`)
)

// NewCmdCreateServiceNodePort is a macro command for creating secrets to work with Docker registries
func NewCmdCreateServiceNodePort(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "nodeport NAME [--tcp=port:targetPort] [--dry-run]",
		Short:   "Create a NodePort service.",
		Long:    serviceNodePortLong,
		Example: serviceNodePortExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateServiceNodePort(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ServiceNodePortGeneratorV1Name)
	addPortFlags(cmd)
	return cmd
}

// CreateServiceNodePort is the implementation of the create secret docker-registry command
func CreateServiceNodePort(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.ServiceNodePortGeneratorV1Name:
		generator = &kubectl.ServiceCommonGeneratorV1{
			Name:      name,
			TCP:       cmdutil.GetFlagStringSlice(cmd, "tcp"),
			Type:      api.ServiceTypeNodePort,
			ClusterIP: "",
		}
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

var (
	serviceLoadBalancerLong = dedent.Dedent(`
                Create a LoadBalancer service with the specified name.`)

	serviceLoadBalancerExample = dedent.Dedent(`
                # Create a new nodeport service named my-lbs
                kubectl create service loadbalancer my-lbs --tcp=5678:8080`)
)

// NewCmdCreateServiceLoadBalancer is a macro command for creating secrets to work with Docker registries
func NewCmdCreateServiceLoadBalancer(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "loadbalancer NAME [--tcp=port:targetPort] [--dry-run]",
		Short:   "Create a LoadBalancer service.",
		Long:    serviceLoadBalancerLong,
		Example: serviceLoadBalancerExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateServiceLoadBalancer(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ServiceLoadBalancerGeneratorV1Name)
	addPortFlags(cmd)
	return cmd
}

// CreateServiceLoadBalancer is the implementation of the create secret tls command
func CreateServiceLoadBalancer(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.ServiceLoadBalancerGeneratorV1Name:
		generator = &kubectl.ServiceCommonGeneratorV1{
			Name:      name,
			TCP:       cmdutil.GetFlagStringSlice(cmd, "tcp"),
			Type:      api.ServiceTypeLoadBalancer,
			ClusterIP: "",
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
