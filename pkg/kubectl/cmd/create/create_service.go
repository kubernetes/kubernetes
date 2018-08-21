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

	"k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// NewCmdCreateService is a macro command to create a new service
func NewCmdCreateService(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "service",
		Aliases: []string{"svc"},
		Short:   i18n.T("Create a service using specified subcommand."),
		Long:    "Create a service using specified subcommand.",
		Run:     cmdutil.DefaultSubCommandRun(ioStreams.ErrOut),
	}
	cmd.AddCommand(NewCmdCreateServiceClusterIP(f, ioStreams))
	cmd.AddCommand(NewCmdCreateServiceNodePort(f, ioStreams))
	cmd.AddCommand(NewCmdCreateServiceLoadBalancer(f, ioStreams))
	cmd.AddCommand(NewCmdCreateServiceExternalName(f, ioStreams))

	return cmd
}

var (
	serviceClusterIPLong = templates.LongDesc(i18n.T(`
    Create a ClusterIP service with the specified name.`))

	serviceClusterIPExample = templates.Examples(i18n.T(`
    # Create a new ClusterIP service named my-cs
    kubectl create service clusterip my-cs --tcp=5678:8080

    # Create a new ClusterIP service named my-cs (in headless mode)
    kubectl create service clusterip my-cs --clusterip="None"`))
)

func addPortFlags(cmd *cobra.Command) {
	cmd.Flags().StringSlice("tcp", []string{}, "Port pairs can be specified as '<port>:<targetPort>'.")
}

type ServiceClusterIPOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateServiceClusterIP is a command to create a ClusterIP service
func NewCmdCreateServiceClusterIP(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &ServiceClusterIPOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use: "clusterip NAME [--tcp=<port>:<targetPort>] [--dry-run]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Create a ClusterIP service."),
		Long:    serviceClusterIPLong,
		Example: serviceClusterIPExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ServiceClusterIPGeneratorV1Name)
	addPortFlags(cmd)
	cmd.Flags().String("clusterip", "", i18n.T("Assign your own ClusterIP or set to 'None' for a 'headless' service (no loadbalancing)."))
	return cmd
}

func errUnsupportedGenerator(cmd *cobra.Command, generatorName string) error {
	return cmdutil.UsageErrorf(cmd, "Generator %s not supported. ", generatorName)
}

func (o *ServiceClusterIPOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
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
			Type:      v1.ServiceTypeClusterIP,
			ClusterIP: cmdutil.GetFlagString(cmd, "clusterip"),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// CreateServiceClusterIP is the implementation of the create service clusterip command
func (o *ServiceClusterIPOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}

var (
	serviceNodePortLong = templates.LongDesc(i18n.T(`
    Create a NodePort service with the specified name.`))

	serviceNodePortExample = templates.Examples(i18n.T(`
    # Create a new NodePort service named my-ns
    kubectl create service nodeport my-ns --tcp=5678:8080`))
)

type ServiceNodePortOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateServiceNodePort is a macro command for creating a NodePort service
func NewCmdCreateServiceNodePort(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &ServiceNodePortOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use: "nodeport NAME [--tcp=port:targetPort] [--dry-run]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Create a NodePort service."),
		Long:    serviceNodePortLong,
		Example: serviceNodePortExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ServiceNodePortGeneratorV1Name)
	cmd.Flags().Int("node-port", 0, "Port used to expose the service on each node in a cluster.")
	addPortFlags(cmd)
	return cmd
}

func (o *ServiceNodePortOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
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
			Type:      v1.ServiceTypeNodePort,
			ClusterIP: "",
			NodePort:  cmdutil.GetFlagInt(cmd, "node-port"),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// CreateServiceNodePort is the implementation of the create service nodeport command
func (o *ServiceNodePortOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}

var (
	serviceLoadBalancerLong = templates.LongDesc(i18n.T(`
    Create a LoadBalancer service with the specified name.`))

	serviceLoadBalancerExample = templates.Examples(i18n.T(`
    # Create a new LoadBalancer service named my-lbs
    kubectl create service loadbalancer my-lbs --tcp=5678:8080`))
)

type ServiceLoadBalancerOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateServiceLoadBalancer is a macro command for creating a LoadBalancer service
func NewCmdCreateServiceLoadBalancer(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &ServiceLoadBalancerOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use: "loadbalancer NAME [--tcp=port:targetPort] [--dry-run]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Create a LoadBalancer service."),
		Long:    serviceLoadBalancerLong,
		Example: serviceLoadBalancerExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ServiceLoadBalancerGeneratorV1Name)
	addPortFlags(cmd)
	return cmd
}

func (o *ServiceLoadBalancerOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
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
			Type:      v1.ServiceTypeLoadBalancer,
			ClusterIP: "",
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// CreateServiceLoadBalancer is the implementation of the create service loadbalancer command
func (o *ServiceLoadBalancerOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}

var (
	serviceExternalNameLong = templates.LongDesc(i18n.T(`
	Create an ExternalName service with the specified name.

	ExternalName service references to an external DNS address instead of
	only pods, which will allow application authors to reference services
	that exist off platform, on other clusters, or locally.`))

	serviceExternalNameExample = templates.Examples(i18n.T(`
	# Create a new ExternalName service named my-ns
	kubectl create service externalname my-ns --external-name bar.com`))
)

type ServiceExternalNameOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateServiceExternalName is a macro command for creating an ExternalName service
func NewCmdCreateServiceExternalName(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := &ServiceExternalNameOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use: "externalname NAME --external-name external.name [--dry-run]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Create an ExternalName service."),
		Long:    serviceExternalNameLong,
		Example: serviceExternalNameExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Run())
		},
	}

	options.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ServiceExternalNameGeneratorV1Name)
	addPortFlags(cmd)
	cmd.Flags().String("external-name", "", i18n.T("External name of service"))
	cmd.MarkFlagRequired("external-name")
	return cmd
}

func (o *ServiceExternalNameOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.ServiceExternalNameGeneratorV1Name:
		generator = &kubectl.ServiceCommonGeneratorV1{
			Name:         name,
			Type:         v1.ServiceTypeExternalName,
			ExternalName: cmdutil.GetFlagString(cmd, "external-name"),
			ClusterIP:    "",
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// CreateExternalNameService is the implementation of the create service externalname command
func (o *ServiceExternalNameOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
