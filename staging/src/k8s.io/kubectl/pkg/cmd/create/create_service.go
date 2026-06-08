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
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	utilsnet "k8s.io/utils/net"
)

// NewCmdCreateService is a macro command to create a new service
func NewCmdCreateService(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "service",
		Aliases: []string{"svc"},
		Short:   i18n.T("Create a service using a specified subcommand"),
		Long:    i18n.T("Create a service using a specified subcommand."),
		Run:     cmdutil.DefaultSubCommandRun(ioStreams.ErrOut),
	}
	cmd.AddCommand(NewCmdCreateServiceClusterIP(f, ioStreams))
	cmd.AddCommand(NewCmdCreateServiceNodePort(f, ioStreams))
	cmd.AddCommand(NewCmdCreateServiceLoadBalancer(f, ioStreams))
	cmd.AddCommand(NewCmdCreateServiceExternalName(f, ioStreams))

	return cmd
}

// ServiceOptions holds the options for 'create service' sub command
type ServiceOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   func(obj runtime.Object) error

	Name         string
	TCP          []string
	Type         corev1.ServiceType
	ClusterIP    string
	NodePort     int
	ExternalName string

	FieldManager     string
	CreateAnnotation bool
	Namespace        string
	EnforceNamespace bool

	Client              corev1client.CoreV1Interface
	DryRunStrategy      cmdutil.DryRunStrategy
	ValidationDirective string
	genericiooptions.IOStreams
}

// NewServiceOptions creates a ServiceOptions struct
func NewServiceOptions(ioStreams genericiooptions.IOStreams, serviceType corev1.ServiceType) *ServiceOptions {
	return &ServiceOptions{
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
		Type:       serviceType,
	}
}

// Complete completes all the required options
func (o *ServiceOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	o.Name = name

	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.Client, err = corev1client.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	o.ValidationDirective, err = cmdutil.GetValidationDirective(cmd)
	if err != nil {
		return err
	}

	return nil
}

// Validate if the options are valid
func (o *ServiceOptions) Validate() error {
	if o.ClusterIP == corev1.ClusterIPNone && o.Type != corev1.ServiceTypeClusterIP {
		return fmt.Errorf("ClusterIP=None can only be used with ClusterIP service type")
	}
	if o.ClusterIP != corev1.ClusterIPNone && len(o.TCP) == 0 && o.Type != corev1.ServiceTypeExternalName {
		return fmt.Errorf("at least one tcp port specifier must be provided")
	}
	if o.Type == corev1.ServiceTypeExternalName {
		if errs := validation.IsDNS1123Subdomain(o.ExternalName); len(errs) != 0 {
			return fmt.Errorf("invalid service external name %s", o.ExternalName)
		}
	}
	return nil
}

func (o *ServiceOptions) createService() (*corev1.Service, error) {
	ports := []corev1.ServicePort{}
	for _, tcpString := range o.TCP {
		port, targetPort, err := parsePorts(tcpString)
		if err != nil {
			return nil, err
		}

		portName := strings.Replace(tcpString, ":", "-", -1)
		ports = append(ports, corev1.ServicePort{
			Name:       portName,
			Port:       port,
			TargetPort: targetPort,
			Protocol:   corev1.Protocol("TCP"),
			NodePort:   int32(o.NodePort),
		})
	}

	// setup default label and selector
	labels := map[string]string{}
	labels["app"] = o.Name
	selector := map[string]string{}
	selector["app"] = o.Name

	namespace := ""
	if o.EnforceNamespace {
		namespace = o.Namespace
	}

	service := corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      o.Name,
			Labels:    labels,
			Namespace: namespace,
		},
		Spec: corev1.ServiceSpec{
			Type:         o.Type,
			Selector:     selector,
			Ports:        ports,
			ExternalName: o.ExternalName,
		},
	}
	if len(o.ClusterIP) > 0 {
		service.Spec.ClusterIP = o.ClusterIP
	}
	return &service, nil
}

// Run the service command
func (o *ServiceOptions) Run() error {
	service, err := o.createService()
	if err != nil {
		return err
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, service, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		createOptions.FieldValidation = o.ValidationDirective
		if o.DryRunStrategy == cmdutil.DryRunServer {
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		var err error
		service, err = o.Client.Services(o.Namespace).Create(context.TODO(), service, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create %s service: %v", o.Type, err)
		}
	}
	return o.PrintObj(service)
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

// NewCmdCreateServiceClusterIP is a command to create a ClusterIP service
func NewCmdCreateServiceClusterIP(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewServiceOptions(ioStreams, corev1.ServiceTypeClusterIP)

	cmd := &cobra.Command{
		Use:                   "clusterip NAME [--tcp=<port>:<targetPort>] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a ClusterIP service"),
		Long:                  serviceClusterIPLong,
		Example:               serviceClusterIPExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().StringSliceVar(&o.TCP, "tcp", o.TCP, "Port pairs can be specified as '<port>:<targetPort>'.")
	cmd.Flags().StringVar(&o.ClusterIP, "clusterip", o.ClusterIP, i18n.T("Assign your own ClusterIP or set to 'None' for a 'headless' service (no loadbalancing)."))
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	cmdutil.AddDryRunFlag(cmd)

	return cmd
}

var (
	serviceNodePortLong = templates.LongDesc(i18n.T(`
    Create a NodePort service with the specified name.`))

	serviceNodePortExample = templates.Examples(i18n.T(`
    # Create a new NodePort service named my-ns
    kubectl create service nodeport my-ns --tcp=5678:8080`))
)

// NewCmdCreateServiceNodePort is a macro command for creating a NodePort service
func NewCmdCreateServiceNodePort(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewServiceOptions(ioStreams, corev1.ServiceTypeNodePort)

	cmd := &cobra.Command{
		Use:                   "nodeport NAME [--tcp=port:targetPort] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a NodePort service"),
		Long:                  serviceNodePortLong,
		Example:               serviceNodePortExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().IntVar(&o.NodePort, "node-port", o.NodePort, "Port used to expose the service on each node in a cluster.")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	cmd.Flags().StringSliceVar(&o.TCP, "tcp", o.TCP, "Port pairs can be specified as '<port>:<targetPort>'.")
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

var (
	serviceLoadBalancerLong = templates.LongDesc(i18n.T(`
    Create a LoadBalancer service with the specified name.`))

	serviceLoadBalancerExample = templates.Examples(i18n.T(`
    # Create a new LoadBalancer service named my-lbs
    kubectl create service loadbalancer my-lbs --tcp=5678:8080`))
)

// NewCmdCreateServiceLoadBalancer is a macro command for creating a LoadBalancer service
func NewCmdCreateServiceLoadBalancer(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewServiceOptions(ioStreams, corev1.ServiceTypeLoadBalancer)

	cmd := &cobra.Command{
		Use:                   "loadbalancer NAME [--tcp=port:targetPort] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a LoadBalancer service"),
		Long:                  serviceLoadBalancerLong,
		Example:               serviceLoadBalancerExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().StringSliceVar(&o.TCP, "tcp", o.TCP, "Port pairs can be specified as '<port>:<targetPort>'.")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	cmdutil.AddDryRunFlag(cmd)
	return cmd
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

// NewCmdCreateServiceExternalName is a macro command for creating an ExternalName service
func NewCmdCreateServiceExternalName(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewServiceOptions(ioStreams, corev1.ServiceTypeExternalName)

	cmd := &cobra.Command{
		Use:                   "externalname NAME --external-name external.name [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create an ExternalName service"),
		Long:                  serviceExternalNameLong,
		Example:               serviceExternalNameExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmd.Flags().StringSliceVar(&o.TCP, "tcp", o.TCP, "Port pairs can be specified as '<port>:<targetPort>'.")
	cmd.Flags().StringVar(&o.ExternalName, "external-name", o.ExternalName, i18n.T("External name of service"))
	cmd.MarkFlagRequired("external-name")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

func parsePorts(portString string) (int32, intstr.IntOrString, error) {
	portStringSlice := strings.Split(portString, ":")

	port, err := utilsnet.ParsePort(portStringSlice[0], true)
	if err != nil {
		return 0, intstr.FromInt32(0), err
	}

	if len(portStringSlice) == 1 {
		port32 := int32(port)
		return port32, intstr.FromInt32(port32), nil
	}

	var targetPort intstr.IntOrString
	if portNum, err := strconv.Atoi(portStringSlice[1]); err != nil {
		if errs := validation.IsValidPortName(portStringSlice[1]); len(errs) != 0 {
			return 0, intstr.FromInt32(0), errors.New(strings.Join(errs, ","))
		}
		targetPort = intstr.FromString(portStringSlice[1])
	} else {
		if errs := validation.IsValidPortNum(portNum); len(errs) != 0 {
			return 0, intstr.FromInt32(0), errors.New(strings.Join(errs, ","))
		}
		targetPort = intstr.FromInt32(int32(portNum))
	}
	return int32(port), targetPort, nil
}
