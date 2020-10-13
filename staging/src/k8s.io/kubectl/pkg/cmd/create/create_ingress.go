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
	"context"
	"fmt"
	"strconv"

	"github.com/spf13/cobra"

	"k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	networkingv1client "k8s.io/client-go/kubernetes/typed/networking/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	ingressLong = templates.LongDesc(i18n.T(`
		Create an ingress with the specified name.`))

	ingressExample = templates.Examples(i18n.T(`
		# Create a new ingress named my-app.
		kubectl create ingress my-app --host=foo.bar.com --service-name=my-svc`))
)

// CreateIngressOptions is returned by NewCmdCreateIngress
type CreateIngressOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	PrintObj func(obj runtime.Object) error

	Name        string
	Host        string
	ServiceName string
	ServicePort string
	Path        string

	Namespace      string
	Client         *networkingv1client.NetworkingV1Client
	DryRunStrategy cmdutil.DryRunStrategy
	DryRunVerifier *resource.DryRunVerifier
	Builder        *resource.Builder
	Cmd            *cobra.Command

	genericclioptions.IOStreams
}

// NewCreateCreateIngressOptions creates and returns an instance of CreateIngressOptions
func NewCreateCreateIngressOptions(ioStreams genericclioptions.IOStreams) *CreateIngressOptions {
	return &CreateIngressOptions{
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// NewCmdCreateIngress is a macro command to create a new ingress.
func NewCmdCreateIngress(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCreateCreateIngressOptions(ioStreams)

	cmd := &cobra.Command{
		Use:     "ingress NAME --host=hostname| --service-name=servicename [--service-port=serviceport] [--path=path] [--dry-run]",
		Aliases: []string{"ing"},
		Short:   i18n.T("Create an ingress with the specified name."),
		Long:    ingressLong,
		Example: ingressExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringVar(&o.Host, "host", o.Host, i18n.T("Host name this Ingress should route traffic on"))
	cmd.Flags().StringVar(&o.ServiceName, "service-name", o.ServiceName, i18n.T("Service this Ingress should route traffic to"))
	cmd.Flags().StringVar(&o.ServicePort, "service-port", o.ServicePort, "Port name or number of the Service to route traffic to")
	cmd.Flags().StringVar(&o.Path, "path", o.Path, "Path on which to route traffic to")
	cmd.MarkFlagRequired("host")
	cmd.MarkFlagRequired("service-name")
	return cmd
}

// Complete completes all the options
func (o *CreateIngressOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	o.Name = name

	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.Client, err = networkingv1client.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	o.Builder = f.NewBuilder()
	o.Cmd = cmd

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	discoveryClient, err := f.ToDiscoveryClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewDryRunVerifier(dynamicClient, discoveryClient)
	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	return nil
}

func (o *CreateIngressOptions) Validate() error {
	return nil
}

// Run performs the execution of 'create ingress' sub command
func (o *CreateIngressOptions) Run() error {
	var ingress *v1.Ingress
	ingress = o.createIngress()

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(ingress.GroupVersionKind()); err != nil {
				return err
			}
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		var err error
		ingress, err = o.Client.Ingresses(o.Namespace).Create(context.TODO(), ingress, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create ingress: %v", err)
		}
	}

	return o.PrintObj(ingress)
}

func (o *CreateIngressOptions) createIngress() *v1.Ingress {
	i := &v1.Ingress{
		TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Ingress"},
		ObjectMeta: metav1.ObjectMeta{
			Name: o.Name,
		},
		Spec: v1.IngressSpec{
			Rules: []v1.IngressRule{
				{
					Host: o.Host,
					IngressRuleValue: v1.IngressRuleValue{
						HTTP: &v1.HTTPIngressRuleValue{
							Paths: []v1.HTTPIngressPath{
								{
									Path: o.Path,
									Backend: v1.IngressBackend{
										Service: &v1.IngressServiceBackend{
											Name: o.ServiceName,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	var port v1.ServiceBackendPort
	if n, err := strconv.Atoi(o.ServicePort); err != nil {
		port.Name = o.ServicePort
	} else {
		port.Number = int32(n)
	}

	i.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].Backend.Service.Port = port

	return i
}
