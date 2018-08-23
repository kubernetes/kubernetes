/*
Copyright 2015 The Kubernetes Authors.

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
	"strconv"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"

	ct "github.com/daviddengcn/go-colortext"
	"github.com/spf13/cobra"
)

var (
	longDescr = templates.LongDesc(i18n.T(`
  Display addresses of the master and services with label kubernetes.io/cluster-service=true
  To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.`))

	clusterinfoExample = templates.Examples(i18n.T(`
		# Print the address of the master and cluster services
		kubectl cluster-info`))
)

type ClusterInfoOptions struct {
	genericclioptions.IOStreams

	Namespace string

	Builder *resource.Builder
	Client  *restclient.Config
}

func NewCmdClusterInfo(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := &ClusterInfoOptions{
		IOStreams: ioStreams,
	}

	cmd := &cobra.Command{
		Use:     "cluster-info",
		Short:   i18n.T("Display cluster info"),
		Long:    longDescr,
		Example: clusterinfoExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(o.Run())
		},
	}
	cmd.AddCommand(NewCmdClusterInfoDump(f, ioStreams))
	return cmd
}

func (o *ClusterInfoOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	var err error
	o.Client, err = f.ToRESTConfig()
	if err != nil {
		return err
	}

	cmdNamespace := cmdutil.GetFlagString(cmd, "namespace")
	if cmdNamespace == "" {
		cmdNamespace = metav1.NamespaceSystem
	}
	o.Namespace = cmdNamespace

	o.Builder = f.NewBuilder()
	return nil
}

func (o *ClusterInfoOptions) Run() error {
	// TODO use generalized labels once they are implemented (#341)
	b := o.Builder.
		WithScheme(legacyscheme.Scheme).
		NamespaceParam(o.Namespace).DefaultNamespace().
		LabelSelectorParam("kubernetes.io/cluster-service=true").
		ResourceTypeOrNameArgs(false, []string{"services"}...).
		Latest()
	err := b.Do().Visit(func(r *resource.Info, err error) error {
		if err != nil {
			return err
		}
		printService(o.Out, "Kubernetes master", o.Client.Host)

		services := r.Object.(*api.ServiceList).Items
		for _, service := range services {
			var link string
			if len(service.Status.LoadBalancer.Ingress) > 0 {
				ingress := service.Status.LoadBalancer.Ingress[0]
				ip := ingress.IP
				if ip == "" {
					ip = ingress.Hostname
				}
				for _, port := range service.Spec.Ports {
					link += "http://" + ip + ":" + strconv.Itoa(int(port.Port)) + " "
				}
			} else {
				name := service.ObjectMeta.Name

				if len(service.Spec.Ports) > 0 {
					port := service.Spec.Ports[0]

					// guess if the scheme is https
					scheme := ""
					if port.Name == "https" || port.Port == 443 {
						scheme = "https"
					}

					// format is <scheme>:<service-name>:<service-port-name>
					name = utilnet.JoinSchemeNamePort(scheme, service.ObjectMeta.Name, port.Name)
				}

				if len(o.Client.GroupVersion.Group) == 0 {
					link = o.Client.Host + "/api/" + o.Client.GroupVersion.Version + "/namespaces/" + service.ObjectMeta.Namespace + "/services/" + name + "/proxy"
				} else {
					link = o.Client.Host + "/api/" + o.Client.GroupVersion.Group + "/" + o.Client.GroupVersion.Version + "/namespaces/" + service.ObjectMeta.Namespace + "/services/" + name + "/proxy"

				}
			}
			name := service.ObjectMeta.Labels["kubernetes.io/name"]
			if len(name) == 0 {
				name = service.ObjectMeta.Name
			}
			printService(o.Out, name, link)
		}
		return nil
	})
	o.Out.Write([]byte("\nTo further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.\n"))
	return err

	// TODO consider printing more information about cluster
}

func printService(out io.Writer, name, link string) {
	ct.ChangeColor(ct.Green, false, ct.None, false)
	fmt.Fprint(out, name)
	ct.ResetColor()
	fmt.Fprint(out, " is running at ")
	ct.ChangeColor(ct.Yellow, false, ct.None, false)
	fmt.Fprint(out, link)
	ct.ResetColor()
	fmt.Fprintln(out, "")
}
