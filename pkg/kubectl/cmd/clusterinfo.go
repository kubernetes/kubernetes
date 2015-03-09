/*
Copyright 2015 Google Inc. All rights reserved.

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
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"

	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdClusterInfo(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "clusterinfo",
		Short: "Display cluster info",
		Long:  "Display addresses of the master and services with label kubernetes.io/cluster-service=true",
		Run: func(cmd *cobra.Command, args []string) {
			RunClusterInfo(f, out, cmd)
		},
	}
	return cmd
}

func RunClusterInfo(factory *Factory, out io.Writer, cmd *cobra.Command) {
	client, err := factory.ClientConfig(cmd)
	util.CheckErr(err)
	fmt.Fprintf(out, "Kubernetes master is running at %v\n", client.Host)

	mapper, typer := factory.Object(cmd)
	cmdNamespace, err := factory.DefaultNamespace(cmd)
	util.CheckErr(err)

	// TODO: use generalized labels once they are implemented (#341)
	b := resource.NewBuilder(mapper, typer, factory.ClientMapperForCommand(cmd)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		SelectorParam("kubernetes.io/cluster-service=true").
		ResourceTypeOrNameArgs(false, []string{"services"}...).
		Latest()
	b.Do().Visit(func(r *resource.Info) error {
		services := r.Object.(*api.ServiceList).Items
		for _, service := range services {
			splittedLink := strings.Split(strings.Split(service.ObjectMeta.SelfLink, "?")[0], "/")
			// insert "proxy" into the link
			splittedLink = append(splittedLink, "")
			copy(splittedLink[4:], splittedLink[3:])
			splittedLink[3] = "proxy"
			link := strings.Join(splittedLink, "/")
			fmt.Fprintf(out, "%v is running at %v%v/\n", service.ObjectMeta.Labels["name"], client.Host, link)
		}
		return nil
	})

	// TODO: consider printing more information about cluster
}
