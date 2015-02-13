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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"

	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdClusterInfo(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "clusterinfo",
		Short: "Display cluster info",
		Long: `Display IPs of the master and the following services running in cluster:
	Grafana, Elasticsearch and Kibana`,
		Run: func(cmd *cobra.Command, args []string) {
			RunClusterInfo(f, out, cmd)
		},
	}
	return cmd
}

func RunClusterInfo(f *Factory, out io.Writer, cmd *cobra.Command) {
	client, err := f.ClientConfig(cmd)
	checkErr(err)
	fmt.Fprintf(out, "Kubernetes master is running at %v\n", client.Host)

	printPodInfo(f, out, cmd, "name=influxGrafana", "Grafana dashboard", 0)
	printPodInfo(f, out, cmd, "name=elasticsearch-logging", "Elasticsearch", 9200)
	printPodInfo(f, out, cmd, "name=kibana-logging", "Kibana", 5601)
}

func printPodInfo(f *Factory, out io.Writer, cmd *cobra.Command, selector string, name string, port int) {
	mapper, typer := f.Object(cmd)
	cmdNamespace, err := f.DefaultNamespace(cmd)
	checkErr(err)

	args := []string{"pods"}

	b := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand(cmd)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		SelectorParam(selector).
		ResourceTypeOrNameArgs(args...).
		Latest()

	b.Do().Visit(func(r *resource.Info) error {
		pods := r.Object.(*api.PodList).Items
		if len(pods) == 0 {
			return fmt.Errorf("No pods with %v expected 1", selector)
		}
		if len(pods) > 1 {
			return fmt.Errorf("Too many pods with %v, expected 1 but found %v", selector, len(pods))
		}

		fmt.Fprintf(out, "%v is running at http://%v", name, pods[0].Status.HostIP)
		if port > 0 {
			fmt.Fprintf(out, ":%v", port)
		}
		fmt.Fprint(out, "\n")

		return nil
	})
}
