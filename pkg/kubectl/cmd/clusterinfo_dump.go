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
	"os"
	"path"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// NewCmdCreateSecret groups subcommands to create various types of secrets
func NewCmdClusterInfoDump(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "dump",
		Short:   "Dump lots of relevant info for debugging and diagnosis",
		Long:    dumpLong,
		Example: dumpExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(dumpClusterInfo(f, cmd, args, cmdOut))
		},
	}
	cmd.Flags().String("output-directory", "", "Where to output the files.  If empty or '-' uses stdout, otherwise creates a directory hierarchy in that directory")
	cmd.Flags().StringSlice("namespaces", []string{}, "A comma separated list of namespaces to dump.")
	cmd.Flags().Bool("all-namespaces", false, "If true, dump all namespaces.  If true, --namespaces is ignored.")
	return cmd
}

const (
	dumpLong = `
Dumps cluster info out suitable for debugging and diagnosing cluster problems.  By default, dumps everything to
stdout. You can optionally specify a directory with --output-directory.  If you specify a directory, kubernetes will
build a set of files in that directory.  By default only dumps things in the 'kube-system' namespace, but you can
switch to a different namespace with the --namespaces flag, or specify --all-namespaces to dump all namespaces.

The command also dumps the logs of all of the pods in the cluster, these logs are dumped into different directories
based on namespace and pod name.
`

	dumpExample = `# Dump current cluster state to stdout
kubectl cluster-info dump

# Dump current cluster state to /path/to/cluster-state
kubectl cluster-info dump --output-directory=/path/to/cluster-state

# Dump all namespaces to stdout
kubectl cluster-info dump --all-namespaces

# Dump a set of namespaces to /path/to/cluster-state
kubectl cluster-info dump --namespaces default,kube-system --output-directory=/path/to/cluster-state`
)

func setupOutputWriter(cmd *cobra.Command, defaultWriter io.Writer, filename string) io.Writer {
	dir := cmdutil.GetFlagString(cmd, "output-directory")
	if len(dir) == 0 || dir == "-" {
		return defaultWriter
	}
	fullFile := path.Join(dir, filename)
	parent := path.Dir(fullFile)
	cmdutil.CheckErr(os.MkdirAll(parent, 0755))

	file, err := os.Create(path.Join(dir, filename))
	cmdutil.CheckErr(err)
	return file
}

func dumpClusterInfo(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	var c *unversioned.Client
	var err error
	if c, err = f.Client(); err != nil {
		return err
	}
	printer, _, err := kubectl.GetPrinter("json", "", false)
	if err != nil {
		return err
	}

	nodes, err := c.Nodes().List(api.ListOptions{})
	if err != nil {
		return err
	}

	if err := printer.PrintObj(nodes, setupOutputWriter(cmd, out, "nodes.json")); err != nil {
		return err
	}

	var namespaces []string
	if cmdutil.GetFlagBool(cmd, "all-namespaces") {
		namespaceList, err := c.Namespaces().List(api.ListOptions{})
		if err != nil {
			return err
		}
		for ix := range namespaceList.Items {
			namespaces = append(namespaces, namespaceList.Items[ix].Name)
		}
	} else {
		namespaces = cmdutil.GetFlagStringSlice(cmd, "namespaces")
		if len(namespaces) == 0 {
			cmdNamespace, _, err := f.DefaultNamespace()
			if err != nil {
				return err
			}
			namespaces = []string{
				api.NamespaceSystem,
				cmdNamespace,
			}
		}
	}
	for _, namespace := range namespaces {
		// TODO: this is repetitive in the extreme.  Use reflection or
		// something to make this a for loop.
		events, err := c.Events(namespace).List(api.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(events, setupOutputWriter(cmd, out, path.Join(namespace, "events.json"))); err != nil {
			return err
		}

		rcs, err := c.ReplicationControllers(namespace).List(api.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(rcs, setupOutputWriter(cmd, out, path.Join(namespace, "replication-controllers.json"))); err != nil {
			return err
		}

		svcs, err := c.Services(namespace).List(api.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(svcs, setupOutputWriter(cmd, out, path.Join(namespace, "services.json"))); err != nil {
			return err
		}

		sets, err := c.DaemonSets(namespace).List(api.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(sets, setupOutputWriter(cmd, out, path.Join(namespace, "daemonsets.json"))); err != nil {
			return err
		}

		deps, err := c.Deployments(namespace).List(api.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(deps, setupOutputWriter(cmd, out, path.Join(namespace, "deployments.json"))); err != nil {
			return err
		}

		rps, err := c.ReplicaSets(namespace).List(api.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(rps, setupOutputWriter(cmd, out, path.Join(namespace, "replicasets.json"))); err != nil {
			return err
		}

		pods, err := c.Pods(namespace).List(api.ListOptions{})
		if err != nil {
			return err
		}

		if err := printer.PrintObj(pods, setupOutputWriter(cmd, out, path.Join(namespace, "pods.json"))); err != nil {
			return err
		}

		for ix := range pods.Items {
			pod := &pods.Items[ix]
			writer := setupOutputWriter(cmd, out, path.Join(namespace, pod.Name, "logs.txt"))
			writer.Write([]byte(fmt.Sprintf("==== START logs for %s/%s ====\n", pod.Namespace, pod.Name)))
			request, err := f.LogsForObject(pod, &api.PodLogOptions{})
			if err != nil {
				return err
			}

			data, err := request.DoRaw()
			if err != nil {
				return err
			}
			writer.Write(data)
			writer.Write([]byte(fmt.Sprintf("==== END logs for %s/%s ====\n", pod.Namespace, pod.Name)))
		}
	}
	dir := cmdutil.GetFlagString(cmd, "output-directory")
	if len(dir) == 0 {
		dir = "."
	}
	if dir != "-" {
		fmt.Fprintf(out, "Cluster info dumped to %s", dir)
	}
	return nil
}
