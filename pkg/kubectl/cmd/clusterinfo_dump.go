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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/printers"
)

// NewCmdCreateSecret groups subcommands to create various types of secrets
func NewCmdClusterInfoDump(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "dump",
		Short:   i18n.T("Dump lots of relevant info for debugging and diagnosis"),
		Long:    dumpLong,
		Example: dumpExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(dumpClusterInfo(f, cmd, cmdOut))
		},
	}
	cmd.Flags().String("output-directory", "", i18n.T("Where to output the files.  If empty or '-' uses stdout, otherwise creates a directory hierarchy in that directory"))
	cmd.Flags().StringSlice("namespaces", []string{}, "A comma separated list of namespaces to dump.")
	cmd.Flags().Bool("all-namespaces", false, "If true, dump all namespaces.  If true, --namespaces is ignored.")
	cmdutil.AddPodRunningTimeoutFlag(cmd, defaultPodLogsTimeout)
	return cmd
}

var (
	dumpLong = templates.LongDesc(i18n.T(`
    Dumps cluster info out suitable for debugging and diagnosing cluster problems.  By default, dumps everything to
    stdout. You can optionally specify a directory with --output-directory.  If you specify a directory, kubernetes will
    build a set of files in that directory.  By default only dumps things in the 'kube-system' namespace, but you can
    switch to a different namespace with the --namespaces flag, or specify --all-namespaces to dump all namespaces.

    The command also dumps the logs of all of the pods in the cluster, these logs are dumped into different directories
    based on namespace and pod name.`))

	dumpExample = templates.Examples(i18n.T(`
    # Dump current cluster state to stdout
    kubectl cluster-info dump

    # Dump current cluster state to /path/to/cluster-state
    kubectl cluster-info dump --output-directory=/path/to/cluster-state

    # Dump all namespaces to stdout
    kubectl cluster-info dump --all-namespaces

    # Dump a set of namespaces to /path/to/cluster-state
    kubectl cluster-info dump --namespaces default,kube-system --output-directory=/path/to/cluster-state`))
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

func dumpClusterInfo(f cmdutil.Factory, cmd *cobra.Command, out io.Writer) error {
	timeout, err := cmdutil.GetPodRunningTimeoutFlag(cmd)
	if err != nil {
		return cmdutil.UsageErrorf(cmd, err.Error())
	}

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}

	printer := &printers.JSONPrinter{}

	nodes, err := clientset.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return err
	}

	if err := printer.PrintObj(nodes, setupOutputWriter(cmd, out, "nodes.json")); err != nil {
		return err
	}

	var namespaces []string
	if cmdutil.GetFlagBool(cmd, "all-namespaces") {
		namespaceList, err := clientset.Core().Namespaces().List(metav1.ListOptions{})
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
				metav1.NamespaceSystem,
				cmdNamespace,
			}
		}
	}
	for _, namespace := range namespaces {
		// TODO: this is repetitive in the extreme.  Use reflection or
		// something to make this a for loop.
		events, err := clientset.Core().Events(namespace).List(metav1.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(events, setupOutputWriter(cmd, out, path.Join(namespace, "events.json"))); err != nil {
			return err
		}

		rcs, err := clientset.Core().ReplicationControllers(namespace).List(metav1.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(rcs, setupOutputWriter(cmd, out, path.Join(namespace, "replication-controllers.json"))); err != nil {
			return err
		}

		svcs, err := clientset.Core().Services(namespace).List(metav1.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(svcs, setupOutputWriter(cmd, out, path.Join(namespace, "services.json"))); err != nil {
			return err
		}

		sets, err := clientset.Extensions().DaemonSets(namespace).List(metav1.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(sets, setupOutputWriter(cmd, out, path.Join(namespace, "daemonsets.json"))); err != nil {
			return err
		}

		deps, err := clientset.Extensions().Deployments(namespace).List(metav1.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(deps, setupOutputWriter(cmd, out, path.Join(namespace, "deployments.json"))); err != nil {
			return err
		}

		rps, err := clientset.Extensions().ReplicaSets(namespace).List(metav1.ListOptions{})
		if err != nil {
			return err
		}
		if err := printer.PrintObj(rps, setupOutputWriter(cmd, out, path.Join(namespace, "replicasets.json"))); err != nil {
			return err
		}

		pods, err := clientset.Core().Pods(namespace).List(metav1.ListOptions{})
		if err != nil {
			return err
		}

		if err := printer.PrintObj(pods, setupOutputWriter(cmd, out, path.Join(namespace, "pods.json"))); err != nil {
			return err
		}

		printContainer := func(writer io.Writer, container api.Container, pod *api.Pod) {
			writer.Write([]byte(fmt.Sprintf("==== START logs for container %s of pod %s/%s ====\n", container.Name, pod.Namespace, pod.Name)))
			defer writer.Write([]byte(fmt.Sprintf("==== END logs for container %s of pod %s/%s ====\n", container.Name, pod.Namespace, pod.Name)))

			request, err := f.LogsForObject(pod, &api.PodLogOptions{Container: container.Name}, timeout)
			if err != nil {
				// Print error and return.
				writer.Write([]byte(fmt.Sprintf("Create log request error: %s\n", err.Error())))
				return
			}

			data, err := request.DoRaw()
			if err != nil {
				// Print error and return.
				writer.Write([]byte(fmt.Sprintf("Request log error: %s\n", err.Error())))
				return
			}
			writer.Write(data)
		}

		for ix := range pods.Items {
			pod := &pods.Items[ix]
			containers := pod.Spec.Containers
			writer := setupOutputWriter(cmd, out, path.Join(namespace, pod.Name, "logs.txt"))

			for i := range containers {
				printContainer(writer, containers[i], pod)
			}
		}
	}
	dir := cmdutil.GetFlagString(cmd, "output-directory")
	if len(dir) == 0 {
		dir = "standard output"
	}
	if dir != "-" {
		fmt.Fprintf(out, "Cluster info dumped to %s\n", dir)
	}
	return nil
}
