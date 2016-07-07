/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/json"

	//"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	//"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	//"k8s.io/kubernetes/pkg/kubectl/resource"
	//client "k8s.io/kubernetes/pkg/client/unversioned"
	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	//"k8s.io/kubernetes/pkg/runtime"
	//utilerrors "k8s.io/kubernetes/pkg/util/errors"
	//"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/api/v1"
	//"github.com/vmware/govmomi/object"
	//"k8s.io/kubernetes/pkg/api"
	//"k8s.io/kubernetes/third_party/golang/go/doc/testdata"
	"k8s.io/kubernetes/pkg/api/resource"
)

var topLongDescr = `Display CPU and Memory usage of one or many resources.`

func NewCmdTop(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &GetOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, false, false, false, false, false, false, []string{})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "top (TYPE [NAME]) [flags]",
		Short:   "Display CPU and Memory usage of one or many resources",
		Long:    topLongDescr,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunTop(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().Bool("all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

var ValidateAvailableResourceName = func(resourceName string) (string, error) {
	// TODO: use existing functions
	if resourceName == "pod" || resourceName == "pods" {
		return "pods", nil
	} else if resourceName == "node" || resourceName == "nodes" {
		return "nodes", nil
	}
	return "", nil
}

func RunTop(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *GetOptions) error {

	selector := cmdutil.GetFlagString(cmd, "selector")
	//allNamespaces := cmdutil.GetFlagBool(cmd, "all-namespaces")

	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if len(args) == 0 {
		fmt.Fprint(out, "You must specify the type of resource to get.")
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}

	resName, err := ValidateAvailableResourceName(args[0])
	if err != nil {
		return err
	}

	client, err := f.Client()
	if err != nil {
		return err
	}

	if (resName == "nodes") {
		metricPath := fmt.Sprintf("/apis/metrics/v1alpha1/nodes/")
		params := map[string]string{"labelSelector": selector};
		resultRaw, err := client.Services(DefaultHeapsterNamespace).
		ProxyGet(DefaultHeapsterScheme, DefaultHeapsterService, DefaultHeapsterPort, metricPath, params).
		DoRaw()

		metrics := make([]metrics_api.NodeMetrics, 0)
		err = json.Unmarshal(resultRaw, &metrics)
		if err != nil {
			fmt.Errorf("failed to unmarshall heapster response: %v", err)
			return err
		}

		for _, m := range metrics {
			cpu, found := m.Usage[v1.ResourceCPU]
			if !found {
				fmt.Errorf("no cpu for node %v/%v", m.Namespace, m.Name)
			}
			mem, found := m.Usage[v1.ResourceMemory]
			if !found {
				fmt.Errorf("no memory for node %v/%v", m.Namespace, m.Name)
			}
			fmt.Fprintf(out, "%s\t%s\t%vm\t%vMi\n", m.Namespace, m.Name, cpu.MilliValue(), mem.MilliValue() / (1024*1024))
		}
	} else if (resName == "pods") {
		metricPath := fmt.Sprintf("/apis/metrics/v1alpha1/namespaces/%s/pods/", cmdNamespace)
		params := map[string]string{"labelSelector": selector};
		resultRaw, err := client.Services(DefaultHeapsterNamespace).
		ProxyGet(DefaultHeapsterScheme, DefaultHeapsterService, DefaultHeapsterPort, metricPath, params).
		DoRaw()

		metrics := make([]metrics_api.PodMetrics, 0)
		err = json.Unmarshal(resultRaw, &metrics)
		if err != nil {
			fmt.Errorf("failed to unmarshall heapster response: %v", err)
			return err
		}

		for _, m := range metrics {
			sumCpu, _ := resource.ParseQuantity("0")
			sumMem, _ := resource.ParseQuantity("0")
			for _, c := range m.Containers {
				cpu, found := c.Usage[v1.ResourceCPU]
				if !found {
					fmt.Errorf("no cpu for pod %v/%v", m.Namespace, m.Name)
				}
				sumCpu.Add(cpu)
				mem, found := c.Usage[v1.ResourceMemory]
				if !found {
					fmt.Errorf("no memory for pod %v/%v", m.Namespace, m.Name)
				}
				sumMem.Add(mem)
			}
			fmt.Fprintf(out, "%s\t%s\t%vm\t%vMi\n", m.Namespace, m.Name, sumCpu.MilliValue(), sumMem.MilliValue() / (1024*1024))
		}
	}

	return nil
}

const (
	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme = "http"
	DefaultHeapsterService = "heapster"
	DefaultHeapsterPort = "" // use the first exposed port on the service
)