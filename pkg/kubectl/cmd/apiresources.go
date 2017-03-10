/*
Copyright 2017 The Kubernetes Authors.

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
	"sort"
	"text/tabwriter"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	apiresources_example = templates.Examples(`
		# Print the supported API Resources
		kubectl api-resources`)
)

func NewCmdApiResources(f cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "api-resources",
		Short:   "Print the supported API resources on the server",
		Long:    "Print the supported API resources on the server",
		Example: apiresources_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunApiResources(f, out)
			cmdutil.CheckErr(err)
		},
	}
	return cmd
}

func RunApiResources(f cmdutil.Factory, output io.Writer) error {
	w := tabwriter.NewWriter(output, 10, 4, 3, ' ', 0)
	fmt.Fprintln(w, "NAME\tNAMESPACED\tKIND\tVERBS")

	discoveryclient, err := f.DiscoveryClient()
	if err != nil {
		return err
	}

	groupList, err := discoveryclient.ServerPreferredResources()
	if err != nil {
		return fmt.Errorf("Couldn't get available api resources from server: %v\n", err)
	}
	resources := []string{}
	m := map[string]metav1.APIResource{}
	for _, group := range groupList {
		for _, resource := range group.APIResources {
			if contain(resources, resource.Name) {
				continue
			}
			resources = append(resources, resource.Name)
			m[resource.Name] = resource
		}
	}
	sort.Strings(resources)
	for _, r := range resources {
		if _, err := fmt.Fprintf(w, "%s\t%v\t%s\t%v\n", m[r].Name, m[r].Namespaced, m[r].Kind, m[r].Verbs); err != nil {
			return err
		}
	}
	w.Flush()
	return nil
}

func contain(slice []string, item string) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}
