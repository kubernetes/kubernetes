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
	"strings"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/printers"
)

var (
	apiresources_example = templates.Examples(`
		# Print the supported API Resources
		kubectl api-resources`)
)

type APIResourceOptions struct {
	APIGroup    string
	APIResource metav1.APIResource
}

func NewCmdApiResources(f cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "api-resources",
		Short:   "Print the supported API resources on the server",
		Long:    "Print the supported API resources on the server",
		Example: apiresources_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(RunApiResources(f, out))
		},
	}
	return cmd
}

func RunApiResources(f cmdutil.Factory, output io.Writer) error {
	w := printers.GetNewTabWriter(output)
	defer w.Flush()

	discoveryclient, err := f.DiscoveryClient()
	if err != nil {
		return err
	}

	lists, err := discoveryclient.ServerPreferredResources()
	if err != nil {
		return fmt.Errorf("Couldn't get available api resources from server: %v", err)
	}

	resources := []string{}
	m := map[string]APIResourceOptions{}
	for _, list := range lists {
		if len(list.APIResources) == 0 {
			continue
		}
		parts := strings.SplitN(list.GroupVersion, "/", 2)
		for _, resource := range list.APIResources {
			if len(resource.Verbs) == 0 || contain(resources, resource.Name) {
				continue
			}
			resources = append(resources, resource.Name)
			m[resource.Name] = APIResourceOptions{
				APIGroup:    parts[0],
				APIResource: resource,
			}
		}
	}
	fmt.Fprintln(w, "NAME\tNAMESPACED\tAPIGROUP\tKIND\tVERBS")
	sort.Strings(resources)
	for _, r := range resources {
		if _, err := fmt.Fprintf(w, "%s\t%v\t%s\t%s\t%v\n",
			m[r].APIResource.Name,
			m[r].APIResource.Namespaced,
			m[r].APIGroup,
			m[r].APIResource.Kind,
			m[r].APIResource.Verbs); err != nil {
			return err
		}
	}
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
