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
	"os"
	"sort"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	apiversions_example = templates.Examples(`
		# Print the supported API versions
		kubectl api-versions`)
)

func NewCmdApiVersions(f cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use: "api-versions",
		// apiversions is deprecated.
		Aliases: []string{"apiversions"},
		Short:   "Print the supported API versions on the server, in the form of \"group/version\"",
		Long:    "Print the supported API versions on the server, in the form of \"group/version\"",
		Example: apiversions_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunApiVersions(f, out)
			cmdutil.CheckErr(err)
		},
	}
	return cmd
}

func RunApiVersions(f cmdutil.Factory, w io.Writer) error {
	if len(os.Args) > 1 && os.Args[1] == "apiversions" {
		printDeprecationWarning("api-versions", "apiversions")
	}

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}

	groupList, err := clientset.Discovery().ServerGroups()
	if err != nil {
		return fmt.Errorf("Couldn't get available api versions from server: %v\n", err)
	}
	apiVersions := metav1.ExtractGroupVersions(groupList)
	sort.Strings(apiVersions)
	for _, v := range apiVersions {
		fmt.Fprintln(w, v)
	}
	return nil
}
