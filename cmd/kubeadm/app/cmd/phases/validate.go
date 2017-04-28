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

package phases

import (
	"os"

	"k8s.io/kubernetes/cmd/kubeadm/app/phases/validate"

	"github.com/spf13/cobra"
)

func NewCmdValidate() *cobra.Command {
	var kubeconfig string

	cmd := &cobra.Command{
		Use:   "validate",
		Short: "Run end to end validation",
		RunE: func(cmd *cobra.Command, args []string) error {

			return validate.Validate(kubeconfig)
		},
	}

	//TODO: what's the convention for defaulting a kubeconfig?
	cmd.Flags().StringVar(&kubeconfig, "kubeconfig", os.Getenv("HOME")+"/.kube/config", "path to kubeconfig")

	return cmd
}
