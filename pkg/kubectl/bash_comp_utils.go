/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// A set of common functions needed by cmd/kubectl and pkg/kubectl packages.

package kubectl

import (
	"k8s.io/kubernetes/pkg/util"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

func AddJsonFilenameFlag(cmd *cobra.Command, value *util.StringList, usage string) {
	annotations := []string{"json", "yaml", "yml"}
	annotation := make(map[string][]string)
	annotation[cobra.BashCompFilenameExt] = annotations

	flag := &pflag.Flag{
		Name:        "filename",
		Shorthand:   "f",
		Usage:       usage,
		Value:       value,
		DefValue:    value.String(),
		Annotations: annotation,
	}
	cmd.Flags().AddFlag(flag)
}

// AddLabelsToColumnsFlag added a user flag to print resource labels into columns. Currently used in kubectl get command
func AddLabelsToColumnsFlag(cmd *cobra.Command, value *util.StringList, usage string) {
	flag := &pflag.Flag{
		Name:      "label-columns",
		Shorthand: "L",
		Usage:     usage,
		Value:     value,
		DefValue:  value.String(),
	}
	cmd.Flags().AddFlag(flag)
}
