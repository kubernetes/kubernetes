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
	"io"

	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"

	"github.com/spf13/cobra"
)

var (
	hpaExample = templates.Examples(`
 		# Auto scale a deployment "foo", with the number of pods between 2 and 10, target CPU utilization specified so a default autoscaling policy will be used:
 		kubectl create horizontalpodautoscaler deployment foo --min=2 --max=10
 
 		# Auto scale a replication controller "foo", with the number of pods between 1 and 5, target CPU utilization at 80%:
 		kubectl create horizontalpodautoscaler rc foo --max=5 --cpu-percent=80`)
)

// HorizontalPodAutoscaler is a command to ease creating Horizontal Pod Autoscaler.
func NewCmdCreateHorizontalPodAutoscaler(f cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "horizontalpodautoscaler (TYPE NAME | TYPE/NAME) [--min=MINPODS] --max=MAXPODS [--cpu-percent=CPU] [flags]",
		Aliases: []string{"hpa"},
		Short:   autoscaleShort,
		Long:    autoscaleLong,
		Example: hpaExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunAutoscale(f, out, cmd, args, &resource.FilenameOptions{})
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	applyAutoscaleCommonFlags(cmd)

	return cmd
}
