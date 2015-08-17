/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	libutil "k8s.io/kubernetes/pkg/util"
	"github.com/spf13/cobra"
)

const (
	log_example = `// Returns snapshot of ruby-container logs from pod 123456-7890.
$ kubectl logs 123456-7890 ruby-container

// Returns snapshot of previous terminated ruby-container logs from pod 123456-7890.
$ kubectl logs -p 123456-7890 ruby-container

// Starts streaming of ruby-container logs from pod 123456-7890.
$ kubectl logs -f 123456-7890 ruby-container`
)

func selectContainer(pod *api.Pod, in io.Reader, out io.Writer) string {
	fmt.Fprintf(out, "Please select a container:\n")
	options := libutil.StringSet{}
	for ix := range pod.Spec.Containers {
		fmt.Fprintf(out, "[%d] %s\n", ix+1, pod.Spec.Containers[ix].Name)
		options.Insert(pod.Spec.Containers[ix].Name)
	}
	for {
		var input string
		fmt.Fprintf(out, "> ")
		fmt.Fscanln(in, &input)
		if options.Has(input) {
			return input
		}
		ix, err := strconv.Atoi(input)
		if err == nil && ix > 0 && ix <= len(pod.Spec.Containers) {
			return pod.Spec.Containers[ix-1].Name
		}
		fmt.Fprintf(out, "Invalid input: %s", input)
	}
}

type logParams struct {
	containerName string
}

// NewCmdLog creates a new pod log command
func NewCmdLog(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	params := &logParams{}
	cmd := &cobra.Command{
		Use:     "logs [-f] [-p] POD [-c CONTAINER]",
		Short:   "Print the logs for a container in a pod.",
		Long:    "Print the logs for a container in a pod. If the pod has only one container, the container name is optional.",
		Example: log_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunLog(f, out, cmd, args, params)
			cmdutil.CheckErr(err)
		},
		Aliases: []string{"log"},
	}
	cmd.Flags().BoolP("follow", "f", false, "Specify if the logs should be streamed.")
	cmd.Flags().Bool("interactive", true, "If true, prompt the user for input when required. Default true.")
	cmd.Flags().BoolP("previous", "p", false, "If true, print the logs for the previous instance of the container in a pod if it exists.")
	cmd.Flags().StringVarP(&params.containerName, "container", "c", "", "Container name")
	return cmd
}

// RunLog retrieves a pod log
func RunLog(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, p *logParams) error {
	if len(os.Args) > 1 && os.Args[1] == "log" {
		printDeprecationWarning("logs", "log")
	}

	if len(args) == 0 {
		return cmdutil.UsageError(cmd, "POD is required for log")
	}

	if len(args) > 2 {
		return cmdutil.UsageError(cmd, "log POD [CONTAINER]")
	}

	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	client, err := f.Client()
	if err != nil {
		return err
	}

	podID := args[0]

	pod, err := client.Pods(namespace).Get(podID)
	if err != nil {
		return err
	}

	var container string
	if cmdutil.GetFlagString(cmd, "container") != "" {
		// [-c CONTAINER]
		container = p.containerName
	} else {
		// [CONTAINER] (container as arg not flag) is supported as legacy behavior. See PR #10519 for more details.
		if len(args) == 1 {
			if len(pod.Spec.Containers) != 1 {
				return fmt.Errorf("POD %s has more than one container; please specify the container to print logs for", pod.ObjectMeta.Name)
			}
			container = pod.Spec.Containers[0].Name
		} else {
			container = args[1]
		}
	}

	follow := false
	if cmdutil.GetFlagBool(cmd, "follow") {
		follow = true
	}

	previous := false
	if cmdutil.GetFlagBool(cmd, "previous") {
		previous = true
	}

	readCloser, err := client.RESTClient.Get().
		Namespace(namespace).
		Name(podID).
		Resource("pods").
		SubResource("log").
		Param("follow", strconv.FormatBool(follow)).
		Param("container", container).
		Param("previous", strconv.FormatBool(previous)).
		Stream()
	if err != nil {
		return err
	}

	defer readCloser.Close()
	_, err = io.Copy(out, readCloser)
	return err
}
