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
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	log_example = `# Return snapshot of ruby-container logs from pod 123456-7890.
$ kubectl logs 123456-7890 ruby-container

# Return snapshot of previous terminated ruby-container logs from pod 123456-7890.
$ kubectl logs -p 123456-7890 ruby-container

# Start streaming of ruby-container logs from pod 123456-7890.
$ kubectl logs -f 123456-7890 ruby-container`
)

func selectContainer(pod *api.Pod, in io.Reader, out io.Writer) string {
	fmt.Fprintf(out, "Please select a container:\n")
	options := sets.String{}
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

	// [-c CONTAINER]
	container := p.containerName
	if len(container) == 0 {
		// [CONTAINER] (container as arg not flag) is supported as legacy behavior. See PR #10519 for more details.
		if len(args) == 1 {
			if len(pod.Spec.Containers) != 1 {
				podContainersNames := []string{}
				for _, container := range pod.Spec.Containers {
					podContainersNames = append(podContainersNames, container.Name)
				}

				return fmt.Errorf("Pod %s has the following containers: %s; please specify the container to print logs for with -c", pod.ObjectMeta.Name, strings.Join(podContainersNames, ", "))
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
	return handleLog(client, namespace, podID, container, follow, previous, out)
}

func handleLog(client *client.Client, namespace, podID, container string, follow, previous bool, out io.Writer) error {
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
