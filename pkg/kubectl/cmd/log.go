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
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	log_example = `# Return snapshot logs from pod nginx with only one container
$ kubectl logs nginx

# Return snapshot of previous terminated ruby container logs from pod web-1
$ kubectl logs -p -c ruby web-1

# Begin streaming the logs of the ruby container in pod web-1
$ kubectl logs -f -c ruby web-1

# Display only the most recent 20 lines of output in pod nginx
$ kubectl logs --tail=20 nginx

# Show all logs from pod nginx written in the last hour
$ kubectl logs --since=1h nginx`
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
	cmd.Flags().Bool("timestamps", false, "Include timestamps on each line in the log output")
	cmd.Flags().Bool("interactive", true, "If true, prompt the user for input when required. Default true.")
	cmd.Flags().BoolP("previous", "p", false, "If true, print the logs for the previous instance of the container in a pod if it exists.")
	cmd.Flags().Int("limit-bytes", 0, "Maximum bytes of logs to return. Defaults to no limit.")
	cmd.Flags().Int("tail", -1, "Lines of recent log file to display. Defaults to -1, showing all log lines.")
	cmd.Flags().String("since-time", "", "Only return logs after a specific date (RFC3339). Defaults to all logs. Only one of since-time / since may be used.")
	cmd.Flags().Duration("since", 0, "Only return logs newer than a relative duration like 5s, 2m, or 3h. Defaults to all logs. Only one of since-time / since may be used.")
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

	sinceSeconds := cmdutil.GetFlagDuration(cmd, "since")
	sinceTime := cmdutil.GetFlagString(cmd, "since-time")
	if len(sinceTime) > 0 && sinceSeconds > 0 {
		return cmdutil.UsageError(cmd, "only one of --since, --since-time may be specified")
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

	logOptions := &api.PodLogOptions{
		Container:  container,
		Follow:     cmdutil.GetFlagBool(cmd, "follow"),
		Previous:   cmdutil.GetFlagBool(cmd, "previous"),
		Timestamps: cmdutil.GetFlagBool(cmd, "timestamps"),
	}
	if sinceSeconds > 0 {
		// round up to the nearest second
		sec := int64(math.Ceil(float64(sinceSeconds) / float64(time.Second)))
		logOptions.SinceSeconds = &sec
	}
	if t, err := api.ParseRFC3339(sinceTime, unversioned.Now); err == nil {
		logOptions.SinceTime = &t
	}
	if limitBytes := cmdutil.GetFlagInt(cmd, "limit-bytes"); limitBytes != 0 {
		i := int64(limitBytes)
		logOptions.LimitBytes = &i
	}
	if tail := cmdutil.GetFlagInt(cmd, "tail"); tail >= 0 {
		i := int64(tail)
		logOptions.TailLines = &i
	}

	return handleLog(client, namespace, podID, logOptions, out)
}

func handleLog(client *client.Client, namespace, podID string, logOptions *api.PodLogOptions, out io.Writer) error {
	// TODO: transform this into a PodLogOptions call
	req := client.RESTClient.Get().
		Namespace(namespace).
		Name(podID).
		Resource("pods").
		SubResource("log").
		Param("follow", strconv.FormatBool(logOptions.Follow)).
		Param("container", logOptions.Container).
		Param("previous", strconv.FormatBool(logOptions.Previous)).
		Param("timestamps", strconv.FormatBool(logOptions.Timestamps))

	if logOptions.SinceSeconds != nil {
		req.Param("sinceSeconds", strconv.FormatInt(*logOptions.SinceSeconds, 10))
	}
	if logOptions.SinceTime != nil {
		req.Param("sinceTime", logOptions.SinceTime.Format(time.RFC3339))
	}
	if logOptions.LimitBytes != nil {
		req.Param("limitBytes", strconv.FormatInt(*logOptions.LimitBytes, 10))
	}
	if logOptions.TailLines != nil {
		req.Param("tailLines", strconv.FormatInt(*logOptions.TailLines, 10))
	}
	readCloser, err := req.Stream()
	if err != nil {
		return err
	}

	defer readCloser.Close()
	_, err = io.Copy(out, readCloser)
	return err
}
