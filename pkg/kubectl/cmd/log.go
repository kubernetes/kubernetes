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
	"errors"
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

type LogsOptions struct {
	Client *client.Client

	PodNamespace  string
	PodName       string
	ContainerName string
	Follow        bool
	Timestamps    bool
	Previous      bool
	LimitBytes    int
	Tail          int
	SinceTime     *unversioned.Time
	SinceSeconds  time.Duration

	Out io.Writer
}

// NewCmdLog creates a new pod log command
func NewCmdLog(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	o := &LogsOptions{
		Out:  out,
		Tail: -1,
	}

	cmd := &cobra.Command{
		Use:     "logs [-f] [-p] POD [-c CONTAINER]",
		Short:   "Print the logs for a container in a pod.",
		Long:    "Print the logs for a container in a pod. If the pod has only one container, the container name is optional.",
		Example: log_example,
		Run: func(cmd *cobra.Command, args []string) {
			if len(os.Args) > 1 && os.Args[1] == "log" {
				printDeprecationWarning("logs", "log")
			}

			cmdutil.CheckErr(o.Complete(f, out, cmd, args))
			if err := o.Validate(); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
			cmdutil.CheckErr(o.RunLog())
		},
		Aliases: []string{"log"},
	}
	cmd.Flags().BoolVarP(&o.Follow, "follow", "f", o.Follow, "Specify if the logs should be streamed.")
	cmd.Flags().BoolVar(&o.Timestamps, "timestamps", o.Timestamps, "Include timestamps on each line in the log output")
	cmd.Flags().Bool("interactive", true, "If true, prompt the user for input when required. Default true.")
	cmd.Flags().MarkDeprecated("interactive", "This flag is no longer respected and there is no replacement.")
	cmd.Flags().IntVar(&o.LimitBytes, "limit-bytes", o.LimitBytes, "Maximum bytes of logs to return. Defaults to no limit.")
	cmd.Flags().BoolVarP(&o.Previous, "previous", "p", o.Previous, "If true, print the logs for the previous instance of the container in a pod if it exists.")
	cmd.Flags().IntVar(&o.Tail, "tail", o.Tail, "Lines of recent log file to display. Defaults to -1, showing all log lines.")
	cmd.Flags().String("since-time", "", "Only return logs after a specific date (RFC3339). Defaults to all logs. Only one of since-time / since may be used.")
	cmd.Flags().DurationVar(&o.SinceSeconds, "since", o.SinceSeconds, "Only return logs newer than a relative duration like 5s, 2m, or 3h. Defaults to all logs. Only one of since-time / since may be used.")
	cmd.Flags().StringVarP(&o.ContainerName, "container", "c", o.ContainerName, "Container name")
	return cmd
}

func (o *LogsOptions) Complete(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	switch len(args) {
	case 0:
		return cmdutil.UsageError(cmd, "POD is required for log")

	case 1:
		o.PodName = args[0]
	case 2:
		if cmd.Flag("container").Changed {
			return cmdutil.UsageError(cmd, "only one of -c, [CONTAINER] arg is allowed")
		}
		o.PodName = args[0]
		o.ContainerName = args[1]

	default:
		return cmdutil.UsageError(cmd, "log POD [-c CONTAINER]")
	}

	var err error
	o.PodNamespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}
	o.Client, err = f.Client()
	if err != nil {
		return err
	}

	sinceTime := cmdutil.GetFlagString(cmd, "since-time")
	if len(sinceTime) > 0 {
		t, err := api.ParseRFC3339(sinceTime, unversioned.Now)
		if err != nil {
			return err
		}
		o.SinceTime = &t
	}

	return nil
}

func (o *LogsOptions) Validate() error {
	if len(o.PodName) == 0 {
		return errors.New("POD must be specified")
	}
	if o.LimitBytes < 0 {
		return errors.New("--limit-bytes must be greater than or equal to zero")
	}
	if o.Tail < -1 {
		return errors.New("--tail must be greater than or equal to -1")
	}
	if o.SinceTime != nil && o.SinceSeconds > 0 {
		return errors.New("only one of --since, --since-time may be specified")
	}

	return nil
}

// RunLog retrieves a pod log
func (o *LogsOptions) RunLog() error {
	pod, err := o.Client.Pods(o.PodNamespace).Get(o.PodName)
	if err != nil {
		return err
	}

	// [-c CONTAINER]
	container := o.ContainerName
	if len(container) == 0 {
		// [CONTAINER] (container as arg not flag) is supported as legacy behavior. See PR #10519 for more details.
		if len(pod.Spec.Containers) != 1 {
			podContainersNames := []string{}
			for _, container := range pod.Spec.Containers {
				podContainersNames = append(podContainersNames, container.Name)
			}

			return fmt.Errorf("Pod %s has the following containers: %s; please specify the container to print logs for with -c", pod.ObjectMeta.Name, strings.Join(podContainersNames, ", "))
		}
		container = pod.Spec.Containers[0].Name
	}

	logOptions := &api.PodLogOptions{
		Container:  container,
		Follow:     o.Follow,
		Previous:   o.Previous,
		Timestamps: o.Timestamps,
	}
	if o.SinceSeconds > 0 {
		// round up to the nearest second
		sec := int64(math.Ceil(float64(o.SinceSeconds) / float64(time.Second)))
		logOptions.SinceSeconds = &sec
	}
	logOptions.SinceTime = o.SinceTime
	if o.LimitBytes != 0 {
		i := int64(o.LimitBytes)
		logOptions.LimitBytes = &i
	}
	if o.Tail >= 0 {
		i := int64(o.Tail)
		logOptions.TailLines = &i
	}

	return handleLog(o.Client, o.PodNamespace, o.PodName, logOptions, o.Out)
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
