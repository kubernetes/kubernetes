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
	"io"
	"math"
	"os"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	kerrors "k8s.io/kubernetes/pkg/util/errors"
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
	Namespace   string
	ResourceArg string
	Options     runtime.Object

	Mapper       meta.RESTMapper
	Typer        runtime.ObjectTyper
	ClientMapper resource.ClientMapper

	LogsForObject func(object, options runtime.Object) (*client.Request, error)

	Out io.Writer
}

// NewCmdLog creates a new pod log command
func NewCmdLog(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	o := &LogsOptions{}
	cmd := &cobra.Command{
		Use:     "logs [-f] [-p] POD [-c CONTAINER]",
		Short:   "Print the logs for a container in a pod.",
		Long:    "Print the logs for a container in a pod. If the pod has only one container, the container name is optional.",
		Example: log_example,
		PreRun: func(cmd *cobra.Command, args []string) {
			if len(os.Args) > 1 && os.Args[1] == "log" {
				printDeprecationWarning("logs", "log")
			}
		},
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, out, cmd, args))
			if err := o.Validate(); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
			_, err := o.RunLog()
			cmdutil.CheckErr(err)
		},
		Aliases: []string{"log"},
	}
	cmd.Flags().BoolP("follow", "f", false, "Specify if the logs should be streamed.")
	cmd.Flags().Bool("timestamps", false, "Include timestamps on each line in the log output")
	cmd.Flags().Int64("limit-bytes", 0, "Maximum bytes of logs to return. Defaults to no limit.")
	cmd.Flags().BoolP("previous", "p", false, "If true, print the logs for the previous instance of the container in a pod if it exists.")
	cmd.Flags().Int64("tail", -1, "Lines of recent log file to display. Defaults to -1, showing all log lines.")
	cmd.Flags().String("since-time", "", "Only return logs after a specific date (RFC3339). Defaults to all logs. Only one of since-time / since may be used.")
	cmd.Flags().Duration("since", 0, "Only return logs newer than a relative duration like 5s, 2m, or 3h. Defaults to all logs. Only one of since-time / since may be used.")
	cmd.Flags().StringP("container", "c", "", "Print the logs of this container")

	cmd.Flags().Bool("interactive", false, "If true, prompt the user for input when required.")
	cmd.Flags().MarkDeprecated("interactive", "This flag is no longer respected and there is no replacement.")
	return cmd
}

func (o *LogsOptions) Complete(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	containerName := cmdutil.GetFlagString(cmd, "container")
	switch len(args) {
	case 0:
		return cmdutil.UsageError(cmd, "POD is required for logs")
	case 1:
		o.ResourceArg = args[0]
	case 2:
		if cmd.Flag("container").Changed {
			return cmdutil.UsageError(cmd, "only one of -c, [CONTAINER] arg is allowed")
		}
		o.ResourceArg = args[0]
		containerName = args[1]
	default:
		return cmdutil.UsageError(cmd, "logs POD [-c CONTAINER]")
	}
	var err error
	o.Namespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}

	logOptions := &api.PodLogOptions{
		Container:  containerName,
		Follow:     cmdutil.GetFlagBool(cmd, "follow"),
		Previous:   cmdutil.GetFlagBool(cmd, "previous"),
		Timestamps: cmdutil.GetFlagBool(cmd, "timestamps"),
	}
	if sinceTime := cmdutil.GetFlagString(cmd, "since-time"); len(sinceTime) > 0 {
		t, err := api.ParseRFC3339(sinceTime, unversioned.Now)
		if err != nil {
			return err
		}
		logOptions.SinceTime = &t
	}
	if limit := cmdutil.GetFlagInt64(cmd, "limit-bytes"); limit != 0 {
		logOptions.LimitBytes = &limit
	}
	if tail := cmdutil.GetFlagInt64(cmd, "tail"); tail != -1 {
		logOptions.TailLines = &tail
	}
	if sinceSeconds := cmdutil.GetFlagDuration(cmd, "since"); sinceSeconds != 0 {
		// round up to the nearest second
		sec := int64(math.Ceil(float64(sinceSeconds) / float64(time.Second)))
		logOptions.SinceSeconds = &sec
	}
	o.Options = logOptions

	o.Mapper, o.Typer = f.Object()
	o.ClientMapper = f.ClientMapperForCommand()
	o.LogsForObject = f.LogsForObject

	o.Out = out

	return nil
}

func (o LogsOptions) Validate() error {
	if len(o.ResourceArg) == 0 {
		return errors.New("a pod must be specified")
	}
	logOptions, ok := o.Options.(*api.PodLogOptions)
	if !ok {
		return errors.New("unexpected log options object")
	}
	if errs := validation.ValidatePodLogOptions(logOptions); len(errs) > 0 {
		return kerrors.NewAggregate(errs)
	}

	return nil
}

// RunLog retrieves a pod log
func (o LogsOptions) RunLog() (int64, error) {
	infos, err := resource.NewBuilder(o.Mapper, o.Typer, o.ClientMapper).
		NamespaceParam(o.Namespace).DefaultNamespace().
		ResourceNames("pods", o.ResourceArg).
		SingleResourceType().
		Do().Infos()
	if err != nil {
		return 0, err
	}
	if len(infos) != 1 {
		return 0, errors.New("expected a resource")
	}
	info := infos[0]

	req, err := o.LogsForObject(info.Object, o.Options)
	if err != nil {
		return 0, err
	}

	readCloser, err := req.Stream()
	if err != nil {
		return 0, err
	}
	defer readCloser.Close()

	return io.Copy(o.Out, readCloser)
}
