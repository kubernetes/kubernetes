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
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/polymorphichelpers"
	"k8s.io/kubernetes/pkg/kubectl/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	logsExample = templates.Examples(i18n.T(`
		# Return snapshot logs from pod nginx with only one container
		kubectl logs nginx

		# Return snapshot logs from pod nginx with multi containers
		kubectl logs nginx --all-containers=true

		# Return snapshot logs from all containers in pods defined by label app=nginx
		kubectl logs -lapp=nginx --all-containers=true

		# Return snapshot of previous terminated ruby container logs from pod web-1
		kubectl logs -p -c ruby web-1

		# Begin streaming the logs of the ruby container in pod web-1
		kubectl logs -f -c ruby web-1

		# Display only the most recent 20 lines of output in pod nginx
		kubectl logs --tail=20 nginx

		# Show all logs from pod nginx written in the last hour
		kubectl logs --since=1h nginx

		# Return snapshot logs from first container of a job named hello
		kubectl logs job/hello

		# Return snapshot logs from container nginx-1 of a deployment named nginx
		kubectl logs deployment/nginx -c nginx-1`))

	selectorTail int64 = 10
)

const (
	logsUsageStr = "expected 'logs (POD | TYPE/NAME) [CONTAINER_NAME]'.\nPOD or TYPE/NAME is a required argument for the logs command"
)

type LogsOptions struct {
	Namespace     string
	ResourceArg   string
	AllContainers bool
	Options       runtime.Object

	Object           runtime.Object
	GetPodTimeout    time.Duration
	RESTClientGetter genericclioptions.RESTClientGetter
	LogsForObject    polymorphichelpers.LogsForObjectFunc

	genericclioptions.IOStreams
}

func NewLogsOptions(streams genericclioptions.IOStreams, allContainers bool) *LogsOptions {
	return &LogsOptions{
		IOStreams:     streams,
		AllContainers: allContainers,
	}
}

// NewCmdLogs creates a new pod logs command
func NewCmdLogs(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewLogsOptions(streams, false)

	cmd := &cobra.Command{
		Use: "logs [-f] [-p] (POD | TYPE/NAME) [-c CONTAINER]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Print the logs for a container in a pod"),
		Long:    "Print the logs for a container in a pod or specified resource. If the pod has only one container, the container name is optional.",
		Example: logsExample,
		PreRun: func(cmd *cobra.Command, args []string) {
			if len(os.Args) > 1 && os.Args[1] == "log" {
				printDeprecationWarning(o.ErrOut, "logs", "log")
			}
		},
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunLogs())
		},
		Aliases: []string{"log"},
	}
	cmd.Flags().BoolVar(&o.AllContainers, "all-containers", o.AllContainers, "Get all containers's logs in the pod(s).")
	cmd.Flags().BoolP("follow", "f", false, "Specify if the logs should be streamed.")
	cmd.Flags().Bool("timestamps", false, "Include timestamps on each line in the log output")
	cmd.Flags().Int64("limit-bytes", 0, "Maximum bytes of logs to return. Defaults to no limit.")
	cmd.Flags().BoolP("previous", "p", false, "If true, print the logs for the previous instance of the container in a pod if it exists.")
	cmd.Flags().Int64("tail", -1, "Lines of recent log file to display. Defaults to -1 with no selector, showing all log lines otherwise 10, if a selector is provided.")
	cmd.Flags().String("since-time", "", i18n.T("Only return logs after a specific date (RFC3339). Defaults to all logs. Only one of since-time / since may be used."))
	cmd.Flags().Duration("since", 0, "Only return logs newer than a relative duration like 5s, 2m, or 3h. Defaults to all logs. Only one of since-time / since may be used.")
	cmd.Flags().StringP("container", "c", "", "Print the logs of this container")
	cmd.Flags().Bool("interactive", false, "If true, prompt the user for input when required.")
	cmd.Flags().MarkDeprecated("interactive", "This flag is no longer respected and there is no replacement.")
	cmdutil.AddPodRunningTimeoutFlag(cmd, defaultPodLogsTimeout)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on.")
	return cmd
}

func (o *LogsOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	containerName := cmdutil.GetFlagString(cmd, "container")
	selector := cmdutil.GetFlagString(cmd, "selector")
	switch len(args) {
	case 0:
		if len(selector) == 0 {
			return cmdutil.UsageErrorf(cmd, "%s", logsUsageStr)
		}
	case 1:
		o.ResourceArg = args[0]
		if len(selector) != 0 {
			return cmdutil.UsageErrorf(cmd, "only a selector (-l) or a POD name is allowed")
		}
	case 2:
		if cmd.Flag("container").Changed {
			return cmdutil.UsageErrorf(cmd, "only one of -c or an inline [CONTAINER] arg is allowed")
		}
		o.ResourceArg = args[0]
		containerName = args[1]
	default:
		return cmdutil.UsageErrorf(cmd, "%s", logsUsageStr)
	}
	var err error
	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
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
		t, err := util.ParseRFC3339(sinceTime, metav1.Now)
		if err != nil {
			return err
		}
		logOptions.SinceTime = &t
	}
	if limit := cmdutil.GetFlagInt64(cmd, "limit-bytes"); limit != 0 {
		logOptions.LimitBytes = &limit
	}
	tail := cmdutil.GetFlagInt64(cmd, "tail")
	if tail != -1 {
		logOptions.TailLines = &tail
	}
	if sinceSeconds := cmdutil.GetFlagDuration(cmd, "since"); sinceSeconds != 0 {
		// round up to the nearest second
		sec := int64(sinceSeconds.Round(time.Second).Seconds())
		logOptions.SinceSeconds = &sec
	}
	o.GetPodTimeout, err = cmdutil.GetPodRunningTimeoutFlag(cmd)
	if err != nil {
		return err
	}
	o.Options = logOptions
	o.RESTClientGetter = f
	o.LogsForObject = polymorphichelpers.LogsForObjectFn

	if len(selector) != 0 {
		if logOptions.Follow {
			return cmdutil.UsageErrorf(cmd, "only one of follow (-f) or selector (-l) is allowed")
		}
		if logOptions.TailLines == nil && tail != -1 {
			logOptions.TailLines = &selectorTail
		}
	}

	if o.Object == nil {
		builder := f.NewBuilder().
			WithScheme(legacyscheme.Scheme).
			NamespaceParam(o.Namespace).DefaultNamespace().
			SingleResourceType()
		if o.ResourceArg != "" {
			builder.ResourceNames("pods", o.ResourceArg)
		}
		if selector != "" {
			builder.ResourceTypes("pods").LabelSelectorParam(selector)
		}
		infos, err := builder.Do().Infos()
		if err != nil {
			return err
		}
		if selector == "" && len(infos) != 1 {
			return errors.New("expected a resource")
		}
		o.Object = infos[0].Object
	}

	return nil
}

func (o LogsOptions) Validate() error {
	logsOptions, ok := o.Options.(*api.PodLogOptions)
	if !ok {
		return errors.New("unexpected logs options object")
	}
	if o.AllContainers && len(logsOptions.Container) > 0 {
		return fmt.Errorf("--all-containers=true should not be specified with container name %s", logsOptions.Container)
	}
	if errs := validation.ValidatePodLogOptions(logsOptions); len(errs) > 0 {
		return errs.ToAggregate()
	}

	return nil
}

// RunLogs retrieves a pod log
func (o LogsOptions) RunLogs() error {
	switch t := o.Object.(type) {
	case *api.PodList:
		for _, p := range t.Items {
			if err := o.getPodLogs(&p); err != nil {
				return err
			}
		}
		return nil
	case *api.Pod:
		return o.getPodLogs(t)
	default:
		return o.getLogs(o.Object)
	}
}

// getPodLogs checks whether o.AllContainers is set to true.
// If so, it retrives all containers' log in the pod.
func (o LogsOptions) getPodLogs(pod *api.Pod) error {
	if !o.AllContainers {
		return o.getLogs(pod)
	}

	for _, c := range pod.Spec.InitContainers {
		o.Options.(*api.PodLogOptions).Container = c.Name
		if err := o.getLogs(pod); err != nil {
			return err
		}
	}
	for _, c := range pod.Spec.Containers {
		o.Options.(*api.PodLogOptions).Container = c.Name
		if err := o.getLogs(pod); err != nil {
			return err
		}
	}
	return nil
}

func (o LogsOptions) getLogs(obj runtime.Object) error {
	req, err := o.LogsForObject(o.RESTClientGetter, obj, o.Options, o.GetPodTimeout)
	if err != nil {
		return err
	}

	readCloser, err := req.Stream()
	if err != nil {
		return err
	}
	defer readCloser.Close()

	_, err = io.Copy(o.Out, readCloser)
	return err
}
