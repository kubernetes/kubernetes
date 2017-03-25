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
	"math"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	logs_example = templates.Examples(`
		# Return snapshot logs from pod nginx with only one container
		kubectl logs nginx

		# Return snapshot logs for the pods defined by label app=nginx
		kubectl logs -lapp=nginx

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
		kubectl logs deployment/nginx -c nginx-1`)

	selectorTail int64 = 10
)

const (
	logsUsageStr = "expected 'logs (POD | TYPE/NAME) [CONTAINER_NAME]'.\nPOD or TYPE/NAME is a required argument for the logs command"
)

type LogsOptions struct {
	Namespace   string
	ResourceArg string
	Selector    string
	Options     runtime.Object

	Mapper       meta.RESTMapper
	Typer        runtime.ObjectTyper
	ClientMapper resource.ClientMapper
	Decoder      runtime.Decoder

	Object          runtime.Object
	PrefixMatchList []string
	LogsForObject   func(object, options runtime.Object) (*restclient.Request, error)

	Out io.Writer
}

// NewCmdLog creates a new pod logs command
func NewCmdLogs(f cmdutil.Factory, out io.Writer) *cobra.Command {
	o := &LogsOptions{}
	cmd := &cobra.Command{
		Use:     "logs [-f] [-p] (POD | TYPE/NAME) [-c CONTAINER]",
		Short:   i18n.T("Print the logs for a container in a pod"),
		Long:    "Print the logs for a container in a pod or specified resource. If the pod has only one container, the container name is optional.",
		Example: logs_example,
		PreRun: func(cmd *cobra.Command, args []string) {
			if len(os.Args) > 1 && os.Args[1] == "log" {
				printDeprecationWarning("logs", "log")
			}
		},
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, out, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunLogs(out, args))
		},
		Aliases: []string{"log"},
	}
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
	cmdutil.AddInclude3rdPartyFlags(cmd)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on.")
	return cmd
}

func (o *LogsOptions) Complete(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	containerName := cmdutil.GetFlagString(cmd, "container")
	o.Selector = cmdutil.GetFlagString(cmd, "selector")
	switch len(args) {
	case 0:
		if len(o.Selector) == 0 {
			return cmdutil.UsageError(cmd, logsUsageStr)
		}
	case 1:
		o.ResourceArg = args[0]
		if len(o.Selector) != 0 {
			return cmdutil.UsageError(cmd, "only a selector (-l) or a POD name is allowed")
		}
	case 2:
		if cmd.Flag("container").Changed {
			return cmdutil.UsageError(cmd, "only one of -c or an inline [CONTAINER] arg is allowed")
		}
		o.ResourceArg = args[0]
		containerName = args[1]
	default:
		return cmdutil.UsageError(cmd, logsUsageStr)
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
		t, err := api.ParseRFC3339(sinceTime, metav1.Now)
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
	o.LogsForObject = f.LogsForObject
	o.ClientMapper = resource.ClientMapperFunc(f.ClientForMapping)
	o.Out = out

	if len(o.Selector) != 0 {
		if logOptions.Follow {
			return cmdutil.UsageError(cmd, "only one of follow (-f) or selector (-l) is allowed")
		}
		if len(logOptions.Container) != 0 {
			return cmdutil.UsageError(cmd, "a container cannot be specified when using a selector (-l)")
		}
		if logOptions.TailLines == nil {
			logOptions.TailLines = &selectorTail
		}
	}

	o.Mapper, o.Typer = f.Object()
	o.Decoder = f.Decoder(true)
	if o.Object == nil {
		builder := resource.NewBuilder(o.Mapper, o.Typer, o.ClientMapper, o.Decoder).
			NamespaceParam(o.Namespace).DefaultNamespace().
			SingleResourceType()
		if o.ResourceArg != "" {
			builder.ResourceNames("pods", o.ResourceArg)
		}
		if o.Selector != "" {
			builder.ResourceTypes("pods").SelectorParam(o.Selector)
		}
		infos, err := builder.Do().Infos()
		// if name not found use prefix match
		aggErrs := []error{}
		if err != nil {
			aggErrs = append(aggErrs, err)
			if apierrors.IsNotFound(err) && o.ResourceArg != "" {
				err = o.prefixMatchResourceList(simpleTrimResourceTypeName(o.ResourceArg), err)
				if err != nil {
					aggErrs = append(aggErrs, err)
					return utilerrors.NewAggregate(aggErrs)
				}
			}
		}

		if o.Selector == "" && len(infos) != 1 {
			return errors.New("expected a resource")
		}

		if o.Object == nil {
			o.Object = infos[0].Object
		}
	}
	return nil
}

func simpleTrimResourceTypeName(s string) string {
	if strings.Contains(s, "/") {
		seg := strings.Split(s, "/")
		return seg[0]
	}
	return s
}

func (o LogsOptions) prefixMatchResourceList(resourceType string, originalError error) error {
	infos, err := resource.NewBuilder(o.Mapper, o.Typer, o.ClientMapper, o.Decoder).
		NamespaceParam(o.Namespace).DefaultNamespace().
		SingleResourceType().ResourceTypeOrNameArgs(true, resourceType).
		Flatten().Do().Infos()
	if err != nil {
		return err
	}
	//match pods using prefix
	prefixTimes := 0
	for i, info := range infos {
		if strings.HasPrefix(info.Name, o.ResourceArg) {
			o.PrefixMatchList = append(o.PrefixMatchList, info.Name)
			o.Object = infos[i].Object
			prefixTimes++
		}
	}

	if prefixTimes == 0 {
		return originalError
	}

	return nil
}

func (o LogsOptions) Validate() error {
	logsOptions, ok := o.Options.(*api.PodLogOptions)
	if !ok {
		return errors.New("unexpected logs options object")
	}
	if errs := validation.ValidatePodLogOptions(logsOptions); len(errs) > 0 {
		return errs.ToAggregate()
	}

	return nil
}

// RunLogs retrieves a pod log
func (o LogsOptions) RunLogs(out io.Writer, args []string) error {
	if matchLen := len(o.PrefixMatchList); matchLen > 1 {
		if matchLen > 5 {
			o.PrefixMatchList = o.PrefixMatchList[:5]
		}
		first := true
		fmt.Fprintf(out, "No pod named %s, did you mean? ", args[0])
		for i := range o.PrefixMatchList {
			if first {
				fmt.Fprintf(out, "%s", o.PrefixMatchList[i])
				first = false
			} else {
				fmt.Fprintf(out, ", %s", o.PrefixMatchList[i])
			}
		}
		fmt.Fprintf(out, "\n")
	} else {
		switch t := o.Object.(type) {
		case *api.PodList:
			for _, p := range t.Items {
				if err := o.getLogs(&p); err != nil {
					return err
				}
			}
			return nil
		default:
			return o.getLogs(o.Object)
		}
	}

	return nil
}

func (o LogsOptions) getLogs(obj runtime.Object) error {
	req, err := o.LogsForObject(obj, o.Options)
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
