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

package logs

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sync"
	"time"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

const (
	logsUsageStr = "logs [-f] [-p] (POD | TYPE/NAME) [-c CONTAINER]"
)

var (
	logsLong = templates.LongDesc(i18n.T(`
		Print the logs for a container in a pod or specified resource. 
		If the pod has only one container, the container name is optional.`))

	logsExample = templates.Examples(i18n.T(`
		# Return snapshot logs from pod nginx with only one container
		kubectl logs nginx

		# Return snapshot logs from pod nginx with multi containers
		kubectl logs nginx --all-containers=true

		# Return snapshot logs from all containers in pods defined by label app=nginx
		kubectl logs -l app=nginx --all-containers=true

		# Return snapshot of previous terminated ruby container logs from pod web-1
		kubectl logs -p -c ruby web-1

		# Begin streaming the logs of the ruby container in pod web-1
		kubectl logs -f -c ruby web-1

		# Begin streaming the logs from all containers in pods defined by label app=nginx
		kubectl logs -f -l app=nginx --all-containers=true

		# Display only the most recent 20 lines of output in pod nginx
		kubectl logs --tail=20 nginx

		# Show all logs from pod nginx written in the last hour
		kubectl logs --since=1h nginx

		# Show logs from a kubelet with an expired serving certificate
		kubectl logs --insecure-skip-tls-verify-backend nginx

		# Return snapshot logs from first container of a job named hello
		kubectl logs job/hello

		# Return snapshot logs from container nginx-1 of a deployment named nginx
		kubectl logs deployment/nginx -c nginx-1`))

	selectorTail    int64 = 10
	logsUsageErrStr       = fmt.Sprintf("expected '%s'.\nPOD or TYPE/NAME is a required argument for the logs command", logsUsageStr)
)

const (
	defaultPodLogsTimeout = 20 * time.Second
)

var defaultContainerNameFromRefSpecRegexp = regexp.MustCompile(`spec\.(?:initContainers|containers|ephemeralContainers){(.+)}`)

// LogsFlags reflects the information that the CLI is gathering via flags.
// They will be converted to PodLogOptions for runtime requirements.
type LogsFlags struct {
	// required objects
	RESTClientGetter genericclioptions.RESTClientGetter
	ResourceBuilder  *resource.Builder

	// flags
	AllContainers                bool
	Container                    string
	Follow                       bool
	IgnoreLogErrors              bool
	InsecureSkipTLSVerifyBackend bool
	LimitBytes                   int64
	MaxFollowConcurrency         int
	Prefix                       bool
	Previous                     bool
	PodRunningTimeout            time.Duration
	Selector                     string
	Since                        time.Duration
	SinceTime                    string
	Tail                         int64
	Timestamps                   bool

	// internal items
	containerNameSpecified bool
	namespace              string

	genericclioptions.IOStreams
}

// NewLogsFlags returns a default LogsFlags.
func NewLogsFlags(f cmdutil.Factory, streams genericclioptions.IOStreams) *LogsFlags {
	namespace, _, _ := f.ToRawKubeConfigLoader().Namespace()

	return &LogsFlags{
		RESTClientGetter: f,
		ResourceBuilder: f.NewBuilder().SingleResourceType().
			WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...),
		Tail:                 -1,
		MaxFollowConcurrency: 5,
		namespace:            namespace,
		IOStreams:            streams,
	}
}

// ToOptions converts from CLI inputs to runtime inputs.
func (flags *LogsFlags) ToOptions(cmd *cobra.Command, args []string) (*LogsOptions, error) {
	flags.containerNameSpecified = cmd.Flag("container").Changed

	o := &LogsOptions{
		RESTClientGetter: flags.RESTClientGetter,
		ResourceBuilder:  flags.ResourceBuilder,
		LogsForObject:    polymorphichelpers.LogsForObjectFn,
		Namespace:        flags.namespace,
		Resources:        args,
		IOStreams:        flags.IOStreams,

		AllContainers:                flags.AllContainers,
		Container:                    flags.Container,
		Follow:                       flags.Follow,
		IgnoreLogErrors:              flags.IgnoreLogErrors,
		InsecureSkipTLSVerifyBackend: flags.InsecureSkipTLSVerifyBackend,
		LimitBytes:                   flags.LimitBytes,
		MaxFollowConcurrency:         flags.MaxFollowConcurrency,
		Prefix:                       flags.Prefix,
		Previous:                     flags.Previous,
		Selector:                     flags.Selector,
		SinceTime:                    flags.SinceTime,
		Since:                        flags.Since,
		Tail:                         flags.Tail,
		Timestamps:                   flags.Timestamps,

		ContainerNameSpecified: flags.containerNameSpecified,
		TailSpecified:          cmd.Flag("tail").Changed,

		containerNameFromRefSpecRegexp: defaultContainerNameFromRefSpecRegexp,
	}

	switch len(args) {
	case 0:
		if len(flags.Selector) == 0 {
			return nil, cmdutil.UsageErrorf(cmd, "%s", logsUsageErrStr)
		}
	case 1:
		o.ResourceArg = args[0]
		if len(flags.Selector) != 0 {
			return nil, cmdutil.UsageErrorf(cmd, "only a selector (-l) or a POD name is allowed")
		}
	case 2:
		o.ResourceArg = args[0]
		o.Container = args[1]
	default:
		return nil, cmdutil.UsageErrorf(cmd, "%s", logsUsageErrStr)
	}

	o.ResourceBuilder.NamespaceParam(o.Namespace)

	if len(flags.Selector) > 0 {
		o.ResourceBuilder.ResourceTypes("pods").LabelSelectorParam(o.Selector)
	}

	if len(o.ResourceArg) > 0 {
		o.ResourceBuilder.ResourceNames("pods", o.ResourceArg)
	}

	var err error
	o.ConsumeRequestFn = DefaultConsumeRequest
	o.GetPodTimeout, err = cmdutil.GetPodRunningTimeoutFlag(cmd)
	if err != nil {
		return nil, err
	}

	o.Options, err = o.ToLogOptions()
	if err != nil {
		return nil, err
	}

	infos, err := o.ResourceBuilder.Do().Infos()
	if err != nil {
		return nil, err
	}
	if o.Selector == "" && len(infos) != 1 {
		return nil, fmt.Errorf("expected a resource")
	}
	o.Object = infos[0].Object
	if o.Selector != "" && len(o.Object.(*corev1.PodList).Items) == 0 {
		fmt.Fprintf(o.ErrOut, "No resources found in %s namespace.\n", o.Namespace)
	}

	return o, o.Validate()
}

// AddFlags registers flags for the CLI.
func (flags *LogsFlags) AddFlags(cmd *cobra.Command) {
	cmdutil.AddPodRunningTimeoutFlag(cmd, defaultPodLogsTimeout)

	cmd.Flags().BoolVar(&flags.AllContainers, "all-containers", flags.AllContainers, "Get all containers' logs in the pod(s).")
	cmd.Flags().StringVarP(&flags.Container, "container", "c", flags.Container, "Print the logs of this container")
	cmd.Flags().BoolVarP(&flags.Follow, "follow", "f", flags.Follow, "Specify if the logs should be streamed.")
	cmd.Flags().BoolVar(&flags.IgnoreLogErrors, "ignore-errors", flags.IgnoreLogErrors, "If watching / following pod logs, allow for any errors that occur to be non-fatal")
	cmd.Flags().BoolVar(&flags.InsecureSkipTLSVerifyBackend, "insecure-skip-tls-verify-backend", flags.InsecureSkipTLSVerifyBackend,
		"Skip verifying the identity of the kubelet that logs are requested from.  In theory, an attacker could provide invalid log content back. You might want to use this if your kubelet serving certificates have expired.")
	cmd.Flags().Int64Var(&flags.LimitBytes, "limit-bytes", flags.LimitBytes, "Maximum bytes of logs to return. Defaults to no limit.")
	cmd.Flags().IntVar(&flags.MaxFollowConcurrency, "max-log-requests", flags.MaxFollowConcurrency, "Specify maximum number of concurrent logs to follow when using by a selector. Defaults to 5.")
	cmd.Flags().BoolVar(&flags.Prefix, "prefix", flags.Prefix, "Prefix each log line with the log source (pod name and container name)")
	cmd.Flags().BoolVarP(&flags.Previous, "previous", "p", flags.Previous, "If true, print the logs for the previous instance of the container in a pod if it exists.")
	cmd.Flags().StringVarP(&flags.Selector, "selector", "l", flags.Selector, "Selector (label query) to filter on.")
	cmd.Flags().DurationVar(&flags.Since, "since", flags.Since, "Only return logs newer than a relative duration like 5s, 2m, or 3h. Defaults to all logs. Only one of since-time / since may be used.")
	cmd.Flags().StringVar(&flags.SinceTime, "since-time", flags.SinceTime, i18n.T("Only return logs after a specific date (RFC3339). Defaults to all logs. Only one of since-time / since may be used."))
	cmd.Flags().Int64Var(&flags.Tail, "tail", flags.Tail, "Lines of recent log file to display. Defaults to -1 with no selector, showing all log lines otherwise 10, if a selector is provided.")
	cmd.Flags().BoolVar(&flags.Timestamps, "timestamps", flags.Timestamps, "Include timestamps on each line in the log output")
}

type LogsOptions struct {
	Namespace   string
	ResourceArg string
	Options     runtime.Object
	Resources   []string

	ConsumeRequestFn func(rest.ResponseWrapper, io.Writer) error

	// PodLogOptions
	AllContainers                bool
	SinceTime                    string
	Since                        time.Duration
	Follow                       bool
	Previous                     bool
	Timestamps                   bool
	IgnoreLogErrors              bool
	LimitBytes                   int64
	Tail                         int64
	Container                    string
	InsecureSkipTLSVerifyBackend bool
	Selector                     string
	MaxFollowConcurrency         int
	Prefix                       bool

	//internal items
	ContainerNameSpecified bool
	TailSpecified          bool

	// initialized during ToOptions operation
	Object           runtime.Object
	GetPodTimeout    time.Duration
	RESTClientGetter genericclioptions.RESTClientGetter
	LogsForObject    polymorphichelpers.LogsForObjectFunc
	ResourceBuilder  *resource.Builder

	genericclioptions.IOStreams

	containerNameFromRefSpecRegexp *regexp.Regexp
}

// NewCmdLogs creates a new pod logs command
func NewCmdLogs(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	flags := NewLogsFlags(f, streams)

	cmd := &cobra.Command{
		Use:                   logsUsageStr,
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Print the logs for a container in a pod"),
		Long:                  logsLong,
		Example:               logsExample,
		ValidArgsFunction:     util.PodResourceNameAndContainerCompletionFunc(f),
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(cmd, args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.RunLogs())
		},
	}

	flags.AddFlags(cmd)
	return cmd
}

func (o *LogsOptions) ToLogOptions() (*corev1.PodLogOptions, error) {
	logOptions := &corev1.PodLogOptions{
		Container:                    o.Container,
		Follow:                       o.Follow,
		Previous:                     o.Previous,
		Timestamps:                   o.Timestamps,
		InsecureSkipTLSVerifyBackend: o.InsecureSkipTLSVerifyBackend,
	}

	if len(o.SinceTime) > 0 {
		t, err := util.ParseRFC3339(o.SinceTime, metav1.Now)
		if err != nil {
			return nil, err
		}

		logOptions.SinceTime = &t
	}

	if o.LimitBytes != 0 {
		logOptions.LimitBytes = &o.LimitBytes
	}

	if o.Since != 0 {
		// round up to the nearest second
		sec := int64(o.Since.Round(time.Second).Seconds())
		logOptions.SinceSeconds = &sec
	}

	if len(o.Selector) > 0 && o.Tail == -1 && !o.TailSpecified {
		logOptions.TailLines = &selectorTail
	} else if o.Tail != -1 {
		logOptions.TailLines = &o.Tail
	}

	return logOptions, nil
}

func (o LogsOptions) Validate() error {
	if len(o.SinceTime) > 0 && o.Since != 0 {
		return fmt.Errorf("at most one of `sinceTime` or `since` may be specified")
	}

	logsOptions, ok := o.Options.(*corev1.PodLogOptions)
	if !ok {
		return errors.New("unexpected logs options object")
	}
	if o.AllContainers && len(logsOptions.Container) > 0 {
		return fmt.Errorf("--all-containers=true should not be specified with container name %s", logsOptions.Container)
	}

	if o.ContainerNameSpecified && len(o.Resources) == 2 {
		return fmt.Errorf("only one of -c or an inline [CONTAINER] arg is allowed")
	}

	if o.LimitBytes < 0 {
		return fmt.Errorf("--limit-bytes must be greater than 0")
	}

	if logsOptions.SinceSeconds != nil && *logsOptions.SinceSeconds < int64(0) {
		return fmt.Errorf("--since must be greater than 0")
	}

	if logsOptions.TailLines != nil && *logsOptions.TailLines < -1 {
		return fmt.Errorf("--tail must be greater than or equal to -1")
	}

	return nil
}

// RunLogs retrieves a pod log
func (o LogsOptions) RunLogs() error {
	requests, err := o.LogsForObject(o.RESTClientGetter, o.Object, o.Options, o.GetPodTimeout, o.AllContainers)
	if err != nil {
		return err
	}

	if o.Follow && len(requests) > 1 {
		if len(requests) > o.MaxFollowConcurrency {
			return fmt.Errorf(
				"you are attempting to follow %d log streams, but maximum allowed concurrency is %d, use --max-log-requests to increase the limit",
				len(requests), o.MaxFollowConcurrency,
			)
		}

		return o.parallelConsumeRequest(requests)
	}

	return o.sequentialConsumeRequest(requests)
}

func (o LogsOptions) parallelConsumeRequest(requests map[corev1.ObjectReference]rest.ResponseWrapper) error {
	reader, writer := io.Pipe()
	wg := &sync.WaitGroup{}
	wg.Add(len(requests))
	for objRef, request := range requests {
		go func(objRef corev1.ObjectReference, request rest.ResponseWrapper) {
			defer wg.Done()
			out := o.addPrefixIfNeeded(objRef, writer)
			if err := o.ConsumeRequestFn(request, out); err != nil {
				if !o.IgnoreLogErrors {
					writer.CloseWithError(err)

					// It's important to return here to propagate the error via the pipe
					return
				}

				fmt.Fprintf(writer, "error: %v\n", err)
			}

		}(objRef, request)
	}

	go func() {
		wg.Wait()
		writer.Close()
	}()

	_, err := io.Copy(o.Out, reader)
	return err
}

func (o LogsOptions) sequentialConsumeRequest(requests map[corev1.ObjectReference]rest.ResponseWrapper) error {
	for objRef, request := range requests {
		out := o.addPrefixIfNeeded(objRef, o.Out)
		if err := o.ConsumeRequestFn(request, out); err != nil {
			if !o.IgnoreLogErrors {
				return err
			}

			fmt.Fprintf(o.Out, "error: %v\n", err)
		}
	}

	return nil
}

func (o LogsOptions) addPrefixIfNeeded(ref corev1.ObjectReference, writer io.Writer) io.Writer {
	if !o.Prefix || ref.FieldPath == "" || ref.Name == "" {
		return writer
	}

	// We rely on ref.FieldPath to contain a reference to a container
	// including a container name (not an index) so we can get a container name
	// without making an extra API request.
	var containerName string
	containerNameMatches := o.containerNameFromRefSpecRegexp.FindStringSubmatch(ref.FieldPath)
	if len(containerNameMatches) == 2 {
		containerName = containerNameMatches[1]
	}

	prefix := fmt.Sprintf("[pod/%s/%s] ", ref.Name, containerName)
	return &prefixingWriter{
		prefix: []byte(prefix),
		writer: writer,
	}
}

// DefaultConsumeRequest reads the data from request and writes into
// the out writer. It buffers data from requests until the newline or io.EOF
// occurs in the data, so it doesn't interleave logs sub-line
// when running concurrently.
//
// A successful read returns err == nil, not err == io.EOF.
// Because the function is defined to read from request until io.EOF, it does
// not treat an io.EOF as an error to be reported.
func DefaultConsumeRequest(request rest.ResponseWrapper, out io.Writer) error {
	readCloser, err := request.Stream(context.TODO())
	if err != nil {
		return err
	}
	defer readCloser.Close()

	r := bufio.NewReader(readCloser)
	for {
		bytes, err := r.ReadBytes('\n')
		if _, err := out.Write(bytes); err != nil {
			return err
		}

		if err != nil {
			if err != io.EOF {
				return err
			}
			return nil
		}
	}
}

type prefixingWriter struct {
	prefix []byte
	writer io.Writer
}

func (pw *prefixingWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}

	// Perform an "atomic" write of a prefix and p to make sure that it doesn't interleave
	// sub-line when used concurrently with io.PipeWrite.
	n, err := pw.writer.Write(append(pw.prefix, p...))
	if n > len(p) {
		// To comply with the io.Writer interface requirements we must
		// return a number of bytes written from p (0 <= n <= len(p)),
		// so we are ignoring the length of the prefix here.
		return len(p), err
	}
	return n, err
}
