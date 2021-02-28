/*
Copyright 2021 The Kubernetes Authors.

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

package events

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/duration"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/kubernetes"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

const (
	eventsUsageStr = "events TODO"
)

var (
	logsLong = templates.LongDesc(i18n.T(`
		Get events TODO.`))

	logsExample = templates.Examples(i18n.T(`
		# Return recent events
		kubectl events

		TODO`))

	selectorTail       int64 = 10
	eventssUsageErrStr       = fmt.Sprintf("TODO usage error string: %s", eventsUsageStr)
)

type EventsOptions struct {
	AllNamespaces bool
	Namespace     string
	Watch         bool

	ctx    context.Context
	client *kubernetes.Clientset

	genericclioptions.IOStreams
}

func NewEventsOptions(streams genericclioptions.IOStreams, watch bool) *EventsOptions {
	return &EventsOptions{
		IOStreams: streams,
		Watch:     watch,
	}
}

// NewCmdEvents creates a new pod logs command
func NewCmdEvents(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewEventsOptions(streams, false)

	cmd := &cobra.Command{
		Use:                   eventsUsageStr,
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Print the logs for a container in a pod"),
		Long:                  logsLong,
		Example:               logsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}
	o.AddFlags(cmd)
	return cmd
}

func (o *EventsOptions) AddFlags(cmd *cobra.Command) {
	cmd.Flags().BoolVarP(&o.Watch, "watch", "w", o.Watch, "After listing the requested events, watch for more events.")
	cmd.Flags().BoolVarP(&o.AllNamespaces, "all-namespaces", "A", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
}

func (o *EventsOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.ctx = cmd.Context()
	o.client, err = f.KubernetesClientSet()

	return nil
}

func (o EventsOptions) Validate() error {
	// Nothing to do here.
	return nil
}

// Run retrieves events
func (o EventsOptions) Run() error {
	namespace := o.Namespace
	if o.AllNamespaces {
		namespace = ""
	}
	el, err := o.client.CoreV1().Events(namespace).List(o.ctx, metav1.ListOptions{})

	if err != nil {
		return err
	}

	w := printers.GetNewTabWriter(o.Out)
	defer w.Flush()

	sort.Sort(SortableEvents(el.Items))

	if o.AllNamespaces {
		fmt.Fprintf(w, "NAMESPACE\t")
	}
	fmt.Fprintf(w, "LAST SEEN\tTYPE\tREASON\tOBJECT\tMESSAGE\n")

	for _, e := range el.Items {
		var interval string
		if e.Count > 1 {
			interval = fmt.Sprintf("%s (x%d over %s)", translateTimestampSince(e.LastTimestamp.Time), e.Count, translateTimestampSince(e.FirstTimestamp.Time))
		} else {
			interval = translateTimestampSince(eventTime(e))
		}
		source := e.Source.Component
		if source == "" {
			source = e.ReportingController
		}
		if o.AllNamespaces {
			fmt.Fprintf(w, "%v\t", e.Namespace)
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s/%s\t%v\n",
			interval,
			e.Type,
			e.Reason,
			e.InvolvedObject.Kind, e.InvolvedObject.Name,
			strings.TrimSpace(e.Message),
		)
	}

	return nil
}

// SortableEvents implements sort.Interface for []api.Event by time
type SortableEvents []corev1.Event

func (list SortableEvents) Len() int {
	return len(list)
}

func (list SortableEvents) Swap(i, j int) {
	list[i], list[j] = list[j], list[i]
}

func (list SortableEvents) Less(i, j int) bool {
	return eventTime(list[i]).Before(eventTime(list[j]))
}

// Some events have just an EventTime; if LastTimestamp is present we prefer that.
func eventTime(event corev1.Event) time.Time {
	if !event.LastTimestamp.Time.IsZero() {
		return event.LastTimestamp.Time
	}
	return event.EventTime.Time
}

// translateTimestampSince returns the elapsed time since timestamp in
// human-readable approximation.
func translateTimestampSince(timestamp time.Time) string {
	if timestamp.IsZero() {
		return "<unknown>"
	}

	return duration.HumanDuration(time.Since(timestamp))
}
