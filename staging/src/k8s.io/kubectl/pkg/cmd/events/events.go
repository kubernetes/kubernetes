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
	"io"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/duration"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	runtimeresource "k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes"
	watchtools "k8s.io/client-go/tools/watch"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/interrupt"
	"k8s.io/kubectl/pkg/util/templates"
)

const (
	eventsUsageStr = "events [--for TYPE/NAME] [--watch]"
)

var (
	eventsLong = templates.LongDesc(i18n.T(`
		Experimental: Display events

		Prints a table of the most important information about events.
		You can request events for a namespace, for all namespace, or
		filtered to only those pertaining to a specified resource.`))

	eventsExample = templates.Examples(i18n.T(`
	# List recent events in the default namespace.
	kubectl alpha events

	# List recent events in all namespaces.
	kubectl alpha events --all-namespaces

	# List recent events for the specified pod, then wait for more events and list them as they arrive.
	kubectl alpha events --for pod/web-pod-13je7 --watch`))
)

// EventsFlags directly reflect the information that CLI is gathering via flags.  They will be converted to Options, which
// reflect the runtime requirements for the command.  This structure reduces the transformation to wiring and makes
// the logic itself easy to unit test.
type EventsFlags struct {
	RESTClientGetter genericclioptions.RESTClientGetter

	AllNamespaces bool
	Watch         bool
	ForObject     string
	ChunkSize     int64
	genericclioptions.IOStreams
}

// NewEventsFlags returns a default EventsFlags
func NewEventsFlags(restClientGetter genericclioptions.RESTClientGetter, streams genericclioptions.IOStreams) *EventsFlags {
	return &EventsFlags{
		RESTClientGetter: restClientGetter,
		IOStreams:        streams,
		ChunkSize:        cmdutil.DefaultChunkSize,
	}
}

// EventsOptions is a set of options that allows you to list events.  This is the object reflects the
// runtime needs of an events command, making the logic itself easy to unit test.
type EventsOptions struct {
	Namespace     string
	AllNamespaces bool
	Watch         bool

	forGVK  schema.GroupVersionKind
	forName string

	ctx    context.Context
	client *kubernetes.Clientset

	genericclioptions.IOStreams
}

// NewCmdEvents creates a new events command
func NewCmdEvents(restClientGetter genericclioptions.RESTClientGetter, streams genericclioptions.IOStreams) *cobra.Command {
	flags := NewEventsFlags(restClientGetter, streams)

	cmd := &cobra.Command{
		Use:                   eventsUsageStr,
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Experimental: List events"),
		Long:                  eventsLong,
		Example:               eventsExample,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(cmd.Context(), args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Run())
		},
	}
	flags.AddFlags(cmd)
	return cmd
}

// AddFlags registers flags for a cli.
func (o *EventsFlags) AddFlags(cmd *cobra.Command) {
	cmd.Flags().BoolVarP(&o.Watch, "watch", "w", o.Watch, "After listing the requested events, watch for more events.")
	cmd.Flags().BoolVarP(&o.AllNamespaces, "all-namespaces", "A", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().StringVar(&o.ForObject, "for", o.ForObject, "Filter events to only those pertaining to the specified resource.")
	cmdutil.AddChunkSizeFlag(cmd, &o.ChunkSize)
}

// ToOptions converts from CLI inputs to runtime inputs.
func (flags *EventsFlags) ToOptions(ctx context.Context, args []string) (*EventsOptions, error) {
	o := &EventsOptions{
		ctx:           ctx,
		AllNamespaces: flags.AllNamespaces,
		Watch:         flags.Watch,
		IOStreams:     flags.IOStreams,
	}
	var err error
	o.Namespace, _, err = flags.RESTClientGetter.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return nil, err
	}

	if flags.ForObject != "" {
		mapper, err := flags.RESTClientGetter.ToRESTMapper()
		if err != nil {
			return nil, err
		}
		var found bool
		o.forGVK, o.forName, found, err = decodeResourceTypeName(mapper, flags.ForObject)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, fmt.Errorf("--for must be in resource/name form")
		}
	}

	clientConfig, err := flags.RESTClientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	o.client, err = kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	return o, nil
}

// Run retrieves events
func (o EventsOptions) Run() error {
	namespace := o.Namespace
	if o.AllNamespaces {
		namespace = ""
	}
	listOptions := metav1.ListOptions{Limit: cmdutil.DefaultChunkSize}
	if o.forName != "" {
		listOptions.FieldSelector = fields.AndSelectors(
			fields.OneTermEqualSelector("involvedObject.kind", o.forGVK.Kind),
			fields.OneTermEqualSelector("involvedObject.name", o.forName)).String()
	}
	if o.Watch {
		return o.runWatch(namespace, listOptions)
	}

	e := o.client.CoreV1().Events(namespace)
	el := &corev1.EventList{}
	err := runtimeresource.FollowContinue(&listOptions,
		func(options metav1.ListOptions) (runtime.Object, error) {
			newEvents, err := e.List(o.ctx, options)
			if err != nil {
				return nil, runtimeresource.EnhanceListError(err, options, "events")
			}
			el.Items = append(el.Items, newEvents.Items...)
			return newEvents, nil
		})

	if err != nil {
		return err
	}

	if len(el.Items) == 0 {
		if o.AllNamespaces {
			fmt.Fprintln(o.ErrOut, "No events found.")
		} else {
			fmt.Fprintf(o.ErrOut, "No events found in %s namespace.\n", o.Namespace)
		}
		return nil
	}

	w := printers.GetNewTabWriter(o.Out)

	sort.Sort(SortableEvents(el.Items))

	printHeadings(w, o.AllNamespaces)
	for _, e := range el.Items {
		printOneEvent(w, e, o.AllNamespaces)
	}
	w.Flush()
	return nil
}

func (o EventsOptions) runWatch(namespace string, listOptions metav1.ListOptions) error {
	eventWatch, err := o.client.CoreV1().Events(namespace).Watch(o.ctx, listOptions)
	if err != nil {
		return err
	}
	w := printers.GetNewTabWriter(o.Out)
	headingsPrinted := false

	ctx, cancel := context.WithCancel(o.ctx)
	defer cancel()
	intr := interrupt.New(nil, cancel)
	intr.Run(func() error {
		_, err := watchtools.UntilWithoutRetry(ctx, eventWatch, func(e watch.Event) (bool, error) {
			if e.Type == watch.Deleted { // events are deleted after 1 hour; don't print that
				return false, nil
			}
			event := e.Object.(*corev1.Event)
			if !headingsPrinted {
				printHeadings(w, o.AllNamespaces)
				headingsPrinted = true
			}
			printOneEvent(w, *event, o.AllNamespaces)
			w.Flush()
			return false, nil
		})
		return err
	})

	return nil
}

func printHeadings(w io.Writer, allNamespaces bool) {
	if allNamespaces {
		fmt.Fprintf(w, "NAMESPACE\t")
	}
	fmt.Fprintf(w, "LAST SEEN\tTYPE\tREASON\tOBJECT\tMESSAGE\n")
}

func printOneEvent(w io.Writer, e corev1.Event, allNamespaces bool) {
	var interval string
	firstTimestampSince := translateMicroTimestampSince(e.EventTime)
	if e.EventTime.IsZero() {
		firstTimestampSince = translateTimestampSince(e.FirstTimestamp)
	}
	if e.Series != nil {
		interval = fmt.Sprintf("%s (x%d over %s)", translateMicroTimestampSince(e.Series.LastObservedTime), e.Series.Count, firstTimestampSince)
	} else if e.Count > 1 {
		interval = fmt.Sprintf("%s (x%d over %s)", translateTimestampSince(e.LastTimestamp), e.Count, firstTimestampSince)
	} else {
		interval = firstTimestampSince
	}
	if allNamespaces {
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

// Return the time that should be used for sorting, which can come from
// various places in corev1.Event.
func eventTime(event corev1.Event) time.Time {
	if event.Series != nil {
		return event.Series.LastObservedTime.Time
	}
	if !event.LastTimestamp.Time.IsZero() {
		return event.LastTimestamp.Time
	}
	return event.EventTime.Time
}

// translateMicroTimestampSince returns the elapsed time since timestamp in
// human-readable approximation.
func translateMicroTimestampSince(timestamp metav1.MicroTime) string {
	if timestamp.IsZero() {
		return "<unknown>"
	}

	return duration.HumanDuration(time.Since(timestamp.Time))
}

// translateTimestampSince returns the elapsed time since timestamp in
// human-readable approximation.
func translateTimestampSince(timestamp metav1.Time) string {
	if timestamp.IsZero() {
		return "<unknown>"
	}

	return duration.HumanDuration(time.Since(timestamp.Time))
}

// Inspired by k8s.io/cli-runtime/pkg/resource splitResourceTypeName()

// decodeResourceTypeName handles type/name resource formats and returns a resource tuple
// (empty or not), whether it successfully found one, and an error
func decodeResourceTypeName(mapper meta.RESTMapper, s string) (gvk schema.GroupVersionKind, name string, found bool, err error) {
	if !strings.Contains(s, "/") {
		return
	}
	seg := strings.Split(s, "/")
	if len(seg) != 2 {
		err = fmt.Errorf("arguments in resource/name form may not have more than one slash")
		return
	}
	resource, name := seg[0], seg[1]

	var gvr schema.GroupVersionResource
	gvr, err = mapper.ResourceFor(schema.GroupVersionResource{Resource: resource})
	if err != nil {
		return
	}
	gvk, err = mapper.KindFor(gvr)
	if err != nil {
		return
	}
	found = true

	return
}
