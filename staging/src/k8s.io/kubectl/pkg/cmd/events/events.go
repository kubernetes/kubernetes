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
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	runtimeresource "k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/interrupt"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	eventsLong = templates.LongDesc(i18n.T(`
		Display events.

		Prints a table of the most important information about events.
		You can request events for a namespace, for all namespace, or
		filtered to only those pertaining to a specified resource.`))

	eventsExample = templates.Examples(i18n.T(`
	# List recent events in the default namespace
	kubectl events

	# List recent events in all namespaces
	kubectl events --all-namespaces

	# List recent events for the specified pod, then wait for more events and list them as they arrive
	kubectl events --for pod/web-pod-13je7 --watch

	# List recent events in YAML format
	kubectl events -oyaml

	# List recent only events of type 'Warning' or 'Normal'
	kubectl events --types=Warning,Normal`))
)

// EventsFlags directly reflect the information that CLI is gathering via flags.  They will be converted to Options, which
// reflect the runtime requirements for the command.  This structure reduces the transformation to wiring and makes
// the logic itself easy to unit test.
type EventsFlags struct {
	RESTClientGetter genericclioptions.RESTClientGetter
	PrintFlags       *genericclioptions.PrintFlags

	AllNamespaces bool
	Watch         bool
	NoHeaders     bool
	ForObject     string
	FilterTypes   []string
	ChunkSize     int64
	genericiooptions.IOStreams
}

// NewEventsFlags returns a default EventsFlags
func NewEventsFlags(restClientGetter genericclioptions.RESTClientGetter, streams genericiooptions.IOStreams) *EventsFlags {
	return &EventsFlags{
		RESTClientGetter: restClientGetter,
		PrintFlags:       genericclioptions.NewPrintFlags("events").WithTypeSetter(scheme.Scheme),
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
	FilterTypes   []string

	forGVK  schema.GroupVersionKind
	forName string

	client *kubernetes.Clientset

	PrintObj printers.ResourcePrinterFunc

	genericiooptions.IOStreams
}

// NewCmdEvents creates a new events command
func NewCmdEvents(restClientGetter genericclioptions.RESTClientGetter, streams genericiooptions.IOStreams) *cobra.Command {
	flags := NewEventsFlags(restClientGetter, streams)

	cmd := &cobra.Command{
		Use:                   fmt.Sprintf("events [(-o|--output=)%s] [--for TYPE/NAME] [--watch] [--types=Normal,Warning]", strings.Join(flags.PrintFlags.AllowedFormats(), "|")),
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("List events"),
		Long:                  eventsLong,
		Example:               eventsExample,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions()
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}
	flags.AddFlags(cmd)
	flags.PrintFlags.AddFlags(cmd)
	return cmd
}

// AddFlags registers flags for a cli.
func (flags *EventsFlags) AddFlags(cmd *cobra.Command) {
	cmd.Flags().BoolVarP(&flags.Watch, "watch", "w", flags.Watch, "After listing the requested events, watch for more events.")
	cmd.Flags().BoolVarP(&flags.AllNamespaces, "all-namespaces", "A", flags.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().StringVar(&flags.ForObject, "for", flags.ForObject, "Filter events to only those pertaining to the specified resource.")
	cmd.Flags().StringSliceVar(&flags.FilterTypes, "types", flags.FilterTypes, "Output only events of given types.")
	cmd.Flags().BoolVar(&flags.NoHeaders, "no-headers", flags.NoHeaders, "When using the default output format, don't print headers.")
	cmdutil.AddChunkSizeFlag(cmd, &flags.ChunkSize)
}

// ToOptions converts from CLI inputs to runtime inputs.
func (flags *EventsFlags) ToOptions() (*EventsOptions, error) {
	o := &EventsOptions{
		AllNamespaces: flags.AllNamespaces,
		Watch:         flags.Watch,
		FilterTypes:   flags.FilterTypes,
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

	if len(o.FilterTypes) > 0 {
		o.FilterTypes = sets.List(sets.New[string](o.FilterTypes...))
	}

	var printer printers.ResourcePrinter
	if flags.PrintFlags.OutputFormat != nil && len(*flags.PrintFlags.OutputFormat) > 0 {
		printer, err = flags.PrintFlags.ToPrinter()
		if err != nil {
			return nil, err
		}
	} else {
		printer = NewEventPrinter(flags.NoHeaders, flags.AllNamespaces)
	}

	o.PrintObj = func(object runtime.Object, writer io.Writer) error {
		return printer.PrintObj(object, writer)
	}

	return o, nil
}

func (o *EventsOptions) Validate() error {
	for _, val := range o.FilterTypes {
		if !strings.EqualFold(val, "Normal") && !strings.EqualFold(val, "Warning") {
			return fmt.Errorf("valid --types are Normal or Warning")
		}
	}

	return nil
}

// Run retrieves events
func (o *EventsOptions) Run() error {
	ctx := context.TODO()
	namespace := o.Namespace
	if o.AllNamespaces {
		namespace = ""
	}
	listOptions := metav1.ListOptions{Limit: cmdutil.DefaultChunkSize}
	if o.forName != "" {
		listOptions.FieldSelector = fields.AndSelectors(
			fields.OneTermEqualSelector("involvedObject.kind", o.forGVK.Kind),
			fields.OneTermEqualSelector("involvedObject.apiVersion", o.forGVK.GroupVersion().String()),
			fields.OneTermEqualSelector("involvedObject.name", o.forName)).String()
	}
	if o.Watch {
		return o.runWatch(ctx, namespace, listOptions)
	}

	e := o.client.CoreV1().Events(namespace)
	el := &corev1.EventList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "EventList",
			APIVersion: "v1",
		},
	}
	err := runtimeresource.FollowContinue(&listOptions,
		func(options metav1.ListOptions) (runtime.Object, error) {
			newEvents, err := e.List(ctx, options)
			if err != nil {
				return nil, runtimeresource.EnhanceListError(err, options, "events")
			}
			el.Items = append(el.Items, newEvents.Items...)
			return newEvents, nil
		})

	if err != nil {
		return err
	}

	var filteredEvents []corev1.Event
	for _, e := range el.Items {
		if !o.filteredEventType(e.Type) {
			continue
		}
		if e.GetObjectKind().GroupVersionKind().Empty() {
			e.SetGroupVersionKind(schema.GroupVersionKind{
				Version: "v1",
				Kind:    "Event",
			})
		}
		filteredEvents = append(filteredEvents, e)
	}

	el.Items = filteredEvents

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

	o.PrintObj(el, w)
	w.Flush()
	return nil
}

func (o *EventsOptions) runWatch(ctx context.Context, namespace string, listOptions metav1.ListOptions) error {
	client := o.client.CoreV1().Events(namespace)
	w := printers.GetNewTabWriter(o.Out)
	printEvent := func(event *corev1.Event) error {
		if !o.filteredEventType(event.Type) {
			return nil
		}
		if event.GetObjectKind().GroupVersionKind().Empty() {
			event.SetGroupVersionKind(corev1.SchemeGroupVersion.WithKind("Event"))
		}
		if err := o.PrintObj(event, w); err != nil {
			return err
		}
		return w.Flush()
	}

	handleEvent := func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Error:
			return false, apierrors.FromObject(event.Object)
		case watch.Added, watch.Modified:
			if ev, ok := event.Object.(*corev1.Event); ok {
				return false, printEvent(ev)
			}
		}
		return false, nil
	}

	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	return interrupt.New(nil, func() { cancel(nil) }).Run(func() error {
		// Start observing changes from the list snapshot so none are missed between listing and watching.
		var resourceVersion string
		err := runtimeresource.FollowContinue(&listOptions, func(options metav1.ListOptions) (runtime.Object, error) {
			eventList, err := client.List(ctx, options)
			if err != nil {
				return nil, runtimeresource.EnhanceListError(err, options, "events")
			}
			resourceVersion = eventList.ResourceVersion
			for i := range eventList.Items {
				if err := printEvent(&eventList.Items[i]); err != nil {
					return nil, err
				}
			}
			return eventList, nil
		})
		if apierrors.IsForbidden(err) && resourceVersion == "" {
			// Watch-only users can get a checkpoint from a bookmark after the initial events.
			listOptions.AllowWatchBookmarks = true
			initialWatch, err := client.Watch(ctx, listOptions)
			if err != nil {
				return err
			}
			_, err = watchtools.UntilWithoutRetry(ctx, initialWatch, func(event watch.Event) (bool, error) {
				if event.Type != watch.Bookmark {
					return handleEvent(event)
				}
				object, err := meta.Accessor(event.Object)
				if err != nil {
					return false, err
				}
				resourceVersion = object.GetResourceVersion()
				return resourceVersion != "" && resourceVersion != "0", nil
			})
			if errors.Is(err, watchtools.ErrWatchClosed) {
				return errors.New("event watch closed before receiving a resource version bookmark")
			}
			if err != nil {
				return err
			}
		} else if err != nil {
			return err
		}

		eventWatch, err := watchtools.NewRetryWatcherWithContext(ctx, resourceVersion, &cache.ListWatch{
			WatchFuncWithContext: func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
				options.FieldSelector = listOptions.FieldSelector
				watcher, err := client.Watch(ctx, options)
				var status apierrors.APIStatus
				var negotiationError runtime.NegotiateError
				switch {
				case errors.As(err, &negotiationError):
					cancel(err)
				case errors.As(err, &status):
					code := status.Status().Code
					// RetryWatcher can keep retrying permanent failures when opening the watch.
					if code >= http.StatusBadRequest && code < http.StatusInternalServerError &&
						code != http.StatusRequestTimeout && code != http.StatusTooManyRequests {
						cancel(err)
					}
				}
				return watcher, err
			},
		})
		if err != nil {
			return err
		}

		_, err = watchtools.UntilWithoutRetry(ctx, eventWatch, handleEvent)
		if cause := context.Cause(ctx); cause != nil {
			return cause
		}
		return err
	})
}

// filteredEventType checks given event can be printed
// by comparing it in filtered event flag.
// If --event flag is not set by user, this function allows
// all events to be printed.
func (o *EventsOptions) filteredEventType(et string) bool {
	if len(o.FilterTypes) == 0 {
		return true
	}

	for _, t := range o.FilterTypes {
		if strings.EqualFold(t, et) {
			return true
		}
	}

	return false
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

	fullySpecifiedGVR, groupResource := schema.ParseResourceArg(strings.ToLower(resource))
	gvr := schema.GroupVersionResource{}
	if fullySpecifiedGVR != nil {
		gvr, _ = mapper.ResourceFor(*fullySpecifiedGVR)
	}
	if gvr.Empty() {
		gvr, err = mapper.ResourceFor(groupResource.WithVersion(""))
		if err != nil {
			return
		}
	}

	gvk, err = mapper.KindFor(gvr)
	if err != nil {
		return
	}
	found = true

	return
}
