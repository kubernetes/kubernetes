/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"io"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/duration"
	"k8s.io/cli-runtime/pkg/printers"
)

// EventPrinter stores required fields to be used for
// default printing for events command.
type EventPrinter struct {
	NoHeaders     bool
	AllNamespaces bool

	headersPrinted bool
}

// PrintObj prints different type of event objects.
func (ep *EventPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	if !ep.NoHeaders && !ep.headersPrinted {
		ep.printHeadings(out)
		ep.headersPrinted = true
	}

	switch t := obj.(type) {
	case *corev1.EventList:
		for _, e := range t.Items {
			ep.printOneEvent(out, e)
		}
	case *corev1.Event:
		ep.printOneEvent(out, *t)
	default:
		return fmt.Errorf("unknown event type %t", t)
	}

	return nil
}

func (ep *EventPrinter) printHeadings(w io.Writer) {
	if ep.AllNamespaces {
		fmt.Fprintf(w, "NAMESPACE\t")
	}
	fmt.Fprintf(w, "LAST SEEN\tTYPE\tREASON\tOBJECT\tMESSAGE\n")
}

func (ep *EventPrinter) printOneEvent(w io.Writer, e corev1.Event) {
	interval := getInterval(e)
	if ep.AllNamespaces {
		fmt.Fprintf(w, "%v\t", e.Namespace)
	}
	fmt.Fprintf(w, "%s\t%s\t%s\t%s/%s\t%v\n",
		interval,
		printers.EscapeTerminal(e.Type),
		printers.EscapeTerminal(e.Reason),
		printers.EscapeTerminal(e.InvolvedObject.Kind),
		printers.EscapeTerminal(e.InvolvedObject.Name),
		printers.EscapeTerminal(strings.TrimSpace(e.Message)),
	)
}

func getInterval(e corev1.Event) string {
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

	return interval
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

func NewEventPrinter(noHeader, allNamespaces bool) *EventPrinter {
	return &EventPrinter{
		NoHeaders:     noHeader,
		AllNamespaces: allNamespaces,
	}
}
