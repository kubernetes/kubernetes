// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package zpages

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"go.opencensus.io/internal"
	"go.opencensus.io/trace"
)

const (
	// spanNameQueryField is the header for span name.
	spanNameQueryField = "zspanname"
	// spanTypeQueryField is the header for type (running = 0, latency = 1, error = 2) to display.
	spanTypeQueryField = "ztype"
	// spanSubtypeQueryField is the header for sub-type:
	// * for latency based samples [0, 8] representing the latency buckets, where 0 is the first one;
	// * for error based samples, 0 means all, otherwise the error code;
	spanSubtypeQueryField = "zsubtype"
	// maxTraceMessageLength is the maximum length of a message in tracez output.
	maxTraceMessageLength = 1024
)

var (
	defaultLatencies = [...]time.Duration{
		10 * time.Microsecond,
		100 * time.Microsecond,
		time.Millisecond,
		10 * time.Millisecond,
		100 * time.Millisecond,
		time.Second,
		10 * time.Second,
		100 * time.Second,
	}
	canonicalCodes = [...]string{
		"OK",
		"CANCELLED",
		"UNKNOWN",
		"INVALID_ARGUMENT",
		"DEADLINE_EXCEEDED",
		"NOT_FOUND",
		"ALREADY_EXISTS",
		"PERMISSION_DENIED",
		"RESOURCE_EXHAUSTED",
		"FAILED_PRECONDITION",
		"ABORTED",
		"OUT_OF_RANGE",
		"UNIMPLEMENTED",
		"INTERNAL",
		"UNAVAILABLE",
		"DATA_LOSS",
		"UNAUTHENTICATED",
	}
)

func canonicalCodeString(code int32) string {
	if code < 0 || int(code) >= len(canonicalCodes) {
		return "error code " + strconv.FormatInt(int64(code), 10)
	}
	return canonicalCodes[code]
}

func tracezHandler(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	name := r.Form.Get(spanNameQueryField)
	t, _ := strconv.Atoi(r.Form.Get(spanTypeQueryField))
	st, _ := strconv.Atoi(r.Form.Get(spanSubtypeQueryField))
	WriteHTMLTracezPage(w, name, t, st)
}

// WriteHTMLTracezPage writes an HTML document to w containing locally-sampled trace spans.
func WriteHTMLTracezPage(w io.Writer, spanName string, spanType, spanSubtype int) {
	if err := headerTemplate.Execute(w, headerData{Title: "Trace Spans"}); err != nil {
		log.Printf("zpages: executing template: %v", err)
	}
	WriteHTMLTracezSummary(w)
	WriteHTMLTracezSpans(w, spanName, spanType, spanSubtype)
	if err := footerTemplate.Execute(w, nil); err != nil {
		log.Printf("zpages: executing template: %v", err)
	}
}

// WriteHTMLTracezSummary writes HTML to w containing a summary of locally-sampled trace spans.
//
// It includes neither a header nor footer, so you can embed this data in other pages.
func WriteHTMLTracezSummary(w io.Writer) {
	if err := summaryTableTemplate.Execute(w, getSummaryPageData()); err != nil {
		log.Printf("zpages: executing template: %v", err)
	}
}

// WriteHTMLTracezSpans writes HTML to w containing locally-sampled trace spans.
//
// It includes neither a header nor footer, so you can embed this data in other pages.
func WriteHTMLTracezSpans(w io.Writer, spanName string, spanType, spanSubtype int) {
	if spanName == "" {
		return
	}
	if err := tracesTableTemplate.Execute(w, traceDataFromSpans(spanName, traceSpans(spanName, spanType, spanSubtype))); err != nil {
		log.Printf("zpages: executing template: %v", err)
	}
}

// WriteTextTracezSpans writes formatted text to w containing locally-sampled trace spans.
func WriteTextTracezSpans(w io.Writer, spanName string, spanType, spanSubtype int) {
	spans := traceSpans(spanName, spanType, spanSubtype)
	data := traceDataFromSpans(spanName, spans)
	writeTextTraces(w, data)
}

// WriteTextTracezSummary writes formatted text to w containing a summary of locally-sampled trace spans.
func WriteTextTracezSummary(w io.Writer) {
	w.Write([]byte("Locally sampled spans summary\n\n"))

	data := getSummaryPageData()
	if len(data.Rows) == 0 {
		return
	}

	tw := tabwriter.NewWriter(w, 8, 8, 1, ' ', 0)

	for i, s := range data.Header {
		if i != 0 {
			tw.Write([]byte("\t"))
		}
		tw.Write([]byte(s))
	}
	tw.Write([]byte("\n"))

	put := func(x int) {
		if x == 0 {
			tw.Write([]byte(".\t"))
			return
		}
		fmt.Fprintf(tw, "%d\t", x)
	}
	for _, r := range data.Rows {
		tw.Write([]byte(r.Name))
		tw.Write([]byte("\t"))
		put(r.Active)
		for _, l := range r.Latency {
			put(l)
		}
		put(r.Errors)
		tw.Write([]byte("\n"))
	}
	tw.Flush()
}

// traceData contains data for the trace data template.
type traceData struct {
	Name string
	Num  int
	Rows []traceRow
}

type traceRow struct {
	Fields [3]string
	trace.SpanContext
	ParentSpanID trace.SpanID
}

type events []interface{}

func (e events) Len() int { return len(e) }
func (e events) Less(i, j int) bool {
	var ti time.Time
	switch x := e[i].(type) {
	case *trace.Annotation:
		ti = x.Time
	case *trace.MessageEvent:
		ti = x.Time
	}
	switch x := e[j].(type) {
	case *trace.Annotation:
		return ti.Before(x.Time)
	case *trace.MessageEvent:
		return ti.Before(x.Time)
	}
	return false
}

func (e events) Swap(i, j int) { e[i], e[j] = e[j], e[i] }

func traceRows(s *trace.SpanData) []traceRow {
	start := s.StartTime

	lasty, lastm, lastd := start.Date()
	wholeTime := func(t time.Time) string {
		return t.Format("2006/01/02-15:04:05") + fmt.Sprintf(".%06d", t.Nanosecond()/1000)
	}
	formatTime := func(t time.Time) string {
		y, m, d := t.Date()
		if y == lasty && m == lastm && d == lastd {
			return t.Format("           15:04:05") + fmt.Sprintf(".%06d", t.Nanosecond()/1000)
		}
		lasty, lastm, lastd = y, m, d
		return wholeTime(t)
	}

	lastTime := start
	formatElapsed := func(t time.Time) string {
		d := t.Sub(lastTime)
		lastTime = t
		u := int64(d / 1000)
		// There are five cases for duration printing:
		// -1234567890s
		// -1234.123456
		//      .123456
		// 12345.123456
		// 12345678901s
		switch {
		case u < -9999999999:
			return fmt.Sprintf("%11ds", u/1e6)
		case u < 0:
			sec := u / 1e6
			u -= sec * 1e6
			return fmt.Sprintf("%5d.%06d", sec, -u)
		case u < 1e6:
			return fmt.Sprintf("     .%6d", u)
		case u <= 99999999999:
			sec := u / 1e6
			u -= sec * 1e6
			return fmt.Sprintf("%5d.%06d", sec, u)
		default:
			return fmt.Sprintf("%11ds", u/1e6)
		}
	}

	firstRow := traceRow{Fields: [3]string{wholeTime(start), "", ""}, SpanContext: s.SpanContext, ParentSpanID: s.ParentSpanID}
	if s.EndTime.IsZero() {
		firstRow.Fields[1] = "            "
	} else {
		firstRow.Fields[1] = formatElapsed(s.EndTime)
		lastTime = start
	}
	out := []traceRow{firstRow}

	formatAttributes := func(a map[string]interface{}) string {
		if len(a) == 0 {
			return ""
		}
		var keys []string
		for key := range a {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		var s []string
		for _, key := range keys {
			val := a[key]
			switch val.(type) {
			case string:
				s = append(s, fmt.Sprintf("%s=%q", key, val))
			default:
				s = append(s, fmt.Sprintf("%s=%v", key, val))
			}
		}
		return "Attributes:{" + strings.Join(s, ", ") + "}"
	}

	if s.Status != (trace.Status{}) {
		msg := fmt.Sprintf("Status{canonicalCode=%s, description=%q}",
			canonicalCodeString(s.Status.Code), s.Status.Message)
		out = append(out, traceRow{Fields: [3]string{"", "", msg}})
	}

	if len(s.Attributes) != 0 {
		out = append(out, traceRow{Fields: [3]string{"", "", formatAttributes(s.Attributes)}})
	}

	var es events
	for i := range s.Annotations {
		es = append(es, &s.Annotations[i])
	}
	for i := range s.MessageEvents {
		es = append(es, &s.MessageEvents[i])
	}
	sort.Sort(es)
	for _, e := range es {
		switch e := e.(type) {
		case *trace.Annotation:
			msg := e.Message
			if len(e.Attributes) != 0 {
				msg = msg + "  " + formatAttributes(e.Attributes)
			}
			row := traceRow{Fields: [3]string{
				formatTime(e.Time),
				formatElapsed(e.Time),
				msg,
			}}
			out = append(out, row)
		case *trace.MessageEvent:
			row := traceRow{Fields: [3]string{formatTime(e.Time), formatElapsed(e.Time)}}
			switch e.EventType {
			case trace.MessageEventTypeSent:
				row.Fields[2] = fmt.Sprintf("sent message [%d bytes, %d compressed bytes]", e.UncompressedByteSize, e.CompressedByteSize)
			case trace.MessageEventTypeRecv:
				row.Fields[2] = fmt.Sprintf("received message [%d bytes, %d compressed bytes]", e.UncompressedByteSize, e.CompressedByteSize)
			}
			out = append(out, row)
		}
	}
	for i := range out {
		if len(out[i].Fields[2]) > maxTraceMessageLength {
			out[i].Fields[2] = out[i].Fields[2][:maxTraceMessageLength]
		}
	}
	return out
}

func traceSpans(spanName string, spanType, spanSubtype int) []*trace.SpanData {
	internalTrace := internal.Trace.(interface {
		ReportActiveSpans(name string) []*trace.SpanData
		ReportSpansByError(name string, code int32) []*trace.SpanData
		ReportSpansByLatency(name string, minLatency, maxLatency time.Duration) []*trace.SpanData
	})
	var spans []*trace.SpanData
	switch spanType {
	case 0: // active
		spans = internalTrace.ReportActiveSpans(spanName)
	case 1: // latency
		var min, max time.Duration
		n := len(defaultLatencies)
		if spanSubtype == 0 {
			max = defaultLatencies[0]
		} else if spanSubtype == n {
			min, max = defaultLatencies[spanSubtype-1], (1<<63)-1
		} else if 0 < spanSubtype && spanSubtype < n {
			min, max = defaultLatencies[spanSubtype-1], defaultLatencies[spanSubtype]
		}
		spans = internalTrace.ReportSpansByLatency(spanName, min, max)
	case 2: // error
		spans = internalTrace.ReportSpansByError(spanName, 0)
	}
	return spans
}

func traceDataFromSpans(name string, spans []*trace.SpanData) traceData {
	data := traceData{
		Name: name,
		Num:  len(spans),
	}
	for _, s := range spans {
		data.Rows = append(data.Rows, traceRows(s)...)
	}
	return data
}

func writeTextTraces(w io.Writer, data traceData) {
	tw := tabwriter.NewWriter(w, 1, 8, 1, ' ', 0)
	fmt.Fprint(tw, "When\tElapsed(s)\tType\n")
	for _, r := range data.Rows {
		tw.Write([]byte(r.Fields[0]))
		tw.Write([]byte("\t"))
		tw.Write([]byte(r.Fields[1]))
		tw.Write([]byte("\t"))
		tw.Write([]byte(r.Fields[2]))
		if sc := r.SpanContext; sc != (trace.SpanContext{}) {
			fmt.Fprintf(tw, "trace_id: %s span_id: %s", sc.TraceID, sc.SpanID)
			if r.ParentSpanID != (trace.SpanID{}) {
				fmt.Fprintf(tw, " parent_span_id: %s", r.ParentSpanID)
			}
		}
		tw.Write([]byte("\n"))
	}
	tw.Flush()
}

type summaryPageData struct {
	Header             []string
	LatencyBucketNames []string
	Links              bool
	TracesEndpoint     string
	Rows               []summaryPageRow
}

type summaryPageRow struct {
	Name    string
	Active  int
	Latency []int
	Errors  int
}

func getSummaryPageData() summaryPageData {
	data := summaryPageData{
		Links:          true,
		TracesEndpoint: "tracez",
	}
	internalTrace := internal.Trace.(interface {
		ReportSpansPerMethod() map[string]internal.PerMethodSummary
	})
	for name, s := range internalTrace.ReportSpansPerMethod() {
		if len(data.Header) == 0 {
			data.Header = []string{"Name", "Active"}
			for _, b := range s.LatencyBuckets {
				l := b.MinLatency
				s := fmt.Sprintf(">%v", l)
				if l == 100*time.Second {
					s = ">100s"
				}
				data.Header = append(data.Header, s)
				data.LatencyBucketNames = append(data.LatencyBucketNames, s)
			}
			data.Header = append(data.Header, "Errors")
		}
		row := summaryPageRow{Name: name, Active: s.Active}
		for _, l := range s.LatencyBuckets {
			row.Latency = append(row.Latency, l.Size)
		}
		for _, e := range s.ErrorBuckets {
			row.Errors += e.Size
		}
		data.Rows = append(data.Rows, row)
	}
	sort.Slice(data.Rows, func(i, j int) bool {
		return data.Rows[i].Name < data.Rows[j].Name
	})
	return data
}
