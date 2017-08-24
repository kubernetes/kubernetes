// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bytes"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"text/tabwriter"
	"time"
)

const maxEventsPerLog = 100

type bucket struct {
	MaxErrAge time.Duration
	String    string
}

var buckets = []bucket{
	{0, "total"},
	{10 * time.Second, "errs<10s"},
	{1 * time.Minute, "errs<1m"},
	{10 * time.Minute, "errs<10m"},
	{1 * time.Hour, "errs<1h"},
	{10 * time.Hour, "errs<10h"},
	{24000 * time.Hour, "errors"},
}

// RenderEvents renders the HTML page typically served at /debug/events.
// It does not do any auth checking. The request may be nil.
//
// Most users will use the Events handler.
func RenderEvents(w http.ResponseWriter, req *http.Request, sensitive bool) {
	now := time.Now()
	data := &struct {
		Families []string // family names
		Buckets  []bucket
		Counts   [][]int // eventLog count per family/bucket

		// Set when a bucket has been selected.
		Family    string
		Bucket    int
		EventLogs eventLogs
		Expanded  bool
	}{
		Buckets: buckets,
	}

	data.Families = make([]string, 0, len(families))
	famMu.RLock()
	for name := range families {
		data.Families = append(data.Families, name)
	}
	famMu.RUnlock()
	sort.Strings(data.Families)

	// Count the number of eventLogs in each family for each error age.
	data.Counts = make([][]int, len(data.Families))
	for i, name := range data.Families {
		// TODO(sameer): move this loop under the family lock.
		f := getEventFamily(name)
		data.Counts[i] = make([]int, len(data.Buckets))
		for j, b := range data.Buckets {
			data.Counts[i][j] = f.Count(now, b.MaxErrAge)
		}
	}

	if req != nil {
		var ok bool
		data.Family, data.Bucket, ok = parseEventsArgs(req)
		if !ok {
			// No-op
		} else {
			data.EventLogs = getEventFamily(data.Family).Copy(now, buckets[data.Bucket].MaxErrAge)
		}
		if data.EventLogs != nil {
			defer data.EventLogs.Free()
			sort.Sort(data.EventLogs)
		}
		if exp, err := strconv.ParseBool(req.FormValue("exp")); err == nil {
			data.Expanded = exp
		}
	}

	famMu.RLock()
	defer famMu.RUnlock()
	if err := eventsTmpl().Execute(w, data); err != nil {
		log.Printf("net/trace: Failed executing template: %v", err)
	}
}

func parseEventsArgs(req *http.Request) (fam string, b int, ok bool) {
	fam, bStr := req.FormValue("fam"), req.FormValue("b")
	if fam == "" || bStr == "" {
		return "", 0, false
	}
	b, err := strconv.Atoi(bStr)
	if err != nil || b < 0 || b >= len(buckets) {
		return "", 0, false
	}
	return fam, b, true
}

// An EventLog provides a log of events associated with a specific object.
type EventLog interface {
	// Printf formats its arguments with fmt.Sprintf and adds the
	// result to the event log.
	Printf(format string, a ...interface{})

	// Errorf is like Printf, but it marks this event as an error.
	Errorf(format string, a ...interface{})

	// Finish declares that this event log is complete.
	// The event log should not be used after calling this method.
	Finish()
}

// NewEventLog returns a new EventLog with the specified family name
// and title.
func NewEventLog(family, title string) EventLog {
	el := newEventLog()
	el.ref()
	el.Family, el.Title = family, title
	el.Start = time.Now()
	el.events = make([]logEntry, 0, maxEventsPerLog)
	el.stack = make([]uintptr, 32)
	n := runtime.Callers(2, el.stack)
	el.stack = el.stack[:n]

	getEventFamily(family).add(el)
	return el
}

func (el *eventLog) Finish() {
	getEventFamily(el.Family).remove(el)
	el.unref() // matches ref in New
}

var (
	famMu    sync.RWMutex
	families = make(map[string]*eventFamily) // family name => family
)

func getEventFamily(fam string) *eventFamily {
	famMu.Lock()
	defer famMu.Unlock()
	f := families[fam]
	if f == nil {
		f = &eventFamily{}
		families[fam] = f
	}
	return f
}

type eventFamily struct {
	mu        sync.RWMutex
	eventLogs eventLogs
}

func (f *eventFamily) add(el *eventLog) {
	f.mu.Lock()
	f.eventLogs = append(f.eventLogs, el)
	f.mu.Unlock()
}

func (f *eventFamily) remove(el *eventLog) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for i, el0 := range f.eventLogs {
		if el == el0 {
			copy(f.eventLogs[i:], f.eventLogs[i+1:])
			f.eventLogs = f.eventLogs[:len(f.eventLogs)-1]
			return
		}
	}
}

func (f *eventFamily) Count(now time.Time, maxErrAge time.Duration) (n int) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	for _, el := range f.eventLogs {
		if el.hasRecentError(now, maxErrAge) {
			n++
		}
	}
	return
}

func (f *eventFamily) Copy(now time.Time, maxErrAge time.Duration) (els eventLogs) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	els = make(eventLogs, 0, len(f.eventLogs))
	for _, el := range f.eventLogs {
		if el.hasRecentError(now, maxErrAge) {
			el.ref()
			els = append(els, el)
		}
	}
	return
}

type eventLogs []*eventLog

// Free calls unref on each element of the list.
func (els eventLogs) Free() {
	for _, el := range els {
		el.unref()
	}
}

// eventLogs may be sorted in reverse chronological order.
func (els eventLogs) Len() int           { return len(els) }
func (els eventLogs) Less(i, j int) bool { return els[i].Start.After(els[j].Start) }
func (els eventLogs) Swap(i, j int)      { els[i], els[j] = els[j], els[i] }

// A logEntry is a timestamped log entry in an event log.
type logEntry struct {
	When    time.Time
	Elapsed time.Duration // since previous event in log
	NewDay  bool          // whether this event is on a different day to the previous event
	What    string
	IsErr   bool
}

// WhenString returns a string representation of the elapsed time of the event.
// It will include the date if midnight was crossed.
func (e logEntry) WhenString() string {
	if e.NewDay {
		return e.When.Format("2006/01/02 15:04:05.000000")
	}
	return e.When.Format("15:04:05.000000")
}

// An eventLog represents an active event log.
type eventLog struct {
	// Family is the top-level grouping of event logs to which this belongs.
	Family string

	// Title is the title of this event log.
	Title string

	// Timing information.
	Start time.Time

	// Call stack where this event log was created.
	stack []uintptr

	// Append-only sequence of events.
	//
	// TODO(sameer): change this to a ring buffer to avoid the array copy
	// when we hit maxEventsPerLog.
	mu            sync.RWMutex
	events        []logEntry
	LastErrorTime time.Time
	discarded     int

	refs int32 // how many buckets this is in
}

func (el *eventLog) reset() {
	// Clear all but the mutex. Mutexes may not be copied, even when unlocked.
	el.Family = ""
	el.Title = ""
	el.Start = time.Time{}
	el.stack = nil
	el.events = nil
	el.LastErrorTime = time.Time{}
	el.discarded = 0
	el.refs = 0
}

func (el *eventLog) hasRecentError(now time.Time, maxErrAge time.Duration) bool {
	if maxErrAge == 0 {
		return true
	}
	el.mu.RLock()
	defer el.mu.RUnlock()
	return now.Sub(el.LastErrorTime) < maxErrAge
}

// delta returns the elapsed time since the last event or the log start,
// and whether it spans midnight.
// L >= el.mu
func (el *eventLog) delta(t time.Time) (time.Duration, bool) {
	if len(el.events) == 0 {
		return t.Sub(el.Start), false
	}
	prev := el.events[len(el.events)-1].When
	return t.Sub(prev), prev.Day() != t.Day()

}

func (el *eventLog) Printf(format string, a ...interface{}) {
	el.printf(false, format, a...)
}

func (el *eventLog) Errorf(format string, a ...interface{}) {
	el.printf(true, format, a...)
}

func (el *eventLog) printf(isErr bool, format string, a ...interface{}) {
	e := logEntry{When: time.Now(), IsErr: isErr, What: fmt.Sprintf(format, a...)}
	el.mu.Lock()
	e.Elapsed, e.NewDay = el.delta(e.When)
	if len(el.events) < maxEventsPerLog {
		el.events = append(el.events, e)
	} else {
		// Discard the oldest event.
		if el.discarded == 0 {
			// el.discarded starts at two to count for the event it
			// is replacing, plus the next one that we are about to
			// drop.
			el.discarded = 2
		} else {
			el.discarded++
		}
		// TODO(sameer): if this causes allocations on a critical path,
		// change eventLog.What to be a fmt.Stringer, as in trace.go.
		el.events[0].What = fmt.Sprintf("(%d events discarded)", el.discarded)
		// The timestamp of the discarded meta-event should be
		// the time of the last event it is representing.
		el.events[0].When = el.events[1].When
		copy(el.events[1:], el.events[2:])
		el.events[maxEventsPerLog-1] = e
	}
	if e.IsErr {
		el.LastErrorTime = e.When
	}
	el.mu.Unlock()
}

func (el *eventLog) ref() {
	atomic.AddInt32(&el.refs, 1)
}

func (el *eventLog) unref() {
	if atomic.AddInt32(&el.refs, -1) == 0 {
		freeEventLog(el)
	}
}

func (el *eventLog) When() string {
	return el.Start.Format("2006/01/02 15:04:05.000000")
}

func (el *eventLog) ElapsedTime() string {
	elapsed := time.Since(el.Start)
	return fmt.Sprintf("%.6f", elapsed.Seconds())
}

func (el *eventLog) Stack() string {
	buf := new(bytes.Buffer)
	tw := tabwriter.NewWriter(buf, 1, 8, 1, '\t', 0)
	printStackRecord(tw, el.stack)
	tw.Flush()
	return buf.String()
}

// printStackRecord prints the function + source line information
// for a single stack trace.
// Adapted from runtime/pprof/pprof.go.
func printStackRecord(w io.Writer, stk []uintptr) {
	for _, pc := range stk {
		f := runtime.FuncForPC(pc)
		if f == nil {
			continue
		}
		file, line := f.FileLine(pc)
		name := f.Name()
		// Hide runtime.goexit and any runtime functions at the beginning.
		if strings.HasPrefix(name, "runtime.") {
			continue
		}
		fmt.Fprintf(w, "#   %s\t%s:%d\n", name, file, line)
	}
}

func (el *eventLog) Events() []logEntry {
	el.mu.RLock()
	defer el.mu.RUnlock()
	return el.events
}

// freeEventLogs is a freelist of *eventLog
var freeEventLogs = make(chan *eventLog, 1000)

// newEventLog returns a event log ready to use.
func newEventLog() *eventLog {
	select {
	case el := <-freeEventLogs:
		return el
	default:
		return new(eventLog)
	}
}

// freeEventLog adds el to freeEventLogs if there's room.
// This is non-blocking.
func freeEventLog(el *eventLog) {
	el.reset()
	select {
	case freeEventLogs <- el:
	default:
	}
}

var eventsTmplCache *template.Template
var eventsTmplOnce sync.Once

func eventsTmpl() *template.Template {
	eventsTmplOnce.Do(func() {
		eventsTmplCache = template.Must(template.New("events").Funcs(template.FuncMap{
			"elapsed":   elapsed,
			"trimSpace": strings.TrimSpace,
		}).Parse(eventsHTML))
	})
	return eventsTmplCache
}

const eventsHTML = `
<html>
	<head>
		<title>events</title>
	</head>
	<style type="text/css">
		body {
			font-family: sans-serif;
		}
		table#req-status td.family {
			padding-right: 2em;
		}
		table#req-status td.active {
			padding-right: 1em;
		}
		table#req-status td.empty {
			color: #aaa;
		}
		table#reqs {
			margin-top: 1em;
		}
		table#reqs tr.first {
			{{if $.Expanded}}font-weight: bold;{{end}}
		}
		table#reqs td {
			font-family: monospace;
		}
		table#reqs td.when {
			text-align: right;
			white-space: nowrap;
		}
		table#reqs td.elapsed {
			padding: 0 0.5em;
			text-align: right;
			white-space: pre;
			width: 10em;
		}
		address {
			font-size: smaller;
			margin-top: 5em;
		}
	</style>
	<body>

<h1>/debug/events</h1>

<table id="req-status">
	{{range $i, $fam := .Families}}
	<tr>
		<td class="family">{{$fam}}</td>

	        {{range $j, $bucket := $.Buckets}}
	        {{$n := index $.Counts $i $j}}
		<td class="{{if not $bucket.MaxErrAge}}active{{end}}{{if not $n}}empty{{end}}">
	                {{if $n}}<a href="?fam={{$fam}}&b={{$j}}{{if $.Expanded}}&exp=1{{end}}">{{end}}
		        [{{$n}} {{$bucket.String}}]
			{{if $n}}</a>{{end}}
		</td>
                {{end}}

	</tr>{{end}}
</table>

{{if $.EventLogs}}
<hr />
<h3>Family: {{$.Family}}</h3>

{{if $.Expanded}}<a href="?fam={{$.Family}}&b={{$.Bucket}}">{{end}}
[Summary]{{if $.Expanded}}</a>{{end}}

{{if not $.Expanded}}<a href="?fam={{$.Family}}&b={{$.Bucket}}&exp=1">{{end}}
[Expanded]{{if not $.Expanded}}</a>{{end}}

<table id="reqs">
	<tr><th>When</th><th>Elapsed</th></tr>
	{{range $el := $.EventLogs}}
	<tr class="first">
		<td class="when">{{$el.When}}</td>
		<td class="elapsed">{{$el.ElapsedTime}}</td>
		<td>{{$el.Title}}
	</tr>
	{{if $.Expanded}}
	<tr>
		<td class="when"></td>
		<td class="elapsed"></td>
		<td><pre>{{$el.Stack|trimSpace}}</pre></td>
	</tr>
	{{range $el.Events}}
	<tr>
		<td class="when">{{.WhenString}}</td>
		<td class="elapsed">{{elapsed .Elapsed}}</td>
		<td>.{{if .IsErr}}E{{else}}.{{end}}. {{.What}}</td>
	</tr>
	{{end}}
	{{end}}
	{{end}}
</table>
{{end}}
	</body>
</html>
`
