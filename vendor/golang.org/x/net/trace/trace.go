// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package trace implements tracing of requests and long-lived objects.
It exports HTTP interfaces on /debug/requests and /debug/events.

A trace.Trace provides tracing for short-lived objects, usually requests.
A request handler might be implemented like this:

	func fooHandler(w http.ResponseWriter, req *http.Request) {
		tr := trace.New("mypkg.Foo", req.URL.Path)
		defer tr.Finish()
		...
		tr.LazyPrintf("some event %q happened", str)
		...
		if err := somethingImportant(); err != nil {
			tr.LazyPrintf("somethingImportant failed: %v", err)
			tr.SetError()
		}
	}

The /debug/requests HTTP endpoint organizes the traces by family,
errors, and duration.  It also provides histogram of request duration
for each family.

A trace.EventLog provides tracing for long-lived objects, such as RPC
connections.

	// A Fetcher fetches URL paths for a single domain.
	type Fetcher struct {
		domain string
		events trace.EventLog
	}

	func NewFetcher(domain string) *Fetcher {
		return &Fetcher{
			domain,
			trace.NewEventLog("mypkg.Fetcher", domain),
		}
	}

	func (f *Fetcher) Fetch(path string) (string, error) {
		resp, err := http.Get("http://" + f.domain + "/" + path)
		if err != nil {
			f.events.Errorf("Get(%q) = %v", path, err)
			return "", err
		}
		f.events.Printf("Get(%q) = %s", path, resp.Status)
		...
	}

	func (f *Fetcher) Close() error {
		f.events.Finish()
		return nil
	}

The /debug/events HTTP endpoint organizes the event logs by family and
by time since the last error.  The expanded view displays recent log
entries and the log's call stack.
*/
package trace

import (
	"bytes"
	"fmt"
	"html/template"
	"io"
	"log"
	"net"
	"net/http"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/net/internal/timeseries"
)

// DebugUseAfterFinish controls whether to debug uses of Trace values after finishing.
// FOR DEBUGGING ONLY. This will slow down the program.
var DebugUseAfterFinish = false

// AuthRequest determines whether a specific request is permitted to load the
// /debug/requests or /debug/events pages.
//
// It returns two bools; the first indicates whether the page may be viewed at all,
// and the second indicates whether sensitive events will be shown.
//
// AuthRequest may be replaced by a program to customize its authorization requirements.
//
// The default AuthRequest function returns (true, true) if and only if the request
// comes from localhost/127.0.0.1/[::1].
var AuthRequest = func(req *http.Request) (any, sensitive bool) {
	// RemoteAddr is commonly in the form "IP" or "IP:port".
	// If it is in the form "IP:port", split off the port.
	host, _, err := net.SplitHostPort(req.RemoteAddr)
	if err != nil {
		host = req.RemoteAddr
	}
	switch host {
	case "localhost", "127.0.0.1", "::1":
		return true, true
	default:
		return false, false
	}
}

func init() {
	// TODO(jbd): Serve Traces from /debug/traces in the future?
	// There is no requirement for a request to be present to have traces.
	http.HandleFunc("/debug/requests", Traces)
	http.HandleFunc("/debug/events", Events)
}

// Traces responds with traces from the program.
// The package initialization registers it in http.DefaultServeMux
// at /debug/requests.
//
// It performs authorization by running AuthRequest.
func Traces(w http.ResponseWriter, req *http.Request) {
	any, sensitive := AuthRequest(req)
	if !any {
		http.Error(w, "not allowed", http.StatusUnauthorized)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	Render(w, req, sensitive)
}

// Events responds with a page of events collected by EventLogs.
// The package initialization registers it in http.DefaultServeMux
// at /debug/events.
//
// It performs authorization by running AuthRequest.
func Events(w http.ResponseWriter, req *http.Request) {
	any, sensitive := AuthRequest(req)
	if !any {
		http.Error(w, "not allowed", http.StatusUnauthorized)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	RenderEvents(w, req, sensitive)
}

// Render renders the HTML page typically served at /debug/requests.
// It does not do any auth checking. The request may be nil.
//
// Most users will use the Traces handler.
func Render(w io.Writer, req *http.Request, sensitive bool) {
	data := &struct {
		Families         []string
		ActiveTraceCount map[string]int
		CompletedTraces  map[string]*family

		// Set when a bucket has been selected.
		Traces        traceList
		Family        string
		Bucket        int
		Expanded      bool
		Traced        bool
		Active        bool
		ShowSensitive bool // whether to show sensitive events

		Histogram       template.HTML
		HistogramWindow string // e.g. "last minute", "last hour", "all time"

		// If non-zero, the set of traces is a partial set,
		// and this is the total number.
		Total int
	}{
		CompletedTraces: completedTraces,
	}

	data.ShowSensitive = sensitive
	if req != nil {
		// Allow show_sensitive=0 to force hiding of sensitive data for testing.
		// This only goes one way; you can't use show_sensitive=1 to see things.
		if req.FormValue("show_sensitive") == "0" {
			data.ShowSensitive = false
		}

		if exp, err := strconv.ParseBool(req.FormValue("exp")); err == nil {
			data.Expanded = exp
		}
		if exp, err := strconv.ParseBool(req.FormValue("rtraced")); err == nil {
			data.Traced = exp
		}
	}

	completedMu.RLock()
	data.Families = make([]string, 0, len(completedTraces))
	for fam := range completedTraces {
		data.Families = append(data.Families, fam)
	}
	completedMu.RUnlock()
	sort.Strings(data.Families)

	// We are careful here to minimize the time spent locking activeMu,
	// since that lock is required every time an RPC starts and finishes.
	data.ActiveTraceCount = make(map[string]int, len(data.Families))
	activeMu.RLock()
	for fam, s := range activeTraces {
		data.ActiveTraceCount[fam] = s.Len()
	}
	activeMu.RUnlock()

	var ok bool
	data.Family, data.Bucket, ok = parseArgs(req)
	switch {
	case !ok:
		// No-op
	case data.Bucket == -1:
		data.Active = true
		n := data.ActiveTraceCount[data.Family]
		data.Traces = getActiveTraces(data.Family)
		if len(data.Traces) < n {
			data.Total = n
		}
	case data.Bucket < bucketsPerFamily:
		if b := lookupBucket(data.Family, data.Bucket); b != nil {
			data.Traces = b.Copy(data.Traced)
		}
	default:
		if f := getFamily(data.Family, false); f != nil {
			var obs timeseries.Observable
			f.LatencyMu.RLock()
			switch o := data.Bucket - bucketsPerFamily; o {
			case 0:
				obs = f.Latency.Minute()
				data.HistogramWindow = "last minute"
			case 1:
				obs = f.Latency.Hour()
				data.HistogramWindow = "last hour"
			case 2:
				obs = f.Latency.Total()
				data.HistogramWindow = "all time"
			}
			f.LatencyMu.RUnlock()
			if obs != nil {
				data.Histogram = obs.(*histogram).html()
			}
		}
	}

	if data.Traces != nil {
		defer data.Traces.Free()
		sort.Sort(data.Traces)
	}

	completedMu.RLock()
	defer completedMu.RUnlock()
	if err := pageTmpl().ExecuteTemplate(w, "Page", data); err != nil {
		log.Printf("net/trace: Failed executing template: %v", err)
	}
}

func parseArgs(req *http.Request) (fam string, b int, ok bool) {
	if req == nil {
		return "", 0, false
	}
	fam, bStr := req.FormValue("fam"), req.FormValue("b")
	if fam == "" || bStr == "" {
		return "", 0, false
	}
	b, err := strconv.Atoi(bStr)
	if err != nil || b < -1 {
		return "", 0, false
	}

	return fam, b, true
}

func lookupBucket(fam string, b int) *traceBucket {
	f := getFamily(fam, false)
	if f == nil || b < 0 || b >= len(f.Buckets) {
		return nil
	}
	return f.Buckets[b]
}

type contextKeyT string

var contextKey = contextKeyT("golang.org/x/net/trace.Trace")

// Trace represents an active request.
type Trace interface {
	// LazyLog adds x to the event log. It will be evaluated each time the
	// /debug/requests page is rendered. Any memory referenced by x will be
	// pinned until the trace is finished and later discarded.
	LazyLog(x fmt.Stringer, sensitive bool)

	// LazyPrintf evaluates its arguments with fmt.Sprintf each time the
	// /debug/requests page is rendered. Any memory referenced by a will be
	// pinned until the trace is finished and later discarded.
	LazyPrintf(format string, a ...interface{})

	// SetError declares that this trace resulted in an error.
	SetError()

	// SetRecycler sets a recycler for the trace.
	// f will be called for each event passed to LazyLog at a time when
	// it is no longer required, whether while the trace is still active
	// and the event is discarded, or when a completed trace is discarded.
	SetRecycler(f func(interface{}))

	// SetTraceInfo sets the trace info for the trace.
	// This is currently unused.
	SetTraceInfo(traceID, spanID uint64)

	// SetMaxEvents sets the maximum number of events that will be stored
	// in the trace. This has no effect if any events have already been
	// added to the trace.
	SetMaxEvents(m int)

	// Finish declares that this trace is complete.
	// The trace should not be used after calling this method.
	Finish()
}

type lazySprintf struct {
	format string
	a      []interface{}
}

func (l *lazySprintf) String() string {
	return fmt.Sprintf(l.format, l.a...)
}

// New returns a new Trace with the specified family and title.
func New(family, title string) Trace {
	tr := newTrace()
	tr.ref()
	tr.Family, tr.Title = family, title
	tr.Start = time.Now()
	tr.maxEvents = maxEventsPerTrace
	tr.events = tr.eventsBuf[:0]

	activeMu.RLock()
	s := activeTraces[tr.Family]
	activeMu.RUnlock()
	if s == nil {
		activeMu.Lock()
		s = activeTraces[tr.Family] // check again
		if s == nil {
			s = new(traceSet)
			activeTraces[tr.Family] = s
		}
		activeMu.Unlock()
	}
	s.Add(tr)

	// Trigger allocation of the completed trace structure for this family.
	// This will cause the family to be present in the request page during
	// the first trace of this family. We don't care about the return value,
	// nor is there any need for this to run inline, so we execute it in its
	// own goroutine, but only if the family isn't allocated yet.
	completedMu.RLock()
	if _, ok := completedTraces[tr.Family]; !ok {
		go allocFamily(tr.Family)
	}
	completedMu.RUnlock()

	return tr
}

func (tr *trace) Finish() {
	tr.Elapsed = time.Now().Sub(tr.Start)
	if DebugUseAfterFinish {
		buf := make([]byte, 4<<10) // 4 KB should be enough
		n := runtime.Stack(buf, false)
		tr.finishStack = buf[:n]
	}

	activeMu.RLock()
	m := activeTraces[tr.Family]
	activeMu.RUnlock()
	m.Remove(tr)

	f := getFamily(tr.Family, true)
	for _, b := range f.Buckets {
		if b.Cond.match(tr) {
			b.Add(tr)
		}
	}
	// Add a sample of elapsed time as microseconds to the family's timeseries
	h := new(histogram)
	h.addMeasurement(tr.Elapsed.Nanoseconds() / 1e3)
	f.LatencyMu.Lock()
	f.Latency.Add(h)
	f.LatencyMu.Unlock()

	tr.unref() // matches ref in New
}

const (
	bucketsPerFamily    = 9
	tracesPerBucket     = 10
	maxActiveTraces     = 20 // Maximum number of active traces to show.
	maxEventsPerTrace   = 10
	numHistogramBuckets = 38
)

var (
	// The active traces.
	activeMu     sync.RWMutex
	activeTraces = make(map[string]*traceSet) // family -> traces

	// Families of completed traces.
	completedMu     sync.RWMutex
	completedTraces = make(map[string]*family) // family -> traces
)

type traceSet struct {
	mu sync.RWMutex
	m  map[*trace]bool

	// We could avoid the entire map scan in FirstN by having a slice of all the traces
	// ordered by start time, and an index into that from the trace struct, with a periodic
	// repack of the slice after enough traces finish; we could also use a skip list or similar.
	// However, that would shift some of the expense from /debug/requests time to RPC time,
	// which is probably the wrong trade-off.
}

func (ts *traceSet) Len() int {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	return len(ts.m)
}

func (ts *traceSet) Add(tr *trace) {
	ts.mu.Lock()
	if ts.m == nil {
		ts.m = make(map[*trace]bool)
	}
	ts.m[tr] = true
	ts.mu.Unlock()
}

func (ts *traceSet) Remove(tr *trace) {
	ts.mu.Lock()
	delete(ts.m, tr)
	ts.mu.Unlock()
}

// FirstN returns the first n traces ordered by time.
func (ts *traceSet) FirstN(n int) traceList {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if n > len(ts.m) {
		n = len(ts.m)
	}
	trl := make(traceList, 0, n)

	// Fast path for when no selectivity is needed.
	if n == len(ts.m) {
		for tr := range ts.m {
			tr.ref()
			trl = append(trl, tr)
		}
		sort.Sort(trl)
		return trl
	}

	// Pick the oldest n traces.
	// This is inefficient. See the comment in the traceSet struct.
	for tr := range ts.m {
		// Put the first n traces into trl in the order they occur.
		// When we have n, sort trl, and thereafter maintain its order.
		if len(trl) < n {
			tr.ref()
			trl = append(trl, tr)
			if len(trl) == n {
				// This is guaranteed to happen exactly once during this loop.
				sort.Sort(trl)
			}
			continue
		}
		if tr.Start.After(trl[n-1].Start) {
			continue
		}

		// Find where to insert this one.
		tr.ref()
		i := sort.Search(n, func(i int) bool { return trl[i].Start.After(tr.Start) })
		trl[n-1].unref()
		copy(trl[i+1:], trl[i:])
		trl[i] = tr
	}

	return trl
}

func getActiveTraces(fam string) traceList {
	activeMu.RLock()
	s := activeTraces[fam]
	activeMu.RUnlock()
	if s == nil {
		return nil
	}
	return s.FirstN(maxActiveTraces)
}

func getFamily(fam string, allocNew bool) *family {
	completedMu.RLock()
	f := completedTraces[fam]
	completedMu.RUnlock()
	if f == nil && allocNew {
		f = allocFamily(fam)
	}
	return f
}

func allocFamily(fam string) *family {
	completedMu.Lock()
	defer completedMu.Unlock()
	f := completedTraces[fam]
	if f == nil {
		f = newFamily()
		completedTraces[fam] = f
	}
	return f
}

// family represents a set of trace buckets and associated latency information.
type family struct {
	// traces may occur in multiple buckets.
	Buckets [bucketsPerFamily]*traceBucket

	// latency time series
	LatencyMu sync.RWMutex
	Latency   *timeseries.MinuteHourSeries
}

func newFamily() *family {
	return &family{
		Buckets: [bucketsPerFamily]*traceBucket{
			{Cond: minCond(0)},
			{Cond: minCond(50 * time.Millisecond)},
			{Cond: minCond(100 * time.Millisecond)},
			{Cond: minCond(200 * time.Millisecond)},
			{Cond: minCond(500 * time.Millisecond)},
			{Cond: minCond(1 * time.Second)},
			{Cond: minCond(10 * time.Second)},
			{Cond: minCond(100 * time.Second)},
			{Cond: errorCond{}},
		},
		Latency: timeseries.NewMinuteHourSeries(func() timeseries.Observable { return new(histogram) }),
	}
}

// traceBucket represents a size-capped bucket of historic traces,
// along with a condition for a trace to belong to the bucket.
type traceBucket struct {
	Cond cond

	// Ring buffer implementation of a fixed-size FIFO queue.
	mu     sync.RWMutex
	buf    [tracesPerBucket]*trace
	start  int // < tracesPerBucket
	length int // <= tracesPerBucket
}

func (b *traceBucket) Add(tr *trace) {
	b.mu.Lock()
	defer b.mu.Unlock()

	i := b.start + b.length
	if i >= tracesPerBucket {
		i -= tracesPerBucket
	}
	if b.length == tracesPerBucket {
		// "Remove" an element from the bucket.
		b.buf[i].unref()
		b.start++
		if b.start == tracesPerBucket {
			b.start = 0
		}
	}
	b.buf[i] = tr
	if b.length < tracesPerBucket {
		b.length++
	}
	tr.ref()
}

// Copy returns a copy of the traces in the bucket.
// If tracedOnly is true, only the traces with trace information will be returned.
// The logs will be ref'd before returning; the caller should call
// the Free method when it is done with them.
// TODO(dsymonds): keep track of traced requests in separate buckets.
func (b *traceBucket) Copy(tracedOnly bool) traceList {
	b.mu.RLock()
	defer b.mu.RUnlock()

	trl := make(traceList, 0, b.length)
	for i, x := 0, b.start; i < b.length; i++ {
		tr := b.buf[x]
		if !tracedOnly || tr.spanID != 0 {
			tr.ref()
			trl = append(trl, tr)
		}
		x++
		if x == b.length {
			x = 0
		}
	}
	return trl
}

func (b *traceBucket) Empty() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.length == 0
}

// cond represents a condition on a trace.
type cond interface {
	match(t *trace) bool
	String() string
}

type minCond time.Duration

func (m minCond) match(t *trace) bool { return t.Elapsed >= time.Duration(m) }
func (m minCond) String() string      { return fmt.Sprintf("≥%gs", time.Duration(m).Seconds()) }

type errorCond struct{}

func (e errorCond) match(t *trace) bool { return t.IsError }
func (e errorCond) String() string      { return "errors" }

type traceList []*trace

// Free calls unref on each element of the list.
func (trl traceList) Free() {
	for _, t := range trl {
		t.unref()
	}
}

// traceList may be sorted in reverse chronological order.
func (trl traceList) Len() int           { return len(trl) }
func (trl traceList) Less(i, j int) bool { return trl[i].Start.After(trl[j].Start) }
func (trl traceList) Swap(i, j int)      { trl[i], trl[j] = trl[j], trl[i] }

// An event is a timestamped log entry in a trace.
type event struct {
	When       time.Time
	Elapsed    time.Duration // since previous event in trace
	NewDay     bool          // whether this event is on a different day to the previous event
	Recyclable bool          // whether this event was passed via LazyLog
	Sensitive  bool          // whether this event contains sensitive information
	What       interface{}   // string or fmt.Stringer
}

// WhenString returns a string representation of the elapsed time of the event.
// It will include the date if midnight was crossed.
func (e event) WhenString() string {
	if e.NewDay {
		return e.When.Format("2006/01/02 15:04:05.000000")
	}
	return e.When.Format("15:04:05.000000")
}

// discarded represents a number of discarded events.
// It is stored as *discarded to make it easier to update in-place.
type discarded int

func (d *discarded) String() string {
	return fmt.Sprintf("(%d events discarded)", int(*d))
}

// trace represents an active or complete request,
// either sent or received by this program.
type trace struct {
	// Family is the top-level grouping of traces to which this belongs.
	Family string

	// Title is the title of this trace.
	Title string

	// Timing information.
	Start   time.Time
	Elapsed time.Duration // zero while active

	// Trace information if non-zero.
	traceID uint64
	spanID  uint64

	// Whether this trace resulted in an error.
	IsError bool

	// Append-only sequence of events (modulo discards).
	mu        sync.RWMutex
	events    []event
	maxEvents int

	refs     int32 // how many buckets this is in
	recycler func(interface{})
	disc     discarded // scratch space to avoid allocation

	finishStack []byte // where finish was called, if DebugUseAfterFinish is set

	eventsBuf [4]event // preallocated buffer in case we only log a few events
}

func (tr *trace) reset() {
	// Clear all but the mutex. Mutexes may not be copied, even when unlocked.
	tr.Family = ""
	tr.Title = ""
	tr.Start = time.Time{}
	tr.Elapsed = 0
	tr.traceID = 0
	tr.spanID = 0
	tr.IsError = false
	tr.maxEvents = 0
	tr.events = nil
	tr.refs = 0
	tr.recycler = nil
	tr.disc = 0
	tr.finishStack = nil
	for i := range tr.eventsBuf {
		tr.eventsBuf[i] = event{}
	}
}

// delta returns the elapsed time since the last event or the trace start,
// and whether it spans midnight.
// L >= tr.mu
func (tr *trace) delta(t time.Time) (time.Duration, bool) {
	if len(tr.events) == 0 {
		return t.Sub(tr.Start), false
	}
	prev := tr.events[len(tr.events)-1].When
	return t.Sub(prev), prev.Day() != t.Day()
}

func (tr *trace) addEvent(x interface{}, recyclable, sensitive bool) {
	if DebugUseAfterFinish && tr.finishStack != nil {
		buf := make([]byte, 4<<10) // 4 KB should be enough
		n := runtime.Stack(buf, false)
		log.Printf("net/trace: trace used after finish:\nFinished at:\n%s\nUsed at:\n%s", tr.finishStack, buf[:n])
	}

	/*
		NOTE TO DEBUGGERS

		If you are here because your program panicked in this code,
		it is almost definitely the fault of code using this package,
		and very unlikely to be the fault of this code.

		The most likely scenario is that some code elsewhere is using
		a trace.Trace after its Finish method is called.
		You can temporarily set the DebugUseAfterFinish var
		to help discover where that is; do not leave that var set,
		since it makes this package much less efficient.
	*/

	e := event{When: time.Now(), What: x, Recyclable: recyclable, Sensitive: sensitive}
	tr.mu.Lock()
	e.Elapsed, e.NewDay = tr.delta(e.When)
	if len(tr.events) < tr.maxEvents {
		tr.events = append(tr.events, e)
	} else {
		// Discard the middle events.
		di := int((tr.maxEvents - 1) / 2)
		if d, ok := tr.events[di].What.(*discarded); ok {
			(*d)++
		} else {
			// disc starts at two to count for the event it is replacing,
			// plus the next one that we are about to drop.
			tr.disc = 2
			if tr.recycler != nil && tr.events[di].Recyclable {
				go tr.recycler(tr.events[di].What)
			}
			tr.events[di].What = &tr.disc
		}
		// The timestamp of the discarded meta-event should be
		// the time of the last event it is representing.
		tr.events[di].When = tr.events[di+1].When

		if tr.recycler != nil && tr.events[di+1].Recyclable {
			go tr.recycler(tr.events[di+1].What)
		}
		copy(tr.events[di+1:], tr.events[di+2:])
		tr.events[tr.maxEvents-1] = e
	}
	tr.mu.Unlock()
}

func (tr *trace) LazyLog(x fmt.Stringer, sensitive bool) {
	tr.addEvent(x, true, sensitive)
}

func (tr *trace) LazyPrintf(format string, a ...interface{}) {
	tr.addEvent(&lazySprintf{format, a}, false, false)
}

func (tr *trace) SetError() { tr.IsError = true }

func (tr *trace) SetRecycler(f func(interface{})) {
	tr.recycler = f
}

func (tr *trace) SetTraceInfo(traceID, spanID uint64) {
	tr.traceID, tr.spanID = traceID, spanID
}

func (tr *trace) SetMaxEvents(m int) {
	// Always keep at least three events: first, discarded count, last.
	if len(tr.events) == 0 && m > 3 {
		tr.maxEvents = m
	}
}

func (tr *trace) ref() {
	atomic.AddInt32(&tr.refs, 1)
}

func (tr *trace) unref() {
	if atomic.AddInt32(&tr.refs, -1) == 0 {
		if tr.recycler != nil {
			// freeTrace clears tr, so we hold tr.recycler and tr.events here.
			go func(f func(interface{}), es []event) {
				for _, e := range es {
					if e.Recyclable {
						f(e.What)
					}
				}
			}(tr.recycler, tr.events)
		}

		freeTrace(tr)
	}
}

func (tr *trace) When() string {
	return tr.Start.Format("2006/01/02 15:04:05.000000")
}

func (tr *trace) ElapsedTime() string {
	t := tr.Elapsed
	if t == 0 {
		// Active trace.
		t = time.Since(tr.Start)
	}
	return fmt.Sprintf("%.6f", t.Seconds())
}

func (tr *trace) Events() []event {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	return tr.events
}

var traceFreeList = make(chan *trace, 1000) // TODO(dsymonds): Use sync.Pool?

// newTrace returns a trace ready to use.
func newTrace() *trace {
	select {
	case tr := <-traceFreeList:
		return tr
	default:
		return new(trace)
	}
}

// freeTrace adds tr to traceFreeList if there's room.
// This is non-blocking.
func freeTrace(tr *trace) {
	if DebugUseAfterFinish {
		return // never reuse
	}
	tr.reset()
	select {
	case traceFreeList <- tr:
	default:
	}
}

func elapsed(d time.Duration) string {
	b := []byte(fmt.Sprintf("%.6f", d.Seconds()))

	// For subsecond durations, blank all zeros before decimal point,
	// and all zeros between the decimal point and the first non-zero digit.
	if d < time.Second {
		dot := bytes.IndexByte(b, '.')
		for i := 0; i < dot; i++ {
			b[i] = ' '
		}
		for i := dot + 1; i < len(b); i++ {
			if b[i] == '0' {
				b[i] = ' '
			} else {
				break
			}
		}
	}

	return string(b)
}

var pageTmplCache *template.Template
var pageTmplOnce sync.Once

func pageTmpl() *template.Template {
	pageTmplOnce.Do(func() {
		pageTmplCache = template.Must(template.New("Page").Funcs(template.FuncMap{
			"elapsed": elapsed,
			"add":     func(a, b int) int { return a + b },
		}).Parse(pageHTML))
	})
	return pageTmplCache
}

const pageHTML = `
{{template "Prolog" .}}
{{template "StatusTable" .}}
{{template "Epilog" .}}

{{define "Prolog"}}
<html>
	<head>
	<title>/debug/requests</title>
	<style type="text/css">
		body {
			font-family: sans-serif;
		}
		table#tr-status td.family {
			padding-right: 2em;
		}
		table#tr-status td.active {
			padding-right: 1em;
		}
		table#tr-status td.latency-first {
			padding-left: 1em;
		}
		table#tr-status td.empty {
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
	</head>
	<body>

<h1>/debug/requests</h1>
{{end}} {{/* end of Prolog */}}

{{define "StatusTable"}}
<table id="tr-status">
	{{range $fam := .Families}}
	<tr>
		<td class="family">{{$fam}}</td>

		{{$n := index $.ActiveTraceCount $fam}}
		<td class="active {{if not $n}}empty{{end}}">
			{{if $n}}<a href="?fam={{$fam}}&b=-1{{if $.Expanded}}&exp=1{{end}}">{{end}}
			[{{$n}} active]
			{{if $n}}</a>{{end}}
		</td>

		{{$f := index $.CompletedTraces $fam}}
		{{range $i, $b := $f.Buckets}}
		{{$empty := $b.Empty}}
		<td {{if $empty}}class="empty"{{end}}>
		{{if not $empty}}<a href="?fam={{$fam}}&b={{$i}}{{if $.Expanded}}&exp=1{{end}}">{{end}}
		[{{.Cond}}]
		{{if not $empty}}</a>{{end}}
		</td>
		{{end}}

		{{$nb := len $f.Buckets}}
		<td class="latency-first">
		<a href="?fam={{$fam}}&b={{$nb}}">[minute]</a>
		</td>
		<td>
		<a href="?fam={{$fam}}&b={{add $nb 1}}">[hour]</a>
		</td>
		<td>
		<a href="?fam={{$fam}}&b={{add $nb 2}}">[total]</a>
		</td>

	</tr>
	{{end}}
</table>
{{end}} {{/* end of StatusTable */}}

{{define "Epilog"}}
{{if $.Traces}}
<hr />
<h3>Family: {{$.Family}}</h3>

{{if or $.Expanded $.Traced}}
  <a href="?fam={{$.Family}}&b={{$.Bucket}}">[Normal/Summary]</a>
{{else}}
  [Normal/Summary]
{{end}}

{{if or (not $.Expanded) $.Traced}}
  <a href="?fam={{$.Family}}&b={{$.Bucket}}&exp=1">[Normal/Expanded]</a>
{{else}}
  [Normal/Expanded]
{{end}}

{{if not $.Active}}
	{{if or $.Expanded (not $.Traced)}}
	<a href="?fam={{$.Family}}&b={{$.Bucket}}&rtraced=1">[Traced/Summary]</a>
	{{else}}
	[Traced/Summary]
	{{end}}
	{{if or (not $.Expanded) (not $.Traced)}}
	<a href="?fam={{$.Family}}&b={{$.Bucket}}&exp=1&rtraced=1">[Traced/Expanded]</a>
        {{else}}
	[Traced/Expanded]
	{{end}}
{{end}}

{{if $.Total}}
<p><em>Showing <b>{{len $.Traces}}</b> of <b>{{$.Total}}</b> traces.</em></p>
{{end}}

<table id="reqs">
	<caption>
		{{if $.Active}}Active{{else}}Completed{{end}} Requests
	</caption>
	<tr><th>When</th><th>Elapsed&nbsp;(s)</th></tr>
	{{range $tr := $.Traces}}
	<tr class="first">
		<td class="when">{{$tr.When}}</td>
		<td class="elapsed">{{$tr.ElapsedTime}}</td>
		<td>{{$tr.Title}}</td>
		{{/* TODO: include traceID/spanID */}}
	</tr>
	{{if $.Expanded}}
	{{range $tr.Events}}
	<tr>
		<td class="when">{{.WhenString}}</td>
		<td class="elapsed">{{elapsed .Elapsed}}</td>
		<td>{{if or $.ShowSensitive (not .Sensitive)}}... {{.What}}{{else}}<em>[redacted]</em>{{end}}</td>
	</tr>
	{{end}}
	{{end}}
	{{end}}
</table>
{{end}} {{/* if $.Traces */}}

{{if $.Histogram}}
<h4>Latency (&micro;s) of {{$.Family}} over {{$.HistogramWindow}}</h4>
{{$.Histogram}}
{{end}} {{/* if $.Histogram */}}

	</body>
</html>
{{end}} {{/* end of Epilog */}}
`
