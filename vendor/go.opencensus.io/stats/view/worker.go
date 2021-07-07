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

package view

import (
	"fmt"
	"sync"
	"time"

	"go.opencensus.io/resource"

	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricproducer"
	"go.opencensus.io/stats"
	"go.opencensus.io/stats/internal"
	"go.opencensus.io/tag"
)

func init() {
	defaultWorker = NewMeter().(*worker)
	go defaultWorker.start()
	internal.DefaultRecorder = record
}

type measureRef struct {
	measure string
	views   map[*viewInternal]struct{}
}

type worker struct {
	measures   map[string]*measureRef
	views      map[string]*viewInternal
	startTimes map[*viewInternal]time.Time

	timer      *time.Ticker
	c          chan command
	quit, done chan bool
	mu         sync.RWMutex
	r          *resource.Resource

	exportersMu sync.RWMutex
	exporters   map[Exporter]struct{}
}

// Meter defines an interface which allows a single process to maintain
// multiple sets of metrics exports (intended for the advanced case where a
// single process wants to report metrics about multiple objects, such as
// multiple databases or HTTP services).
//
// Note that this is an advanced use case, and the static functions in this
// module should cover the common use cases.
type Meter interface {
	stats.Recorder
	// Find returns a registered view associated with this name.
	// If no registered view is found, nil is returned.
	Find(name string) *View
	// Register begins collecting data for the given views.
	// Once a view is registered, it reports data to the registered exporters.
	Register(views ...*View) error
	// Unregister the given views. Data will not longer be exported for these views
	// after Unregister returns.
	// It is not necessary to unregister from views you expect to collect for the
	// duration of your program execution.
	Unregister(views ...*View)
	// SetReportingPeriod sets the interval between reporting aggregated views in
	// the program. If duration is less than or equal to zero, it enables the
	// default behavior.
	//
	// Note: each exporter makes different promises about what the lowest supported
	// duration is. For example, the Stackdriver exporter recommends a value no
	// lower than 1 minute. Consult each exporter per your needs.
	SetReportingPeriod(time.Duration)

	// RegisterExporter registers an exporter.
	// Collected data will be reported via all the
	// registered exporters. Once you no longer
	// want data to be exported, invoke UnregisterExporter
	// with the previously registered exporter.
	//
	// Binaries can register exporters, libraries shouldn't register exporters.
	RegisterExporter(Exporter)
	// UnregisterExporter unregisters an exporter.
	UnregisterExporter(Exporter)
	// SetResource may be used to set the Resource associated with this registry.
	// This is intended to be used in cases where a single process exports metrics
	// for multiple Resources, typically in a multi-tenant situation.
	SetResource(*resource.Resource)

	// Start causes the Meter to start processing Record calls and aggregating
	// statistics as well as exporting data.
	Start()
	// Stop causes the Meter to stop processing calls and terminate data export.
	Stop()

	// RetrieveData gets a snapshot of the data collected for the the view registered
	// with the given name. It is intended for testing only.
	RetrieveData(viewName string) ([]*Row, error)
}

var _ Meter = (*worker)(nil)

var defaultWorker *worker

var defaultReportingDuration = 10 * time.Second

// Find returns a registered view associated with this name.
// If no registered view is found, nil is returned.
func Find(name string) (v *View) {
	return defaultWorker.Find(name)
}

// Find returns a registered view associated with this name.
// If no registered view is found, nil is returned.
func (w *worker) Find(name string) (v *View) {
	req := &getViewByNameReq{
		name: name,
		c:    make(chan *getViewByNameResp),
	}
	w.c <- req
	resp := <-req.c
	return resp.v
}

// Register begins collecting data for the given views.
// Once a view is registered, it reports data to the registered exporters.
func Register(views ...*View) error {
	return defaultWorker.Register(views...)
}

// Register begins collecting data for the given views.
// Once a view is registered, it reports data to the registered exporters.
func (w *worker) Register(views ...*View) error {
	req := &registerViewReq{
		views: views,
		err:   make(chan error),
	}
	w.c <- req
	return <-req.err
}

// Unregister the given views. Data will not longer be exported for these views
// after Unregister returns.
// It is not necessary to unregister from views you expect to collect for the
// duration of your program execution.
func Unregister(views ...*View) {
	defaultWorker.Unregister(views...)
}

// Unregister the given views. Data will not longer be exported for these views
// after Unregister returns.
// It is not necessary to unregister from views you expect to collect for the
// duration of your program execution.
func (w *worker) Unregister(views ...*View) {
	names := make([]string, len(views))
	for i := range views {
		names[i] = views[i].Name
	}
	req := &unregisterFromViewReq{
		views: names,
		done:  make(chan struct{}),
	}
	w.c <- req
	<-req.done
}

// RetrieveData gets a snapshot of the data collected for the the view registered
// with the given name. It is intended for testing only.
func RetrieveData(viewName string) ([]*Row, error) {
	return defaultWorker.RetrieveData(viewName)
}

// RetrieveData gets a snapshot of the data collected for the the view registered
// with the given name. It is intended for testing only.
func (w *worker) RetrieveData(viewName string) ([]*Row, error) {
	req := &retrieveDataReq{
		now: time.Now(),
		v:   viewName,
		c:   make(chan *retrieveDataResp),
	}
	w.c <- req
	resp := <-req.c
	return resp.rows, resp.err
}

func record(tags *tag.Map, ms interface{}, attachments map[string]interface{}) {
	defaultWorker.Record(tags, ms, attachments)
}

// Record records a set of measurements ms associated with the given tags and attachments.
func (w *worker) Record(tags *tag.Map, ms interface{}, attachments map[string]interface{}) {
	req := &recordReq{
		tm:          tags,
		ms:          ms.([]stats.Measurement),
		attachments: attachments,
		t:           time.Now(),
	}
	w.c <- req
}

// SetReportingPeriod sets the interval between reporting aggregated views in
// the program. If duration is less than or equal to zero, it enables the
// default behavior.
//
// Note: each exporter makes different promises about what the lowest supported
// duration is. For example, the Stackdriver exporter recommends a value no
// lower than 1 minute. Consult each exporter per your needs.
func SetReportingPeriod(d time.Duration) {
	defaultWorker.SetReportingPeriod(d)
}

// SetReportingPeriod sets the interval between reporting aggregated views in
// the program. If duration is less than or equal to zero, it enables the
// default behavior.
//
// Note: each exporter makes different promises about what the lowest supported
// duration is. For example, the Stackdriver exporter recommends a value no
// lower than 1 minute. Consult each exporter per your needs.
func (w *worker) SetReportingPeriod(d time.Duration) {
	// TODO(acetechnologist): ensure that the duration d is more than a certain
	// value. e.g. 1s
	req := &setReportingPeriodReq{
		d: d,
		c: make(chan bool),
	}
	w.c <- req
	<-req.c // don't return until the timer is set to the new duration.
}

// NewMeter constructs a Meter instance. You should only need to use this if
// you need to separate out Measurement recordings and View aggregations within
// a single process.
func NewMeter() Meter {
	return &worker{
		measures:   make(map[string]*measureRef),
		views:      make(map[string]*viewInternal),
		startTimes: make(map[*viewInternal]time.Time),
		timer:      time.NewTicker(defaultReportingDuration),
		c:          make(chan command, 1024),
		quit:       make(chan bool),
		done:       make(chan bool),

		exporters: make(map[Exporter]struct{}),
	}
}

// SetResource associates all data collected by this Meter with the specified
// resource. This resource is reported when using metricexport.ReadAndExport;
// it is not provided when used with ExportView/RegisterExporter, because that
// interface does not provide a means for reporting the Resource.
func (w *worker) SetResource(r *resource.Resource) {
	w.r = r
}

func (w *worker) Start() {
	go w.start()
}

func (w *worker) start() {
	prodMgr := metricproducer.GlobalManager()
	prodMgr.AddProducer(w)

	for {
		select {
		case cmd := <-w.c:
			cmd.handleCommand(w)
		case <-w.timer.C:
			w.reportUsage()
		case <-w.quit:
			w.timer.Stop()
			close(w.c)
			w.done <- true
			return
		}
	}
}

func (w *worker) Stop() {
	prodMgr := metricproducer.GlobalManager()
	prodMgr.DeleteProducer(w)

	w.quit <- true
	<-w.done
}

func (w *worker) getMeasureRef(name string) *measureRef {
	if mr, ok := w.measures[name]; ok {
		return mr
	}
	mr := &measureRef{
		measure: name,
		views:   make(map[*viewInternal]struct{}),
	}
	w.measures[name] = mr
	return mr
}

func (w *worker) tryRegisterView(v *View) (*viewInternal, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	vi, err := newViewInternal(v)
	if err != nil {
		return nil, err
	}
	if x, ok := w.views[vi.view.Name]; ok {
		if !x.view.same(vi.view) {
			return nil, fmt.Errorf("cannot register view %q; a different view with the same name is already registered", v.Name)
		}

		// the view is already registered so there is nothing to do and the
		// command is considered successful.
		return x, nil
	}
	w.views[vi.view.Name] = vi
	w.startTimes[vi] = time.Now()
	ref := w.getMeasureRef(vi.view.Measure.Name())
	ref.views[vi] = struct{}{}
	return vi, nil
}

func (w *worker) unregisterView(v *viewInternal) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.views, v.view.Name)
	delete(w.startTimes, v)
	if measure := w.measures[v.view.Measure.Name()]; measure != nil {
		delete(measure.views, v)
	}
}

func (w *worker) reportView(v *viewInternal) {
	if !v.isSubscribed() {
		return
	}
	rows := v.collectedRows()
	viewData := &Data{
		View:  v.view,
		Start: w.startTimes[v],
		End:   time.Now(),
		Rows:  rows,
	}
	w.exportersMu.Lock()
	defer w.exportersMu.Unlock()
	for e := range w.exporters {
		e.ExportView(viewData)
	}
}

func (w *worker) reportUsage() {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, v := range w.views {
		w.reportView(v)
	}
}

func (w *worker) toMetric(v *viewInternal, now time.Time) *metricdata.Metric {
	if !v.isSubscribed() {
		return nil
	}

	var startTime time.Time
	if v.metricDescriptor.Type == metricdata.TypeGaugeInt64 ||
		v.metricDescriptor.Type == metricdata.TypeGaugeFloat64 {
		startTime = time.Time{}
	} else {
		startTime = w.startTimes[v]
	}

	return viewToMetric(v, w.r, now, startTime)
}

// Read reads all view data and returns them as metrics.
// It is typically invoked by metric reader to export stats in metric format.
func (w *worker) Read() []*metricdata.Metric {
	w.mu.Lock()
	defer w.mu.Unlock()
	now := time.Now()
	metrics := make([]*metricdata.Metric, 0, len(w.views))
	for _, v := range w.views {
		metric := w.toMetric(v, now)
		if metric != nil {
			metrics = append(metrics, metric)
		}
	}
	return metrics
}

func (w *worker) RegisterExporter(e Exporter) {
	w.exportersMu.Lock()
	defer w.exportersMu.Unlock()

	w.exporters[e] = struct{}{}
}

func (w *worker) UnregisterExporter(e Exporter) {
	w.exportersMu.Lock()
	defer w.exportersMu.Unlock()

	delete(w.exporters, e)
}
