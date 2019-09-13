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

	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricproducer"
	"go.opencensus.io/stats"
	"go.opencensus.io/stats/internal"
	"go.opencensus.io/tag"
)

func init() {
	defaultWorker = newWorker()
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
}

var defaultWorker *worker

var defaultReportingDuration = 10 * time.Second

// Find returns a registered view associated with this name.
// If no registered view is found, nil is returned.
func Find(name string) (v *View) {
	req := &getViewByNameReq{
		name: name,
		c:    make(chan *getViewByNameResp),
	}
	defaultWorker.c <- req
	resp := <-req.c
	return resp.v
}

// Register begins collecting data for the given views.
// Once a view is registered, it reports data to the registered exporters.
func Register(views ...*View) error {
	req := &registerViewReq{
		views: views,
		err:   make(chan error),
	}
	defaultWorker.c <- req
	return <-req.err
}

// Unregister the given views. Data will not longer be exported for these views
// after Unregister returns.
// It is not necessary to unregister from views you expect to collect for the
// duration of your program execution.
func Unregister(views ...*View) {
	names := make([]string, len(views))
	for i := range views {
		names[i] = views[i].Name
	}
	req := &unregisterFromViewReq{
		views: names,
		done:  make(chan struct{}),
	}
	defaultWorker.c <- req
	<-req.done
}

// RetrieveData gets a snapshot of the data collected for the the view registered
// with the given name. It is intended for testing only.
func RetrieveData(viewName string) ([]*Row, error) {
	req := &retrieveDataReq{
		now: time.Now(),
		v:   viewName,
		c:   make(chan *retrieveDataResp),
	}
	defaultWorker.c <- req
	resp := <-req.c
	return resp.rows, resp.err
}

func record(tags *tag.Map, ms interface{}, attachments map[string]interface{}) {
	req := &recordReq{
		tm:          tags,
		ms:          ms.([]stats.Measurement),
		attachments: attachments,
		t:           time.Now(),
	}
	defaultWorker.c <- req
}

// SetReportingPeriod sets the interval between reporting aggregated views in
// the program. If duration is less than or equal to zero, it enables the
// default behavior.
//
// Note: each exporter makes different promises about what the lowest supported
// duration is. For example, the Stackdriver exporter recommends a value no
// lower than 1 minute. Consult each exporter per your needs.
func SetReportingPeriod(d time.Duration) {
	// TODO(acetechnologist): ensure that the duration d is more than a certain
	// value. e.g. 1s
	req := &setReportingPeriodReq{
		d: d,
		c: make(chan bool),
	}
	defaultWorker.c <- req
	<-req.c // don't return until the timer is set to the new duration.
}

func newWorker() *worker {
	return &worker{
		measures:   make(map[string]*measureRef),
		views:      make(map[string]*viewInternal),
		startTimes: make(map[*viewInternal]time.Time),
		timer:      time.NewTicker(defaultReportingDuration),
		c:          make(chan command, 1024),
		quit:       make(chan bool),
		done:       make(chan bool),
	}
}

func (w *worker) start() {
	prodMgr := metricproducer.GlobalManager()
	prodMgr.AddProducer(w)

	for {
		select {
		case cmd := <-w.c:
			cmd.handleCommand(w)
		case <-w.timer.C:
			w.reportUsage(time.Now())
		case <-w.quit:
			w.timer.Stop()
			close(w.c)
			w.done <- true
			return
		}
	}
}

func (w *worker) stop() {
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
	ref := w.getMeasureRef(vi.view.Measure.Name())
	ref.views[vi] = struct{}{}
	return vi, nil
}

func (w *worker) unregisterView(viewName string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.views, viewName)
}

func (w *worker) reportView(v *viewInternal, now time.Time) {
	if !v.isSubscribed() {
		return
	}
	rows := v.collectedRows()
	_, ok := w.startTimes[v]
	if !ok {
		w.startTimes[v] = now
	}
	viewData := &Data{
		View:  v.view,
		Start: w.startTimes[v],
		End:   time.Now(),
		Rows:  rows,
	}
	exportersMu.Lock()
	for e := range exporters {
		e.ExportView(viewData)
	}
	exportersMu.Unlock()
}

func (w *worker) reportUsage(now time.Time) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, v := range w.views {
		w.reportView(v, now)
	}
}

func (w *worker) toMetric(v *viewInternal, now time.Time) *metricdata.Metric {
	if !v.isSubscribed() {
		return nil
	}

	_, ok := w.startTimes[v]
	if !ok {
		w.startTimes[v] = now
	}

	var startTime time.Time
	if v.metricDescriptor.Type == metricdata.TypeGaugeInt64 ||
		v.metricDescriptor.Type == metricdata.TypeGaugeFloat64 {
		startTime = time.Time{}
	} else {
		startTime = w.startTimes[v]
	}

	return viewToMetric(v, now, startTime)
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
