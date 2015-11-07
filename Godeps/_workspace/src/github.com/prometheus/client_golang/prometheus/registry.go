// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2013, The Prometheus Authors
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file.

package prometheus

import (
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/golang/protobuf/proto"
	"github.com/prometheus/common/expfmt"

	dto "github.com/prometheus/client_model/go"
)

var (
	defRegistry   = newDefaultRegistry()
	errAlreadyReg = errors.New("duplicate metrics collector registration attempted")
)

// Constants relevant to the HTTP interface.
const (
	// APIVersion is the version of the format of the exported data.  This
	// will match this library's version, which subscribes to the Semantic
	// Versioning scheme.
	APIVersion = "0.0.4"

	// DelimitedTelemetryContentType is the content type set on telemetry
	// data responses in delimited protobuf format.
	DelimitedTelemetryContentType = `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited`
	// TextTelemetryContentType is the content type set on telemetry data
	// responses in text format.
	TextTelemetryContentType = `text/plain; version=` + APIVersion
	// ProtoTextTelemetryContentType is the content type set on telemetry
	// data responses in protobuf text format.  (Only used for debugging.)
	ProtoTextTelemetryContentType = `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=text`
	// ProtoCompactTextTelemetryContentType is the content type set on
	// telemetry data responses in protobuf compact text format.  (Only used
	// for debugging.)
	ProtoCompactTextTelemetryContentType = `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=compact-text`

	// Constants for object pools.
	numBufs           = 4
	numMetricFamilies = 1000
	numMetrics        = 10000

	// Capacity for the channel to collect metrics and descriptors.
	capMetricChan = 1000
	capDescChan   = 10

	contentTypeHeader     = "Content-Type"
	contentLengthHeader   = "Content-Length"
	contentEncodingHeader = "Content-Encoding"

	acceptEncodingHeader = "Accept-Encoding"
	acceptHeader         = "Accept"
)

// Handler returns the HTTP handler for the global Prometheus registry. It is
// already instrumented with InstrumentHandler (using "prometheus" as handler
// name). Usually the handler is used to handle the "/metrics" endpoint.
func Handler() http.Handler {
	return InstrumentHandler("prometheus", defRegistry)
}

// UninstrumentedHandler works in the same way as Handler, but the returned HTTP
// handler is not instrumented. This is useful if no instrumentation is desired
// (for whatever reason) or if the instrumentation has to happen with a
// different handler name (or with a different instrumentation approach
// altogether). See the InstrumentHandler example.
func UninstrumentedHandler() http.Handler {
	return defRegistry
}

// Register registers a new Collector to be included in metrics collection. It
// returns an error if the descriptors provided by the Collector are invalid or
// if they - in combination with descriptors of already registered Collectors -
// do not fulfill the consistency and uniqueness criteria described in the Desc
// documentation.
//
// Do not register the same Collector multiple times concurrently. (Registering
// the same Collector twice would result in an error anyway, but on top of that,
// it is not safe to do so concurrently.)
func Register(m Collector) error {
	_, err := defRegistry.Register(m)
	return err
}

// MustRegister works like Register but panics where Register would have
// returned an error.
func MustRegister(m Collector) {
	err := Register(m)
	if err != nil {
		panic(err)
	}
}

// RegisterOrGet works like Register but does not return an error if a Collector
// is registered that equals a previously registered Collector. (Two Collectors
// are considered equal if their Describe method yields the same set of
// descriptors.) Instead, the previously registered Collector is returned (which
// is helpful if the new and previously registered Collectors are equal but not
// identical, i.e. not pointers to the same object).
//
// As for Register, it is still not safe to call RegisterOrGet with the same
// Collector multiple times concurrently.
func RegisterOrGet(m Collector) (Collector, error) {
	return defRegistry.RegisterOrGet(m)
}

// MustRegisterOrGet works like Register but panics where RegisterOrGet would
// have returned an error.
func MustRegisterOrGet(m Collector) Collector {
	existing, err := RegisterOrGet(m)
	if err != nil {
		panic(err)
	}
	return existing
}

// Unregister unregisters the Collector that equals the Collector passed in as
// an argument. (Two Collectors are considered equal if their Describe method
// yields the same set of descriptors.) The function returns whether a Collector
// was unregistered.
func Unregister(c Collector) bool {
	return defRegistry.Unregister(c)
}

// SetMetricFamilyInjectionHook sets a function that is called whenever metrics
// are collected. The hook function must be set before metrics collection begins
// (i.e. call SetMetricFamilyInjectionHook before setting the HTTP handler.) The
// MetricFamily protobufs returned by the hook function are merged with the
// metrics collected in the usual way.
//
// This is a way to directly inject MetricFamily protobufs managed and owned by
// the caller. The caller has full responsibility. As no registration of the
// injected metrics has happened, there is no descriptor to check against, and
// there are no registration-time checks. If collect-time checks are disabled
// (see function EnableCollectChecks), no sanity checks are performed on the
// returned protobufs at all. If collect-checks are enabled, type and uniqueness
// checks are performed, but no further consistency checks (which would require
// knowledge of a metric descriptor).
//
// Sorting concerns: The caller is responsible for sorting the label pairs in
// each metric. However, the order of metrics will be sorted by the registry as
// it is required anyway after merging with the metric families collected
// conventionally.
//
// The function must be callable at any time and concurrently.
func SetMetricFamilyInjectionHook(hook func() []*dto.MetricFamily) {
	defRegistry.metricFamilyInjectionHook = hook
}

// PanicOnCollectError sets the behavior whether a panic is caused upon an error
// while metrics are collected and served to the HTTP endpoint. By default, an
// internal server error (status code 500) is served with an error message.
func PanicOnCollectError(b bool) {
	defRegistry.panicOnCollectError = b
}

// EnableCollectChecks enables (or disables) additional consistency checks
// during metrics collection. These additional checks are not enabled by default
// because they inflict a performance penalty and the errors they check for can
// only happen if the used Metric and Collector types have internal programming
// errors. It can be helpful to enable these checks while working with custom
// Collectors or Metrics whose correctness is not well established yet.
func EnableCollectChecks(b bool) {
	defRegistry.collectChecksEnabled = b
}

// encoder is a function that writes a dto.MetricFamily to an io.Writer in a
// certain encoding. It returns the number of bytes written and any error
// encountered.  Note that pbutil.WriteDelimited and pbutil.MetricFamilyToText
// are encoders.
type encoder func(io.Writer, *dto.MetricFamily) (int, error)

type registry struct {
	mtx                       sync.RWMutex
	collectorsByID            map[uint64]Collector // ID is a hash of the descIDs.
	descIDs                   map[uint64]struct{}
	dimHashesByName           map[string]uint64
	bufPool                   chan *bytes.Buffer
	metricFamilyPool          chan *dto.MetricFamily
	metricPool                chan *dto.Metric
	metricFamilyInjectionHook func() []*dto.MetricFamily

	panicOnCollectError, collectChecksEnabled bool
}

func (r *registry) Register(c Collector) (Collector, error) {
	descChan := make(chan *Desc, capDescChan)
	go func() {
		c.Describe(descChan)
		close(descChan)
	}()

	newDescIDs := map[uint64]struct{}{}
	newDimHashesByName := map[string]uint64{}
	var collectorID uint64 // Just a sum of all desc IDs.
	var duplicateDescErr error

	r.mtx.Lock()
	defer r.mtx.Unlock()
	// Coduct various tests...
	for desc := range descChan {

		// Is the descriptor valid at all?
		if desc.err != nil {
			return c, fmt.Errorf("descriptor %s is invalid: %s", desc, desc.err)
		}

		// Is the descID unique?
		// (In other words: Is the fqName + constLabel combination unique?)
		if _, exists := r.descIDs[desc.id]; exists {
			duplicateDescErr = fmt.Errorf("descriptor %s already exists with the same fully-qualified name and const label values", desc)
		}
		// If it is not a duplicate desc in this collector, add it to
		// the collectorID.  (We allow duplicate descs within the same
		// collector, but their existence must be a no-op.)
		if _, exists := newDescIDs[desc.id]; !exists {
			newDescIDs[desc.id] = struct{}{}
			collectorID += desc.id
		}

		// Are all the label names and the help string consistent with
		// previous descriptors of the same name?
		// First check existing descriptors...
		if dimHash, exists := r.dimHashesByName[desc.fqName]; exists {
			if dimHash != desc.dimHash {
				return nil, fmt.Errorf("a previously registered descriptor with the same fully-qualified name as %s has different label names or a different help string", desc)
			}
		} else {
			// ...then check the new descriptors already seen.
			if dimHash, exists := newDimHashesByName[desc.fqName]; exists {
				if dimHash != desc.dimHash {
					return nil, fmt.Errorf("descriptors reported by collector have inconsistent label names or help strings for the same fully-qualified name, offender is %s", desc)
				}
			} else {
				newDimHashesByName[desc.fqName] = desc.dimHash
			}
		}
	}
	// Did anything happen at all?
	if len(newDescIDs) == 0 {
		return nil, errors.New("collector has no descriptors")
	}
	if existing, exists := r.collectorsByID[collectorID]; exists {
		return existing, errAlreadyReg
	}
	// If the collectorID is new, but at least one of the descs existed
	// before, we are in trouble.
	if duplicateDescErr != nil {
		return nil, duplicateDescErr
	}

	// Only after all tests have passed, actually register.
	r.collectorsByID[collectorID] = c
	for hash := range newDescIDs {
		r.descIDs[hash] = struct{}{}
	}
	for name, dimHash := range newDimHashesByName {
		r.dimHashesByName[name] = dimHash
	}
	return c, nil
}

func (r *registry) RegisterOrGet(m Collector) (Collector, error) {
	existing, err := r.Register(m)
	if err != nil && err != errAlreadyReg {
		return nil, err
	}
	return existing, nil
}

func (r *registry) Unregister(c Collector) bool {
	descChan := make(chan *Desc, capDescChan)
	go func() {
		c.Describe(descChan)
		close(descChan)
	}()

	descIDs := map[uint64]struct{}{}
	var collectorID uint64 // Just a sum of the desc IDs.
	for desc := range descChan {
		if _, exists := descIDs[desc.id]; !exists {
			collectorID += desc.id
			descIDs[desc.id] = struct{}{}
		}
	}

	r.mtx.RLock()
	if _, exists := r.collectorsByID[collectorID]; !exists {
		r.mtx.RUnlock()
		return false
	}
	r.mtx.RUnlock()

	r.mtx.Lock()
	defer r.mtx.Unlock()

	delete(r.collectorsByID, collectorID)
	for id := range descIDs {
		delete(r.descIDs, id)
	}
	// dimHashesByName is left untouched as those must be consistent
	// throughout the lifetime of a program.
	return true
}

func (r *registry) Push(job, instance, pushURL, method string) error {
	if !strings.Contains(pushURL, "://") {
		pushURL = "http://" + pushURL
	}
	pushURL = fmt.Sprintf("%s/metrics/jobs/%s", pushURL, url.QueryEscape(job))
	if instance != "" {
		pushURL += "/instances/" + url.QueryEscape(instance)
	}
	buf := r.getBuf()
	defer r.giveBuf(buf)
	if err := r.writePB(expfmt.NewEncoder(buf, expfmt.FmtProtoDelim)); err != nil {
		if r.panicOnCollectError {
			panic(err)
		}
		return err
	}
	req, err := http.NewRequest(method, pushURL, buf)
	if err != nil {
		return err
	}
	req.Header.Set(contentTypeHeader, DelimitedTelemetryContentType)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 202 {
		return fmt.Errorf("unexpected status code %d while pushing to %s", resp.StatusCode, pushURL)
	}
	return nil
}

func (r *registry) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	contentType := expfmt.Negotiate(req.Header)
	buf := r.getBuf()
	defer r.giveBuf(buf)
	writer, encoding := decorateWriter(req, buf)
	if err := r.writePB(expfmt.NewEncoder(writer, contentType)); err != nil {
		if r.panicOnCollectError {
			panic(err)
		}
		http.Error(w, "An error has occurred:\n\n"+err.Error(), http.StatusInternalServerError)
		return
	}
	if closer, ok := writer.(io.Closer); ok {
		closer.Close()
	}
	header := w.Header()
	header.Set(contentTypeHeader, string(contentType))
	header.Set(contentLengthHeader, fmt.Sprint(buf.Len()))
	if encoding != "" {
		header.Set(contentEncodingHeader, encoding)
	}
	w.Write(buf.Bytes())
}

func (r *registry) writePB(encoder expfmt.Encoder) error {
	var metricHashes map[uint64]struct{}
	if r.collectChecksEnabled {
		metricHashes = make(map[uint64]struct{})
	}
	metricChan := make(chan Metric, capMetricChan)
	wg := sync.WaitGroup{}

	r.mtx.RLock()
	metricFamiliesByName := make(map[string]*dto.MetricFamily, len(r.dimHashesByName))

	// Scatter.
	// (Collectors could be complex and slow, so we call them all at once.)
	wg.Add(len(r.collectorsByID))
	go func() {
		wg.Wait()
		close(metricChan)
	}()
	for _, collector := range r.collectorsByID {
		go func(collector Collector) {
			defer wg.Done()
			collector.Collect(metricChan)
		}(collector)
	}
	r.mtx.RUnlock()

	// Drain metricChan in case of premature return.
	defer func() {
		for _ = range metricChan {
		}
	}()

	// Gather.
	for metric := range metricChan {
		// This could be done concurrently, too, but it required locking
		// of metricFamiliesByName (and of metricHashes if checks are
		// enabled). Most likely not worth it.
		desc := metric.Desc()
		metricFamily, ok := metricFamiliesByName[desc.fqName]
		if !ok {
			metricFamily = r.getMetricFamily()
			defer r.giveMetricFamily(metricFamily)
			metricFamily.Name = proto.String(desc.fqName)
			metricFamily.Help = proto.String(desc.help)
			metricFamiliesByName[desc.fqName] = metricFamily
		}
		dtoMetric := r.getMetric()
		defer r.giveMetric(dtoMetric)
		if err := metric.Write(dtoMetric); err != nil {
			// TODO: Consider different means of error reporting so
			// that a single erroneous metric could be skipped
			// instead of blowing up the whole collection.
			return fmt.Errorf("error collecting metric %v: %s", desc, err)
		}
		switch {
		case metricFamily.Type != nil:
			// Type already set. We are good.
		case dtoMetric.Gauge != nil:
			metricFamily.Type = dto.MetricType_GAUGE.Enum()
		case dtoMetric.Counter != nil:
			metricFamily.Type = dto.MetricType_COUNTER.Enum()
		case dtoMetric.Summary != nil:
			metricFamily.Type = dto.MetricType_SUMMARY.Enum()
		case dtoMetric.Untyped != nil:
			metricFamily.Type = dto.MetricType_UNTYPED.Enum()
		case dtoMetric.Histogram != nil:
			metricFamily.Type = dto.MetricType_HISTOGRAM.Enum()
		default:
			return fmt.Errorf("empty metric collected: %s", dtoMetric)
		}
		if r.collectChecksEnabled {
			if err := r.checkConsistency(metricFamily, dtoMetric, desc, metricHashes); err != nil {
				return err
			}
		}
		metricFamily.Metric = append(metricFamily.Metric, dtoMetric)
	}

	if r.metricFamilyInjectionHook != nil {
		for _, mf := range r.metricFamilyInjectionHook() {
			existingMF, exists := metricFamiliesByName[mf.GetName()]
			if !exists {
				metricFamiliesByName[mf.GetName()] = mf
				if r.collectChecksEnabled {
					for _, m := range mf.Metric {
						if err := r.checkConsistency(mf, m, nil, metricHashes); err != nil {
							return err
						}
					}
				}
				continue
			}
			for _, m := range mf.Metric {
				if r.collectChecksEnabled {
					if err := r.checkConsistency(existingMF, m, nil, metricHashes); err != nil {
						return err
					}
				}
				existingMF.Metric = append(existingMF.Metric, m)
			}
		}
	}

	// Now that MetricFamilies are all set, sort their Metrics
	// lexicographically by their label values.
	for _, mf := range metricFamiliesByName {
		sort.Sort(metricSorter(mf.Metric))
	}

	// Write out MetricFamilies sorted by their name.
	names := make([]string, 0, len(metricFamiliesByName))
	for name := range metricFamiliesByName {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		if err := encoder.Encode(metricFamiliesByName[name]); err != nil {
			return err
		}
	}
	return nil
}

func (r *registry) checkConsistency(metricFamily *dto.MetricFamily, dtoMetric *dto.Metric, desc *Desc, metricHashes map[uint64]struct{}) error {

	// Type consistency with metric family.
	if metricFamily.GetType() == dto.MetricType_GAUGE && dtoMetric.Gauge == nil ||
		metricFamily.GetType() == dto.MetricType_COUNTER && dtoMetric.Counter == nil ||
		metricFamily.GetType() == dto.MetricType_SUMMARY && dtoMetric.Summary == nil ||
		metricFamily.GetType() == dto.MetricType_HISTOGRAM && dtoMetric.Histogram == nil ||
		metricFamily.GetType() == dto.MetricType_UNTYPED && dtoMetric.Untyped == nil {
		return fmt.Errorf(
			"collected metric %s %s is not a %s",
			metricFamily.GetName(), dtoMetric, metricFamily.GetType(),
		)
	}

	// Is the metric unique (i.e. no other metric with the same name and the same label values)?
	h := fnv.New64a()
	var buf bytes.Buffer
	buf.WriteString(metricFamily.GetName())
	buf.WriteByte(separatorByte)
	h.Write(buf.Bytes())
	// Make sure label pairs are sorted. We depend on it for the consistency
	// check. Label pairs must be sorted by contract. But the point of this
	// method is to check for contract violations. So we better do the sort
	// now.
	sort.Sort(LabelPairSorter(dtoMetric.Label))
	for _, lp := range dtoMetric.Label {
		buf.Reset()
		buf.WriteString(lp.GetValue())
		buf.WriteByte(separatorByte)
		h.Write(buf.Bytes())
	}
	metricHash := h.Sum64()
	if _, exists := metricHashes[metricHash]; exists {
		return fmt.Errorf(
			"collected metric %s %s was collected before with the same name and label values",
			metricFamily.GetName(), dtoMetric,
		)
	}
	metricHashes[metricHash] = struct{}{}

	if desc == nil {
		return nil // Nothing left to check if we have no desc.
	}

	// Desc consistency with metric family.
	if metricFamily.GetName() != desc.fqName {
		return fmt.Errorf(
			"collected metric %s %s has name %q but should have %q",
			metricFamily.GetName(), dtoMetric, metricFamily.GetName(), desc.fqName,
		)
	}
	if metricFamily.GetHelp() != desc.help {
		return fmt.Errorf(
			"collected metric %s %s has help %q but should have %q",
			metricFamily.GetName(), dtoMetric, metricFamily.GetHelp(), desc.help,
		)
	}

	// Is the desc consistent with the content of the metric?
	lpsFromDesc := make([]*dto.LabelPair, 0, len(dtoMetric.Label))
	lpsFromDesc = append(lpsFromDesc, desc.constLabelPairs...)
	for _, l := range desc.variableLabels {
		lpsFromDesc = append(lpsFromDesc, &dto.LabelPair{
			Name: proto.String(l),
		})
	}
	if len(lpsFromDesc) != len(dtoMetric.Label) {
		return fmt.Errorf(
			"labels in collected metric %s %s are inconsistent with descriptor %s",
			metricFamily.GetName(), dtoMetric, desc,
		)
	}
	sort.Sort(LabelPairSorter(lpsFromDesc))
	for i, lpFromDesc := range lpsFromDesc {
		lpFromMetric := dtoMetric.Label[i]
		if lpFromDesc.GetName() != lpFromMetric.GetName() ||
			lpFromDesc.Value != nil && lpFromDesc.GetValue() != lpFromMetric.GetValue() {
			return fmt.Errorf(
				"labels in collected metric %s %s are inconsistent with descriptor %s",
				metricFamily.GetName(), dtoMetric, desc,
			)
		}
	}

	r.mtx.RLock() // Remaining checks need the read lock.
	defer r.mtx.RUnlock()

	// Is the desc registered?
	if _, exist := r.descIDs[desc.id]; !exist {
		return fmt.Errorf(
			"collected metric %s %s with unregistered descriptor %s",
			metricFamily.GetName(), dtoMetric, desc,
		)
	}

	return nil
}

func (r *registry) getBuf() *bytes.Buffer {
	select {
	case buf := <-r.bufPool:
		return buf
	default:
		return &bytes.Buffer{}
	}
}

func (r *registry) giveBuf(buf *bytes.Buffer) {
	buf.Reset()
	select {
	case r.bufPool <- buf:
	default:
	}
}

func (r *registry) getMetricFamily() *dto.MetricFamily {
	select {
	case mf := <-r.metricFamilyPool:
		return mf
	default:
		return &dto.MetricFamily{}
	}
}

func (r *registry) giveMetricFamily(mf *dto.MetricFamily) {
	mf.Reset()
	select {
	case r.metricFamilyPool <- mf:
	default:
	}
}

func (r *registry) getMetric() *dto.Metric {
	select {
	case m := <-r.metricPool:
		return m
	default:
		return &dto.Metric{}
	}
}

func (r *registry) giveMetric(m *dto.Metric) {
	m.Reset()
	select {
	case r.metricPool <- m:
	default:
	}
}

func newRegistry() *registry {
	return &registry{
		collectorsByID:   map[uint64]Collector{},
		descIDs:          map[uint64]struct{}{},
		dimHashesByName:  map[string]uint64{},
		bufPool:          make(chan *bytes.Buffer, numBufs),
		metricFamilyPool: make(chan *dto.MetricFamily, numMetricFamilies),
		metricPool:       make(chan *dto.Metric, numMetrics),
	}
}

func newDefaultRegistry() *registry {
	r := newRegistry()
	r.Register(NewProcessCollector(os.Getpid(), ""))
	r.Register(NewGoCollector())
	return r
}

// decorateWriter wraps a writer to handle gzip compression if requested.  It
// returns the decorated writer and the appropriate "Content-Encoding" header
// (which is empty if no compression is enabled).
func decorateWriter(request *http.Request, writer io.Writer) (io.Writer, string) {
	header := request.Header.Get(acceptEncodingHeader)
	parts := strings.Split(header, ",")
	for _, part := range parts {
		part := strings.TrimSpace(part)
		if part == "gzip" || strings.HasPrefix(part, "gzip;") {
			return gzip.NewWriter(writer), "gzip"
		}
	}
	return writer, ""
}

type metricSorter []*dto.Metric

func (s metricSorter) Len() int {
	return len(s)
}

func (s metricSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s metricSorter) Less(i, j int) bool {
	if len(s[i].Label) != len(s[j].Label) {
		// This should not happen. The metrics are
		// inconsistent. However, we have to deal with the fact, as
		// people might use custom collectors or metric family injection
		// to create inconsistent metrics. So let's simply compare the
		// number of labels in this case. That will still yield
		// reproducible sorting.
		return len(s[i].Label) < len(s[j].Label)
	}
	for n, lp := range s[i].Label {
		vi := lp.GetValue()
		vj := s[j].Label[n].GetValue()
		if vi != vj {
			return vi < vj
		}
	}
	return true
}
