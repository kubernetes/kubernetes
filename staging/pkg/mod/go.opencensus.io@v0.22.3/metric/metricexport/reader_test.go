// Copyright 2019, OpenCensus Authors
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

package metricexport

import (
	"context"
	"sync"
	"testing"
	"time"

	"go.opencensus.io/metric"
	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricproducer"
)

var (
	ir1        *IntervalReader
	ir2        *IntervalReader
	reader1    = NewReader(WithSpanName("test-export-span"))
	exporter1  = &metricExporter{}
	exporter2  = &metricExporter{}
	gaugeEntry *metric.Int64GaugeEntry
	duration1  = 1000 * time.Millisecond
	duration2  = 2000 * time.Millisecond
)

type metricExporter struct {
	sync.Mutex
	metrics []*metricdata.Metric
}

func (e *metricExporter) ExportMetrics(ctx context.Context, metrics []*metricdata.Metric) error {
	e.Lock()
	defer e.Unlock()

	e.metrics = append(e.metrics, metrics...)
	return nil
}

func init() {
	r := metric.NewRegistry()
	metricproducer.GlobalManager().AddProducer(r)
	g, _ := r.AddInt64Gauge("active_request",
		metric.WithDescription("Number of active requests, per method."),
		metric.WithUnit(metricdata.UnitDimensionless),
		metric.WithLabelKeys("method"))
	gaugeEntry, _ = g.GetEntry(metricdata.NewLabelValue("foo"))
}

func TestNewReaderWitDefaultOptions(t *testing.T) {
	r := NewReader()

	if r.spanName != defaultSpanName {
		t.Errorf("span name: got %v, want %v\n", r.spanName, defaultSpanName)
	}
}

func TestNewReaderWitSpanName(t *testing.T) {
	spanName := "test-span"
	r := NewReader(WithSpanName(spanName))

	if r.spanName != spanName {
		t.Errorf("span name: got %+v, want %v\n", r.spanName, spanName)
	}
}

func TestNewReader(t *testing.T) {
	r := NewReader()

	gaugeEntry.Add(1)

	r.ReadAndExport(exporter1)

	checkExportedCount(exporter1, 1, t)
	checkExportedMetricDesc(exporter1, "active_request", t)
	resetExporter(exporter1)
}

func TestNewIntervalReader(t *testing.T) {
	ir1 = createAndStart(exporter1, duration1, t)

	gaugeEntry.Add(1)

	time.Sleep(1500 * time.Millisecond)
	checkExportedCount(exporter1, 1, t)
	checkExportedMetricDesc(exporter1, "active_request", t)
	ir1.Stop()
	resetExporter(exporter1)
}

func TestManualReadForIntervalReader(t *testing.T) {
	ir1 = createAndStart(exporter1, duration1, t)

	gaugeEntry.Set(1)
	reader1.ReadAndExport(exporter1)
	gaugeEntry.Set(4)

	time.Sleep(1500 * time.Millisecond)

	checkExportedCount(exporter1, 2, t)
	checkExportedValues(exporter1, []int64{1, 4}, t) // one for manual read other for time based.
	checkExportedMetricDesc(exporter1, "active_request", t)
	ir1.Stop()
	resetExporter(exporter1)
}

func TestProducerWithIntervalReaderStop(t *testing.T) {
	ir1 = createAndStart(exporter1, duration1, t)
	ir1.Stop()

	gaugeEntry.Add(1)

	time.Sleep(1500 * time.Millisecond)

	checkExportedCount(exporter1, 0, t)
	checkExportedMetricDesc(exporter1, "active_request", t)
	resetExporter(exporter1)
}

func TestProducerWithMultipleIntervalReaders(t *testing.T) {
	ir1 = createAndStart(exporter1, duration1, t)
	ir2 = createAndStart(exporter2, duration2, t)

	gaugeEntry.Add(1)

	time.Sleep(2500 * time.Millisecond)

	checkExportedCount(exporter1, 2, t)
	checkExportedMetricDesc(exporter1, "active_request", t)
	checkExportedCount(exporter2, 1, t)
	checkExportedMetricDesc(exporter2, "active_request", t)
	ir1.Stop()
	ir2.Stop()
	resetExporter(exporter1)
	resetExporter(exporter1)
}

func TestIntervalReaderMultipleStop(t *testing.T) {
	ir1 = createAndStart(exporter1, duration1, t)
	stop := make(chan bool, 1)
	go func() {
		ir1.Stop()
		ir1.Stop()
		stop <- true
	}()

	select {
	case _ = <-stop:
	case <-time.After(1 * time.Second):
		t.Fatalf("ir1 stop got blocked")
	}
}

func TestIntervalReaderMultipleStart(t *testing.T) {
	ir1 = createAndStart(exporter1, duration1, t)
	ir1.Start()

	gaugeEntry.Add(1)

	time.Sleep(1500 * time.Millisecond)

	checkExportedCount(exporter1, 1, t)
	checkExportedMetricDesc(exporter1, "active_request", t)
	ir1.Stop()
	resetExporter(exporter1)
}

func TestNewIntervalReaderWithNilReader(t *testing.T) {
	_, err := NewIntervalReader(nil, exporter1)
	if err == nil {
		t.Fatalf("expected error but got nil\n")
	}
}

func TestNewIntervalReaderWithNilExporter(t *testing.T) {
	_, err := NewIntervalReader(reader1, nil)
	if err == nil {
		t.Fatalf("expected error but got nil\n")
	}
}

func TestNewIntervalReaderStartWithInvalidInterval(t *testing.T) {
	ir, err := NewIntervalReader(reader1, exporter1)
	ir.ReportingInterval = 500 * time.Millisecond
	err = ir.Start()
	if err == nil {
		t.Fatalf("expected error but got nil\n")
	}
}

func checkExportedCount(exporter *metricExporter, wantCount int, t *testing.T) {
	exporter.Lock()
	defer exporter.Unlock()
	gotCount := len(exporter.metrics)
	if gotCount != wantCount {
		t.Fatalf("exported metric count: got %d, want %d\n", gotCount, wantCount)
	}
}

func checkExportedValues(exporter *metricExporter, wantValues []int64, t *testing.T) {
	exporter.Lock()
	defer exporter.Unlock()
	gotCount := len(exporter.metrics)
	wantCount := len(wantValues)
	if gotCount != wantCount {
		t.Errorf("exported metric count: got %d, want %d\n", gotCount, wantCount)
		return
	}
	for i, wantValue := range wantValues {
		var gotValue int64
		switch v := exporter.metrics[i].TimeSeries[0].Points[0].Value.(type) {
		case int64:
			gotValue = v
		default:
			t.Errorf("expected float64 value but found other %T", exporter.metrics[i].TimeSeries[0].Points[0].Value)
		}
		if gotValue != wantValue {
			t.Errorf("values idx %d, got: %v, want %v", i, gotValue, wantValue)
		}
	}
}

func checkExportedMetricDesc(exporter *metricExporter, wantMdName string, t *testing.T) {
	exporter.Lock()
	defer exporter.Unlock()
	for _, metric := range exporter.metrics {
		gotMdName := metric.Descriptor.Name
		if gotMdName != wantMdName {
			t.Errorf("got %s, want %s\n", gotMdName, wantMdName)
		}
	}
	exporter.metrics = nil
}

func resetExporter(exporter *metricExporter) {
	exporter.Lock()
	defer exporter.Unlock()
	exporter.metrics = nil
}

// createAndStart stops the current processors and creates a new one.
func createAndStart(exporter *metricExporter, d time.Duration, t *testing.T) *IntervalReader {
	ir, _ := NewIntervalReader(reader1, exporter)
	ir.ReportingInterval = d
	err := ir.Start()
	if err != nil {
		t.Fatalf("error creating reader %v\n", err)
	}
	return ir
}
