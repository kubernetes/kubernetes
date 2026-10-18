/*
Copyright The Kubernetes Authors.

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

package compression

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func resetMetrics() {
	operationsTotal.Reset()
	durationSeconds.Reset()
	formatErrorsTotal.Reset()
	unsupportedAlgorithmTotal.Reset()
	rewritesNeededTotal.Reset()
	inflationWaitSeconds.Reset()
}

const (
	operationsHeader = `
# HELP apiserver_storage_compression_operations_total [ALPHA] Number of storage compression operations by resource, operation and outcome. The outcome tells compressed, stored verbatim and unframed values apart.
# TYPE apiserver_storage_compression_operations_total counter
`
	rewritesHeader = `
# HELP apiserver_storage_compression_rewrites_needed_total [ALPHA] Number of reads that reported a value stale because its compression framing disagreed with the policy, by resource and direction.
# TYPE apiserver_storage_compression_rewrites_needed_total counter
`
	formatErrorsHeader = `
# HELP apiserver_storage_compression_format_errors_total [ALPHA] Number of storage compression frames that failed their checksum or did not parse, by resource.
# TYPE apiserver_storage_compression_format_errors_total counter
`
	unsupportedHeader = `
# HELP apiserver_storage_compression_unsupported_algorithm_total [ALPHA] Number of storage values read whose compression algorithm this apiserver does not implement, by resource. A newer apiserver wrote them.
# TYPE apiserver_storage_compression_unsupported_algorithm_total counter
`
)

func TestOperationMetrics(t *testing.T) {
	RegisterMetrics()
	write := func(policy *Policy, data []byte) func(t *testing.T) {
		return func(t *testing.T) {
			mustTransformToStorage(t, WithCompression(newRecording(), policy, testGR), data)
		}
	}
	read := func(policy *Policy, stored []byte, wantErr bool) func(t *testing.T) {
		return func(t *testing.T) {
			_, _, err := WithCompression(newRecording(), policy, testGR).TransformFromStorage(context.Background(), stored, testDataCtx)
			if (err != nil) != wantErr {
				t.Fatalf("err = %v, wantErr %v", err, wantErr)
			}
		}
	}
	damaged := mustDeflate(t, compressible)
	damaged[headerSize+3] ^= 0x01

	for _, tc := range []struct {
		desc    string
		run     func(t *testing.T)
		metrics []string
		want    string
	}{
		{
			desc:    "compressed write",
			run:     write(policyFor(t, "configmaps"), compressible),
			metrics: []string{"apiserver_storage_compression_operations_total"},
			want:    operationsHeader + `apiserver_storage_compression_operations_total{operation="to_storage",outcome="deflate",resource="configmaps"} 1` + "\n",
		},
		{
			desc:    "incompressible write is stored",
			run:     write(policyFor(t, "configmaps"), randomBytes(4096)),
			metrics: []string{"apiserver_storage_compression_operations_total"},
			want:    operationsHeader + `apiserver_storage_compression_operations_total{operation="to_storage",outcome="stored",resource="configmaps"} 1` + "\n",
		},
		{
			desc:    "write below the threshold",
			run:     write(policyFor(t, "configmaps"), []byte("{}")),
			metrics: []string{"apiserver_storage_compression_operations_total"},
			want:    operationsHeader + `apiserver_storage_compression_operations_total{operation="to_storage",outcome="unframed_below_threshold",resource="configmaps"} 1` + "\n",
		},
		{
			desc:    "write above the ceiling",
			run:     write(policyFor(t, "configmaps"), bytes.Repeat([]byte("a"), MaxPlaintextBytes+1)),
			metrics: []string{"apiserver_storage_compression_operations_total"},
			want:    operationsHeader + `apiserver_storage_compression_operations_total{operation="to_storage",outcome="unframed_above_max",resource="configmaps"} 1` + "\n",
		},
		{
			desc:    "a framed read is counted with the policy off",
			run:     read(nil, mustDeflate(t, compressible), false),
			metrics: []string{"apiserver_storage_compression_operations_total", "apiserver_storage_compression_rewrites_needed_total"},
			want: operationsHeader + `apiserver_storage_compression_operations_total{operation="from_storage",outcome="deflate",resource="configmaps"} 1` + "\n" +
				rewritesHeader + `apiserver_storage_compression_rewrites_needed_total{direction="to_unframed",resource="configmaps"} 1` + "\n",
		},
		{
			desc:    "an unframed read with the policy on drives migration",
			run:     read(policyFor(t, "configmaps"), compressible, false),
			metrics: []string{"apiserver_storage_compression_operations_total", "apiserver_storage_compression_rewrites_needed_total"},
			want: operationsHeader + `apiserver_storage_compression_operations_total{operation="from_storage",outcome="unframed_legacy",resource="configmaps"} 1` + "\n" +
				rewritesHeader + `apiserver_storage_compression_rewrites_needed_total{direction="to_framed",resource="configmaps"} 1` + "\n",
		},
		{
			desc:    "a checksum mismatch is counted as a format error",
			run:     read(nil, damaged, true),
			metrics: []string{"apiserver_storage_compression_format_errors_total"},
			want:    formatErrorsHeader + `apiserver_storage_compression_format_errors_total{resource="configmaps"} 1` + "\n",
		},
		{
			desc:    "a malformed frame is counted as a format error",
			run:     read(nil, declaredDeflate(16, nil), true),
			metrics: []string{"apiserver_storage_compression_format_errors_total"},
			want:    formatErrorsHeader + `apiserver_storage_compression_format_errors_total{resource="configmaps"} 1` + "\n",
		},
		{
			desc:    "an unsupported algorithm is counted separately",
			run:     read(nil, rawFrame(0xfe, nil), true),
			metrics: []string{"apiserver_storage_compression_unsupported_algorithm_total", "apiserver_storage_compression_format_errors_total"},
			want:    unsupportedHeader + `apiserver_storage_compression_unsupported_algorithm_total{resource="configmaps"} 1` + "\n",
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			resetMetrics()
			defer resetMetrics()
			tc.run(t)
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.want), tc.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func histogramStats(t *testing.T, name string, labels map[string]string) (count uint64, sum float64) {
	t.Helper()
	vec, err := testutil.GetHistogramVecFromGatherer(legacyregistry.DefaultGatherer, name, labels)
	if err != nil {
		return 0, 0
	}
	return vec.GetAggregatedSampleCount(), vec.GetAggregatedSampleSum()
}

const (
	durationName = "apiserver_storage_compression_duration_seconds"
	waitName     = "apiserver_storage_compression_inflation_wait_seconds"
)

func TestDurationAndInflationWaitObserved(t *testing.T) {
	RegisterMetrics()
	resetMetrics()
	defer resetMetrics()

	tr := WithCompression(newRecording(), policyFor(t, "configmaps"), testGR)
	stored := mustTransformToStorage(t, tr, compressible)
	if _, _, err := tr.TransformFromStorage(context.Background(), stored, testDataCtx); err != nil {
		t.Fatal(err)
	}
	for _, op := range []string{toStorage, fromStorage} {
		if n, _ := histogramStats(t, durationName, map[string]string{"operation": op}); n != 1 {
			t.Errorf("duration samples for %s = %d, want 1", op, n)
		}
	}
	if n, _ := histogramStats(t, waitName, nil); n != 1 {
		t.Errorf("inflation wait samples = %d, want 1", n)
	}
}

// TestReadDurationExcludesTheWait keeps the two histograms apart, so an operator
// can tell queueing on the inflation cap from slow inflation.
func TestReadDurationExcludesTheWait(t *testing.T) {
	RegisterMetrics()
	resetMetrics()
	defer resetMetrics()

	// Build everything first, so little runs between taking the slots and the read.
	tr := WithCompression(newRecording(), nil, testGR)
	frame := mustDeflate(t, compressible)

	// Take every inflation slot, so the read below has to wait for one.
	held := 0
	for saturated := false; !saturated; {
		select {
		case inflateBudget <- struct{}{}:
			held++
		default:
			saturated = true
		}
	}
	// Give the slots back after hold; until then the read is stuck waiting.
	const hold = 200 * time.Millisecond
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		time.Sleep(hold)
		for i := 0; i < held; i++ {
			<-inflateBudget
		}
	}()
	// Don't return while slots are still taken, or the next test would wait too.
	defer func() { <-drained }()

	readStart := time.Now()
	if _, _, err := tr.TransformFromStorage(context.Background(), frame, testDataCtx); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(readStart).Seconds()

	// The read waits for a slot and then inflates. Each histogram times one of those
	// steps, so together they can't exceed how long the whole read took, however
	// slow the machine. If the duration also timed the wait, the sum would go over.
	waits, wait := histogramStats(t, waitName, nil)
	reads, d := histogramStats(t, durationName, map[string]string{"operation": fromStorage})
	if waits != 1 || reads != 1 {
		t.Fatalf("got %d wait and %d read duration samples, want 1 each", waits, reads)
	}
	if wait+d > elapsed {
		t.Errorf("wait %vs plus read duration %vs exceeds the read's %vs, so the duration includes the wait", wait, d, elapsed)
	}
}

// TestCancelledReadIsNotAFrameError checks that a read which gave up waiting for a
// slot returns context.Canceled, isn't flagged as corruption and isn't counted as a
// frame error. Otherwise a cancelled request would look like damaged bytes in etcd.
func TestCancelledReadIsNotAFrameError(t *testing.T) {
	RegisterMetrics()
	resetMetrics()
	defer resetMetrics()

	held := 0
	for saturated := false; !saturated; {
		select {
		case inflateBudget <- struct{}{}:
			held++
		default:
			saturated = true
		}
	}
	defer func() {
		for i := 0; i < held; i++ {
			<-inflateBudget
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := WithCompression(newRecording(), nil, testGR).TransformFromStorage(ctx, mustDeflate(t, compressible), testDataCtx)
	if !errors.Is(err, context.Canceled) || value.IsDataCorruption(err) {
		t.Fatalf("err = %v: want context.Canceled, not reported as corruption", err)
	}
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(""),
		"apiserver_storage_compression_format_errors_total", "apiserver_storage_compression_unsupported_algorithm_total"); err != nil {
		t.Fatalf("a cancelled read was counted as a frame error: %v", err)
	}
}

func TestNoMetricsWhenInert(t *testing.T) {
	RegisterMetrics()
	resetMetrics()
	defer resetMetrics()

	tr := WithCompression(newRecording(), nil, testGR)
	stored := mustTransformToStorage(t, tr, compressible)
	if _, _, err := tr.TransformFromStorage(context.Background(), stored, testDataCtx); err != nil {
		t.Fatal(err)
	}
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(""),
		"apiserver_storage_compression_operations_total",
		"apiserver_storage_compression_duration_seconds",
		"apiserver_storage_compression_rewrites_needed_total",
		"apiserver_storage_compression_format_errors_total",
		"apiserver_storage_compression_unsupported_algorithm_total",
		"apiserver_storage_compression_inflation_wait_seconds",
	); err != nil {
		t.Errorf("an inert policy produced metric samples: %v", err)
	}
}

// TestFormatErrorMetricCarriesResourceButNotReason checks the format error
// counter has only a resource label, so /metrics doesn't tell someone tampering
// with etcd which check their bytes failed. It also checks the counter exports
// nothing until the first error, so a healthy cluster shows no series.
func TestFormatErrorMetricCarriesResourceButNotReason(t *testing.T) {
	RegisterMetrics()
	formatErrorsTotal.Reset()
	defer formatErrorsTotal.Reset()

	const name = "apiserver_storage_compression_format_errors_total"
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(""), name); err != nil {
		t.Fatalf("no series may exist before an error is counted: %v", err)
	}
	recordFormatError("configmaps")
	recordFormatError("secrets")
	want := formatErrorsHeader +
		`apiserver_storage_compression_format_errors_total{resource="configmaps"} 1` + "\n" +
		`apiserver_storage_compression_format_errors_total{resource="secrets"} 1` + "\n"
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), name); err != nil {
		t.Fatal(err)
	}
}

// TestExportsOnlyTheListedMetrics fails if this package exports a metric outside
// its list, so a ratio or byte-count metric can't slip in. It also fails if a
// listed metric is missing. The list is kep.yaml's metrics minus
// last_config_info, which compressionconfig/metrics exports.
func TestExportsOnlyTheListedMetrics(t *testing.T) {
	RegisterMetrics()
	resetMetrics()
	defer resetMetrics()

	on := WithCompression(newRecording(), policyFor(t, "configmaps"), testGR)
	mustTransformToStorage(t, on, compressible)
	for _, stored := range [][]byte{mustDeflate(t, compressible), compressible} {
		if _, _, err := on.TransformFromStorage(context.Background(), stored, testDataCtx); err != nil {
			t.Fatal(err)
		}
	}
	for _, bad := range [][]byte{declaredDeflate(16, nil), rawFrame(0xfe, nil)} {
		if _, _, err := on.TransformFromStorage(context.Background(), bad, testDataCtx); err == nil {
			t.Fatal("expected a read error")
		}
	}

	want := map[string]bool{
		"apiserver_storage_compression_operations_total":            true,
		"apiserver_storage_compression_duration_seconds":            true,
		"apiserver_storage_compression_format_errors_total":         true,
		"apiserver_storage_compression_unsupported_algorithm_total": true,
		"apiserver_storage_compression_rewrites_needed_total":       true,
		"apiserver_storage_compression_inflation_wait_seconds":      true,
	}
	families, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, family := range families {
		name := family.GetName()
		if !strings.HasPrefix(name, "apiserver_storage_compression_") {
			continue
		}
		got[name] = true
		if !want[name] {
			t.Errorf("unexpected metric %s", name)
		}
	}
	for name := range want {
		if !got[name] {
			t.Errorf("metric %s was not exported", name)
		}
	}
}
