/*
Copyright 2014 The Kubernetes Authors.

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

package watch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimejson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/framer"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

// getDecoder mimics how k8s.io/client-go/rest.createSerializers creates a decoder
func getDecoder() runtime.Decoder {
	jsonSerializer := runtimejson.NewSerializerWithOptions(runtimejson.DefaultMetaFactory, scheme.Scheme, scheme.Scheme, runtimejson.SerializerOptions{})
	directCodecFactory := scheme.Codecs.WithoutConversion()
	return directCodecFactory.DecoderToVersion(jsonSerializer, v1.SchemeGroupVersion)
}

func TestDecoder(t *testing.T) {
	table := []watch.EventType{watch.Added, watch.Deleted, watch.Modified, watch.Error, watch.Bookmark}

	for _, eventType := range table {
		out, in := io.Pipe()

		decoder := NewDecoder(streaming.NewDecoder(out, getDecoder()), getDecoder())
		expect := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
		encoder := json.NewEncoder(in)
		eType := eventType
		errc := make(chan error)

		go func() {
			data, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expect)
			if err != nil {
				errc <- fmt.Errorf("Unexpected error %v", err)
				return
			}
			event := metav1.WatchEvent{
				Type:   string(eType),
				Object: runtime.RawExtension{Raw: json.RawMessage(data)},
			}
			if err := encoder.Encode(&event); err != nil {
				t.Errorf("Unexpected error %v", err)
			}
			in.Close()
		}()

		done := make(chan struct{})
		go func() {
			action, got, err := decoder.Decode()
			if err != nil {
				errc <- fmt.Errorf("Unexpected error %v", err)
				return
			}
			if e, a := eType, action; e != a {
				t.Errorf("Expected %v, got %v", e, a)
			}
			if e, a := expect, got; !apiequality.Semantic.DeepDerivative(e, a) {
				t.Errorf("Expected %v, got %v", e, a)
			}
			t.Logf("Exited read")
			close(done)
		}()
		select {
		case err := <-errc:
			t.Fatal(err)
		case <-done:
		}

		done = make(chan struct{})
		go func() {
			_, _, err := decoder.Decode()
			if err == nil {
				t.Errorf("Unexpected nil error")
			}
			close(done)
		}()
		<-done

		decoder.Close()
	}
}

func TestDecoder_SourceClose(t *testing.T) {
	out, in := io.Pipe()
	decoder := NewDecoder(streaming.NewDecoder(out, getDecoder()), getDecoder())

	done := make(chan struct{})

	go func() {
		_, _, err := decoder.Decode()
		if err == nil {
			t.Errorf("Unexpected nil error")
		}
		close(done)
	}()

	in.Close()

	select {
	case <-done:
		break
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Timeout")
	}
}

const decodeLoggerTestMsg = "slow test decode"

var decodeLoggerTestEpoch = time.Date(2026, time.January, 1, 0, 0, 0, 0, time.UTC)

// clockSteppingDecoder advances a fake clock inside Decode to simulate a deterministic object decoding cost.
type clockSteppingDecoder struct {
	runtime.Decoder
	clock *testingclock.FakeClock
	cost  time.Duration
}

func (d clockSteppingDecoder) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	d.clock.Step(d.cost)
	return d.Decoder.Decode(data, defaults, into)
}

// clockSteppingReader advances a fake clock inside Read to simulate network latency, which must not count as decoding.
type clockSteppingReader struct {
	io.ReadCloser
	clock *testingclock.FakeClock
	cost  time.Duration
}

func (r clockSteppingReader) Read(p []byte) (int, error) {
	r.clock.Step(r.cost)
	return r.ReadCloser.Read(p)
}

func encodeWatchEvents(t *testing.T, events ...metav1.WatchEvent) io.ReadCloser {
	t.Helper()

	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	for i := range events {
		if err := encoder.Encode(&events[i]); err != nil {
			t.Fatalf("Unexpected error %v", err)
		}
	}
	return framer.NewJSONFramedReader(io.NopCloser(&buf))
}

func podWatchEvent(t *testing.T, name string) metav1.WatchEvent {
	t.Helper()

	data, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}})
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	return metav1.WatchEvent{Type: string(watch.Added), Object: runtime.RawExtension{Raw: json.RawMessage(data)}}
}

// newTestDecoder builds a Decoder whose network reads and object decodes advance a fake clock by fixed costs,
// and swaps in a decode logger with the production thresholds, the fake clock, and a buffered logger.
func newTestDecoder(t *testing.T, stream io.ReadCloser, networkCost, decodeCost time.Duration) (*Decoder, klog.Logger) {
	t.Helper()

	fakeClock := testingclock.NewFakeClock(decodeLoggerTestEpoch)
	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.Verbosity(2), ktesting.BufferLogs(true)))
	reader := clockSteppingReader{ReadCloser: stream, clock: fakeClock, cost: networkCost}
	d := NewDecoder(streaming.NewDecoder(reader, getDecoder()), clockSteppingDecoder{Decoder: getDecoder(), clock: fakeClock, cost: decodeCost})
	d.decodeLogger = watch.NewBlockLoggerWithOptions(decodeLoggerTestMsg, watch.BlockLoggerOptions{
		Threshold:       watch.DefaultDecodeLoggerThreshold,
		SingleThreshold: watch.DefaultBlockLoggerSingleThreshold,
		Logger:          &logger,
		Clock:           fakeClock,
	})
	return d, logger
}

func decodeWarnings(t *testing.T, logger klog.Logger) []map[string]any {
	t.Helper()

	var warnings []map[string]any
	for _, entry := range logger.GetSink().(ktesting.Underlier).GetBuffer().Data() {
		if entry.Message != decodeLoggerTestMsg {
			t.Fatalf("Unexpected log message %q", entry.Message)
		}
		values := map[string]any{}
		for i := 0; i+1 < len(entry.ParameterKVList); i += 2 {
			values[fmt.Sprint(entry.ParameterKVList[i])] = entry.ParameterKVList[i+1]
		}
		warnings = append(warnings, values)
	}
	return warnings
}

func TestDecoder_DecodeLoggerMeasuresOnlyObjectDecoding(t *testing.T) {
	d, logger := newTestDecoder(t, encodeWatchEvents(t, podWatchEvent(t, "foo"), podWatchEvent(t, "bar")), time.Second, 30*time.Millisecond)
	for i := 0; i < 2; i++ {
		if _, _, err := d.Decode(); err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
	}

	// Network reads advanced the clock by 2s in total, but only embedded object decoding counts.
	if got := d.decodeLogger.TimeWaiting(); got != 60*time.Millisecond {
		t.Errorf("Expected 60ms of decoding time, got %v", got)
	}
	if got := d.decodeLogger.EventCount(); got != 2 {
		t.Errorf("Expected decodeLogger event count 2, got %d", got)
	}
	if got := d.decodeLogger.ResourceType(); got != "*v1.Pod" {
		t.Errorf("Expected decodeLogger resourceType '*v1.Pod', got %q", got)
	}
	if warnings := decodeWarnings(t, logger); len(warnings) != 0 {
		t.Errorf("Expected no warnings, got %v", warnings)
	}
}

func TestDecoder_DecodeLoggerFlushesOnClose(t *testing.T) {
	testCases := []struct {
		name       string
		decodeCost time.Duration
		wantWarn   bool
	}{
		{name: "slow decode is reported when the stream closes", decodeCost: 300 * time.Millisecond, wantWarn: true},
		{name: "fast decode stays silent", decodeCost: time.Millisecond},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			d, logger := newTestDecoder(t, encodeWatchEvents(t, podWatchEvent(t, "foo")), 0, tc.decodeCost)
			if _, _, err := d.Decode(); err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if warnings := decodeWarnings(t, logger); len(warnings) != 0 {
				t.Fatalf("Expected no warning before Close, got %v", warnings)
			}

			d.Close()

			warnings := decodeWarnings(t, logger)
			if !tc.wantWarn {
				if len(warnings) != 0 {
					t.Fatalf("Expected no warning, got %v", warnings)
				}
				return
			}
			want := []map[string]any{{
				"resourceType": "*v1.Pod",
				"eventCount":   1,
				"timeWaiting":  300 * time.Millisecond,
				"avgWait":      300 * time.Millisecond,
				"maxWait":      300 * time.Millisecond,
				"window":       300 * time.Millisecond,
			}}
			if !reflect.DeepEqual(want, warnings) {
				t.Errorf("Expected warnings %v, got %v", want, warnings)
			}
		})
	}
}

func TestDecoder_DecodeLoggerSkipsErrorEventsForResourceType(t *testing.T) {
	status, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &metav1.Status{
		Status: metav1.StatusFailure, Code: http.StatusGone, Reason: metav1.StatusReasonExpired,
	})
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	stream := encodeWatchEvents(t,
		metav1.WatchEvent{Type: string(watch.Error), Object: runtime.RawExtension{Raw: json.RawMessage(status)}},
		podWatchEvent(t, "foo"),
	)
	d, _ := newTestDecoder(t, stream, 0, time.Millisecond)

	action, obj, err := d.Decode()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if _, ok := obj.(*metav1.Status); action != watch.Error || !ok {
		t.Fatalf("Expected an Error event with *v1.Status, got %v with %T", action, obj)
	}
	if got := d.decodeLogger.ResourceType(); got != "" {
		t.Errorf("Expected an Error event to leave the resource type unset, got %q", got)
	}

	if _, _, err := d.Decode(); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if got := d.decodeLogger.ResourceType(); got != "*v1.Pod" {
		t.Errorf("Expected decodeLogger resourceType '*v1.Pod', got %q", got)
	}
}

func TestDecoder_DecodeLoggerRecordsFailedDecodes(t *testing.T) {
	stream := encodeWatchEvents(t, metav1.WatchEvent{
		Type:   string(watch.Added),
		Object: runtime.RawExtension{Raw: json.RawMessage(`{"apiVersion":"v1","kind":"NotRegistered"}`)},
	})
	d, _ := newTestDecoder(t, stream, 0, 300*time.Millisecond)

	if _, _, err := d.Decode(); err == nil {
		t.Fatal("Expected a decoding error")
	}
	if got := d.decodeLogger.EventCount(); got != 1 {
		t.Errorf("Expected the failed decode to be recorded, got event count %d", got)
	}
	if got := d.decodeLogger.TimeWaiting(); got != 300*time.Millisecond {
		t.Errorf("Expected 300ms of decoding time, got %v", got)
	}
	if got := d.decodeLogger.ResourceType(); got != "" {
		t.Errorf("Expected a failed decode to leave the resource type unset, got %q", got)
	}
}
