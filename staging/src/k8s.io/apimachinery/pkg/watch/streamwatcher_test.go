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

package watch_test

import (
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	. "k8s.io/apimachinery/pkg/watch"
)

type fakeDecoder struct {
	items chan Event
	err   error
}

func (f fakeDecoder) Decode() (action EventType, object runtime.Object, err error) {
	if f.err != nil {
		return "", nil, f.err
	}
	item, open := <-f.items
	if !open {
		return action, nil, io.EOF
	}
	return item.Type, item.Object, nil
}

func (f fakeDecoder) Close() {
	if f.items != nil {
		close(f.items)
	}
}

type fakeReporter struct {
	err error
}

func (f *fakeReporter) AsObject(err error) runtime.Object {
	f.err = err
	return runtime.Unstructured(nil)
}

func TestStreamWatcher(t *testing.T) {
	table := []Event{
		{Type: Added, Object: testType("foo")},
	}

	fd := fakeDecoder{items: make(chan Event, 5)}
	//nolint:logcheck // Intentionally uses the old API.
	sw := NewStreamWatcher(fd, nil)

	for _, item := range table {
		fd.items <- item
		got, open := <-sw.ResultChan()
		if !open {
			t.Errorf("unexpected early close")
		}
		if e, a := item, got; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %v, got %v", e, a)
		}
	}

	sw.Stop()
	_, open := <-sw.ResultChan()
	if open {
		t.Errorf("Unexpected failure to close")
	}
}

func TestStreamWatcherError(t *testing.T) {
	fd := fakeDecoder{err: fmt.Errorf("test error")}
	fr := &fakeReporter{}
	//nolint:logcheck // Intentionally uses the old API.
	sw := NewStreamWatcher(fd, fr)
	evt, ok := <-sw.ResultChan()
	if !ok {
		t.Fatalf("unexpected close")
	}
	if evt.Type != Error || evt.Object != runtime.Unstructured(nil) {
		t.Fatalf("unexpected object: %#v", evt)
	}
	_, ok = <-sw.ResultChan()
	if ok {
		t.Fatalf("unexpected open channel")
	}

	sw.Stop()
	_, ok = <-sw.ResultChan()
	if ok {
		t.Fatalf("unexpected open channel")
	}
}

func TestStreamWatcherRace(t *testing.T) {
	fd := fakeDecoder{err: fmt.Errorf("test error")}
	fr := &fakeReporter{}
	//nolint:logcheck // Intentionally uses the old API.
	sw := NewStreamWatcher(fd, fr)
	time.Sleep(10 * time.Millisecond)
	sw.Stop()
	time.Sleep(10 * time.Millisecond)
	_, ok := <-sw.ResultChan()
	if ok {
		t.Fatalf("unexpected pending send")
	}
}

const dispatchLoggerMsgPrefix = "Fast watcher, slow consumer processing"

func newDispatchTestLogger(t *testing.T) klog.Logger {
	return ktesting.NewLogger(t, ktesting.NewConfig(ktesting.Verbosity(2), ktesting.BufferLogs(true)))
}

func dispatchWarnings(logger klog.Logger) []map[string]any {
	var warnings []map[string]any
	for _, entry := range logger.GetSink().(ktesting.Underlier).GetBuffer().Data() {
		if !strings.HasPrefix(entry.Message, dispatchLoggerMsgPrefix) {
			continue
		}
		values := map[string]any{"verbosity": entry.Verbosity}
		for i := 0; i+1 < len(entry.ParameterKVList); i += 2 {
			values[fmt.Sprint(entry.ParameterKVList[i])] = entry.ParameterKVList[i+1]
		}
		warnings = append(warnings, values)
	}
	return warnings
}

// waitForStreamWatcherExit drains the result channel, which receive() closes only after flushing its dispatch logger.
func waitForStreamWatcherExit(t *testing.T, sw *StreamWatcher) {
	t.Helper()

	timeout := time.After(30 * time.Second)
	for {
		select {
		case _, ok := <-sw.ResultChan():
			if !ok {
				return
			}
		case <-timeout:
			t.Fatal("Timed out waiting for the StreamWatcher to exit")
		}
	}
}

func TestStreamWatcher_DispatchLoggerReportsBlockedSends(t *testing.T) {
	// Longer than DefaultBlockLoggerSingleThreshold, so a single blocked send must be reported.
	const stall = 300 * time.Millisecond

	testCases := []struct {
		name             string
		decoder          fakeDecoder
		consume          bool
		wantResourceType string
	}{
		{
			name:             "slow consumer eventually receives the event",
			decoder:          fakeDecoder{items: make(chan Event, 1)},
			consume:          true,
			wantResourceType: "watch_test.testType",
		},
		{
			name:             "watcher stopped while the event send is blocked",
			decoder:          fakeDecoder{items: make(chan Event, 1)},
			wantResourceType: "watch_test.testType",
		},
		{
			name:    "watcher stopped while the decode error send is blocked",
			decoder: fakeDecoder{err: fmt.Errorf("corrupted stream")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.decoder.items != nil {
				tc.decoder.items <- Event{Type: Added, Object: testType("foo")}
			}
			logger := newDispatchTestLogger(t)
			sw := NewStreamWatcherWithLogger(logger, tc.decoder, &fakeReporter{})

			time.Sleep(stall)
			if tc.consume {
				if _, ok := <-sw.ResultChan(); !ok {
					t.Fatal("Unexpected early close")
				}
			}
			sw.Stop()
			waitForStreamWatcherExit(t, sw)

			warnings := dispatchWarnings(logger)
			if len(warnings) != 1 {
				t.Fatalf("Expected exactly one dispatch warning, got %d: %v", len(warnings), warnings)
			}
			if got := warnings[0]["verbosity"]; got != 2 {
				t.Errorf("Expected the warning at verbosity 2, got %v", got)
			}
			if got := warnings[0]["eventCount"]; got != 1 {
				t.Errorf("Expected event count 1, got %v", got)
			}
			if got, ok := warnings[0]["maxWait"].(time.Duration); !ok || got < stall {
				t.Errorf("Expected maxWait of at least %v, got %v", stall, warnings[0]["maxWait"])
			}
			if got := warnings[0]["resourceType"]; got != tc.wantResourceType {
				t.Errorf("Expected resourceType %q, got %v", tc.wantResourceType, got)
			}
		})
	}
}

func TestStreamWatcher_DispatchLoggerSkipsErrorEventsForResourceType(t *testing.T) {
	fd := fakeDecoder{items: make(chan Event, 2)}
	fd.items <- Event{Type: Error, Object: &metav1.Status{Status: metav1.StatusFailure, Code: 410}}
	fd.items <- Event{Type: Added, Object: testType("foo")}
	sw := NewStreamWatcherWithLogger(newDispatchTestLogger(t), fd, &fakeReporter{})

	<-sw.ResultChan()
	<-sw.ResultChan()
	if got := sw.DispatchLogger().ResourceType(); got != "watch_test.testType" {
		t.Errorf("Expected the first object after the Error event to set resourceType 'watch_test.testType', got %q", got)
	}

	sw.Stop()
	waitForStreamWatcherExit(t, sw)
}
