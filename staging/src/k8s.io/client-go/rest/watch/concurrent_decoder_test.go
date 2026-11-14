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

package watch

import (
	"fmt"
	"io"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/framer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
)

func podJSON(rv string) string {
	return fmt.Sprintf(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":%q,"resourceVersion":%q}}`, rv, rv)
}

func watchEventJSON(eventType watch.EventType, object string) string {
	return fmt.Sprintf(`{"type":%q,"object":%s}`, eventType, object)
}

func podEvent(eventType watch.EventType, rv string) watch.Event {
	return watch.Event{
		Type:   eventType,
		Object: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: rv, ResourceVersion: rv}},
	}
}

func TestConcurrentWatchEventDecoder(t *testing.T) {
	type result struct {
		event watch.Event
		err   string
	}
	tests := []struct {
		name  string
		input string
		want  []result
	}{
		{
			name: "empty stream",
			want: []result{
				{err: "EOF"},
			},
		},
		{
			name:  "single event",
			input: watchEventJSON(watch.Added, podJSON("1")),
			want: []result{
				{event: podEvent(watch.Added, "1")},
				{err: "EOF"},
			},
		},
		{
			name: "events in input order with more events than queue capacity",
			input: strings.Join([]string{
				watchEventJSON(watch.Added, podJSON("1")),
				watchEventJSON(watch.Modified, podJSON("2")),
				watchEventJSON(watch.Deleted, podJSON("1")),
				watchEventJSON(watch.Bookmark, podJSON("2")),
				watchEventJSON(watch.Error, `{"apiVersion":"v1","kind":"Status","status":"Failure","message":"expired"}`),
			}, ""),
			want: []result{
				{event: podEvent(watch.Added, "1")},
				{event: podEvent(watch.Modified, "2")},
				{event: podEvent(watch.Deleted, "1")},
				{event: podEvent(watch.Bookmark, "2")},
				{event: watch.Event{Type: watch.Error, Object: &metav1.Status{Status: "Failure", Message: "expired"}}},
				{err: "EOF"},
			},
		},
		{
			name:  "invalid event type",
			input: watchEventJSON("INVALID", podJSON("1")),
			want: []result{
				{err: "got invalid watch event type: INVALID"},
			},
		},
		{
			name:  "malformed frame",
			input: `{"type":!}`,
			want: []result{
				{err: "invalid character '!' looking for beginning of value"},
			},
		},
		{
			name:  "malformed frame followed by event",
			input: `{"type":!}` + watchEventJSON(watch.Added, podJSON("1")),
			want: []result{
				{err: "invalid character '!' looking for beginning of value"},
			},
		},
		{
			name:  "truncated frame",
			input: `{"type":"ADDED"`,
			want: []result{
				{err: "unexpected EOF"},
			},
		},
		{
			name:  "invalid object",
			input: watchEventJSON(watch.Added, `{}`),
			want: []result{
				{err: `unable to decode watch event: Object 'Kind' is missing in '{}'`},
			},
		},
		{
			name:  "event before truncated frame",
			input: watchEventJSON(watch.Added, podJSON("1")) + `{"type":"MODIFIED"`,
			want: []result{
				{event: podEvent(watch.Added, "1")},
				{err: "unexpected EOF"},
			},
		},
		{
			name:  "stream error followed by event",
			input: watchEventJSON(watch.Added, podJSON("1")) + `{"type":"INVALID"}` + watchEventJSON(watch.Modified, podJSON("2")),
			want: []result{
				{event: podEvent(watch.Added, "1")},
				{err: "got invalid watch event type: INVALID"},
				{event: podEvent(watch.Modified, "2")},
			},
		},
		{
			name:  "event before object error",
			input: watchEventJSON(watch.Added, podJSON("1")) + watchEventJSON(watch.Modified, `{}`),
			want: []result{
				{event: podEvent(watch.Added, "1")},
				{err: `unable to decode watch event: Object 'Kind' is missing in '{}'`},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := streaming.NewDecoder(framer.NewJSONFramedReader(io.NopCloser(strings.NewReader(tt.input))), getDecoder())
			decoder := newConcurrentWatchEventDecoder(stream, getDecoder(), 1, 1, 2)
			t.Cleanup(decoder.Close)

			for _, want := range tt.want {
				eventType, object, err := decoder.Decode()
				if want.err != "" {
					require.EqualError(t, err, want.err)
					require.Empty(t, eventType)
					require.Nil(t, object)
				} else {
					require.NoError(t, err)
					require.Equal(t, want.event, watch.Event{Type: eventType, Object: object})
				}
			}
		})
	}
}

type objectDecoderFunc func([]byte) (runtime.Object, error)

func (f objectDecoderFunc) Decode(data []byte, _ *schema.GroupVersionKind, _ runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	object, err := f(data)
	return object, nil, err
}

func TestConcurrentWatchEventDecoderPanic(t *testing.T) {
	reallyCrash := utilruntime.ReallyCrash
	utilruntime.ReallyCrash = false
	defer func() { utilruntime.ReallyCrash = reallyCrash }()

	synctest.Test(t, func(t *testing.T) {
		panickingDecoder := objectDecoderFunc(func([]byte) (runtime.Object, error) {
			panic("object decoder panic")
		})
		input := watchEventJSON(watch.Added, podJSON("1"))
		stream := streaming.NewDecoder(framer.NewJSONFramedReader(io.NopCloser(strings.NewReader(input))), getDecoder())
		decoder := NewConcurrentWatchEventDecoder(stream, panickingDecoder)

		eventType, object, err := decoder.Decode()
		require.EqualError(t, err, "panic decoding watch event object: object decoder panic")
		require.Empty(t, eventType)
		require.Nil(t, object)
		decoder.Close()
	})
}

func TestConcurrentWatchEventDecoderPreservesOrder(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		completed := make(chan string, 2)
		objectDecoder := getDecoder()
		delayedDecoder := objectDecoderFunc(func(data []byte) (runtime.Object, error) {
			if string(data) == podJSON("1") {
				// Delay the first decode in virtual time so the second can finish first.
				time.Sleep(time.Second)
			}
			object, err := runtime.Decode(objectDecoder, data)
			completed <- string(data)
			return object, err
		})
		input := watchEventJSON(watch.Added, podJSON("1")) + watchEventJSON(watch.Modified, podJSON("2"))
		stream := streaming.NewDecoder(framer.NewJSONFramedReader(io.NopCloser(strings.NewReader(input))), getDecoder())
		decoder := NewConcurrentWatchEventDecoder(stream, delayedDecoder)
		t.Cleanup(decoder.Close)

		for _, want := range []watch.Event{podEvent(watch.Added, "1"), podEvent(watch.Modified, "2")} {
			eventType, object, err := decoder.Decode()
			require.NoError(t, err)
			require.Equal(t, want, watch.Event{Type: eventType, Object: object})
		}
		require.Equal(t, podJSON("2"), <-completed)
		require.Equal(t, podJSON("1"), <-completed)
		_, _, err := decoder.Decode()
		require.ErrorIs(t, err, io.EOF)
	})
}

func TestConcurrentWatchEventDecoderShutdown(t *testing.T) {
	tests := []struct {
		name         string
		input        string
		wantQueueLen int
		wantEvents   []watch.Event
	}{
		{
			name: "close unblocks Decode waiting for an event",
		},
		{
			name:         "close completes with full queues and no consumer",
			input:        strings.Repeat(watchEventJSON(watch.Added, podJSON("1")), 10),
			wantQueueLen: 1,
			wantEvents:   []watch.Event{podEvent(watch.Added, "1")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				reader := io.NopCloser(strings.NewReader(tt.input))
				if tt.input == "" {
					var writer *io.PipeWriter
					reader, writer = io.Pipe()
					t.Cleanup(func() { require.NoError(t, writer.Close()) })
				}
				stream := streaming.NewDecoder(framer.NewJSONFramedReader(reader), getDecoder())
				decoder := newConcurrentWatchEventDecoder(stream, getDecoder(), 1, 1, 2)
				t.Cleanup(decoder.Close)

				// Let the pipeline block before checking the shutdown preconditions.
				synctest.Wait()
				require.Len(t, decoder.eventDecoder.rawResultsCh, tt.wantQueueLen)
				require.Len(t, decoder.pendingResultsCh, tt.wantQueueLen)
				require.Len(t, decoder.decodedResultsCh, tt.wantQueueLen)
				if tt.input == "" {
					go func() {
						// Close only after the test goroutine blocks in Decode.
						synctest.Wait()
						decoder.Close()
					}()
				} else {
					// Shutdown must complete without a consumer draining the queues.
					decoder.Close()
				}

				eventType, object, err := decoder.Decode()
				// Shutdown may return EOF before all buffered events are consumed.
				for _, want := range tt.wantEvents {
					if err != nil {
						break
					}
					require.Equal(t, want, watch.Event{Type: eventType, Object: object})
					eventType, object, err = decoder.Decode()
				}
				require.ErrorIs(t, err, io.EOF)
				require.Empty(t, eventType)
				require.Nil(t, object)
			})
		})
	}
}
