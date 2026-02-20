/*
Copyright 2025 The Kubernetes Authors.

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

package tracing

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"k8s.io/klog/v2"
	testingclock "k8s.io/utils/clock/testing"
)

type expectedEvent struct {
	name     string
	size     int64
	count    int64
	duration time.Duration
}

func TestTracedReaderLogs(t *testing.T) {
	for _, scenario := range tracedReaderScenarios {
		t.Run(scenario.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			defer klog.SetOutput(nil)

			fakeClock := testingclock.NewFakeClock(time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC))
			ctx := context.Background()

			scenario.testFunc(ctx, fakeClock)

			output := buf.String()
			t.Logf("Log output:\n%s", output)

			for _, expected := range scenario.expectedEvents {
				durationMs := expected.duration.Nanoseconds() / 1e6
				expectedDurationStr := fmt.Sprintf("%dms", durationMs)

				reportBytes := expected.size
				reportCount := expected.count

				if reportBytes > 0 && reportCount > 0 {
					assert.Contains(t, output, fmt.Sprintf(`%q size:%d,count:%d %s`, expected.name, reportBytes, reportCount, expectedDurationStr))
				} else {
					assert.Contains(t, output, fmt.Sprintf(`%q %s`, expected.name, expectedDurationStr))
				}

			}
		})
	}
}

func TestTracedReaderOTEL(t *testing.T) {
	for _, scenario := range tracedReaderScenarios {
		t.Run(scenario.name, func(t *testing.T) {
			fakeRecorder := tracetest.NewSpanRecorder()
			otelTracer := trace.NewTracerProvider(trace.WithSpanProcessor(fakeRecorder)).Tracer(instrumentationScope)

			fakeClock := testingclock.NewFakeClock(time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC))
			ctx := context.Background()

			parentName := "parent otel span"
			if scenario.expectedParentSpanName != "" {
				parentName = scenario.expectedParentSpanName
			}

			ctx, span := otelTracer.Start(ctx, parentName)
			scenario.testFunc(ctx, fakeClock)
			span.End()

			output := fakeRecorder.Ended()
			if len(output) != 2 {
				t.Fatalf("got %d; expected len(output) == 2", len(output))
			}
			child := output[0]
			if child.Name() != "Reading Response" && child.Name() != scenario.name {
				// Some scenarios might use different span names
			}

			if len(child.Events()) != len(scenario.expectedEvents) {
				t.Fatalf("got events %v; expected %d events", child.Events(), len(scenario.expectedEvents))
			}
			for i, event := range child.Events() {
				expected := scenario.expectedEvents[i]
				if event.Name != expected.name {
					t.Errorf("got event %v; expected event name %s", event, expected.name)
				}
				for _, attr := range event.Attributes {
					switch attr.Key {
					case "size":
						if attr.Value.AsInt64() != expected.size {
							t.Errorf("event %s: got size %d; expected %d", event.Name, attr.Value.AsInt64(), expected.size)
						}
					case "count":
						if attr.Value.AsInt64() != expected.count {
							t.Errorf("event %s: got count %d; expected %d", event.Name, attr.Value.AsInt64(), expected.count)
						}
					case "duration":
						if attr.Value.AsString() != expected.duration.String() {
							t.Errorf("event %s: got duration %s; expected %s", event.Name, attr.Value.AsString(), expected.duration.String())
						}
					}
				}
			}

			foundLayers := false
			var expectedLayers []string
			for _, e := range scenario.expectedEvents {
				expectedLayers = append(expectedLayers, e.name)
			}

			for _, attr := range child.Attributes() {
				if attr.Key == "writer.layers" {
					foundLayers = true
					assert.Equal(t, expectedLayers, attr.Value.AsStringSlice())
				}
			}
			if len(expectedLayers) > 0 && !foundLayers {
				t.Error("writer.layers attribute not found")
			}
		})
	}
}

type tracedReaderScenario struct {
	name                   string
	expectedParentSpanName string
	testFunc               func(ctx context.Context, fakeClock *testingclock.FakeClock)
	expectedEvents         []expectedEvent
}

var tracedReaderScenarios = []tracedReaderScenario{
	{
		name:                   "Base",
		expectedParentSpanName: "Base",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "Base")
			span.clock = fakeClock
			defer span.End(0)
		},
		expectedEvents: []expectedEvent{},
	},
	{
		name:                   "NoReadsWithNamedReader",
		expectedParentSpanName: "NoReadsWithNamedReader",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{ sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "NoReadsWithNamedReader")
			span.clock = fakeClock
			defer span.End(0)
			span.WrapReader(reader, "NamedReader")
			fakeClock.Step(10 * time.Millisecond)
		},
		expectedEvents: []expectedEvent{
			{name: "NamedReader", size: 0, count: 0, duration: 0},
		},
	},
	{
		name:                   "NoReadsWithSpan",
		expectedParentSpanName: "NoReadsWithSpan",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{ sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "NoReadsWithSpan")
			span.clock = fakeClock
			defer span.End(0)
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			fakeClock.Step(10 * time.Millisecond)
		},
		expectedEvents: []expectedEvent{
			{name: "Reader", size: 0, count: 0, duration: 0},
			{name: "Deserialize", size: 0, count: 0, duration: 10*time.Millisecond},
		},
	},
	{
		name:                   "NoReadsWithSpanAndNamedReader",
		expectedParentSpanName: "NoReadsWithSpanAndNamedReader",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{ sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "NoReadsWithSpanAndNamedReader")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "NamedReader")
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			fakeClock.Step(10 * time.Millisecond)
		},
		expectedEvents: []expectedEvent{
			{name: "NamedReader", size: 0, count: 0, duration: 0},
			{name: "Deserialize", size: 0, count: 0, duration: 10*time.Millisecond},
		},
	},
	{
		name:                   "SingleReadWithNamedReader",
		expectedParentSpanName: "SingleReadWithNamedReader",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "SingleReadWithNamedReader")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "NamedReader")

			fakeClock.Step(5 * time.Millisecond)
			reader.Read(make([]byte, 5))
		},
		expectedEvents: []expectedEvent{
			{name: "NamedReader", size: 5, count: 1, duration: 7 * time.Millisecond},
		},
	},
	{
		name:                   "SingleReadWithSpan",
		expectedParentSpanName: "SingleReadWithSpan",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "SingleReadWithSpan")
			span.clock = fakeClock
			defer span.End(0)
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			fakeClock.Step(5 * time.Millisecond)
			reader.Read(make([]byte, 5))
		},
		expectedEvents: []expectedEvent{
			{name: "Reader", size: 5, count: 1, duration: 7 * time.Millisecond},
			{name: "Deserialize", size: 0, count: 0, duration: 5 * time.Millisecond},
		},
	},
	{
		name:                   "SingleReadWithSpanAndNamedReader",
		expectedParentSpanName: "SingleReadWithSpanAndNamedReader",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "SingleReadWithSpanAndNamedReader")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "NamedReader")
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			fakeClock.Step(5 * time.Millisecond)
			reader.Read(make([]byte, 5))
		},
		expectedEvents: []expectedEvent{
			{name: "NamedReader", size: 5, count: 1, duration: 7 * time.Millisecond},
			{name: "Deserialize", size: 0, count: 0, duration: 5 * time.Millisecond},
		},
	},
	{
		name:                   "SingleReadWithMultipleLayers",
		expectedParentSpanName: "SingleReadWithMultipleLayers",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "SingleReadWithMultipleLayers")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "NamedReader")
			reader = &mockReader{sleep: 3 * time.Millisecond, clock: fakeClock, next: reader}
			reader = span.WrapReader(reader, "NamedReader2")
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			fakeClock.Step(5 * time.Millisecond)
			reader.Read(make([]byte, 5))
		},
		expectedEvents: []expectedEvent{
			{name: "NamedReader", size: 5, count: 1, duration: 7 * time.Millisecond},
			{name: "NamedReader2", size: 5, count: 1, duration: 3 * time.Millisecond},
			{name: "Deserialize", size: 0, count: 0, duration: 5 * time.Millisecond},
		},
	},
	{
		name:                   "MultipleReads",
		expectedParentSpanName: "MultipleReads",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "MultipleReads")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "NamedReader")
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			for i	:= 0; i < 5; i++ {
				fakeClock.Step(5 * time.Millisecond)
				reader.Read(make([]byte, 5))
			}
		},
		expectedEvents: []expectedEvent{
			{name: "NamedReader", size: 25, count: 5, duration: 35 * time.Millisecond},
			{name: "Deserialize", size: 0, count: 0, duration: 25 * time.Millisecond},
		},
	},
	{
		name:                   "MultipleReadsWithMultipleLayers",
		expectedParentSpanName: "MultipleReadsWithMultipleLayers",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "MultipleReadsWithMultipleLayers")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "NamedReader")
			reader = &mockReader{sleep: 3 * time.Millisecond, clock: fakeClock, next: reader}
			reader = span.WrapReader(reader, "NamedReader2")
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			for i	:= 0; i < 5; i++ {
				fakeClock.Step(5 * time.Millisecond)
				reader.Read(make([]byte, 5))
			}
		},
		expectedEvents: []expectedEvent{
			{name: "NamedReader", size: 25, count: 5, duration: 35 * time.Millisecond},
			{name: "NamedReader2", size: 25, count: 5, duration: 15 * time.Millisecond},
			{name: "Deserialize", size: 0, count: 0, duration: 25 * time.Millisecond},
		},
	},
	{
		name:                   "PrefixReader",
		expectedParentSpanName: "PrefixReader",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "PrefixReader")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "Raw")
			reader = &prefixReader{mockReader: mockReader{next: reader, sleep: 3 * time.Millisecond, clock: fakeClock}, prefix: []byte("HEAD")}
			reader = span.WrapReader(reader, "Prefix")
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			for i := 0; i < 5; i++ {
				fakeClock.Step(5 * time.Millisecond)
				reader.Read(make([]byte, 5))
			}
		},
		expectedEvents: []expectedEvent{
			{name: "Raw", size: 5, count: 5, duration: 35 * time.Millisecond},
			{name: "Prefix", size: 25, count: 5, duration: 15* time.Millisecond},
			{name: "Deserialize", duration: 25 * time.Millisecond},
		},
	},
	{
		name:                   "ChunkingReader",
		expectedParentSpanName: "ChunkingReader",
		testFunc: func(ctx context.Context, fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "ChunkingReader")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "Raw")
			reader = &chunkingReader{chunkSize: 10, mockReader: mockReader{next: reader, sleep: 3 * time.Millisecond, clock: fakeClock}}
			reader = span.WrapReader(reader, "Chunking")
			reader, s := span.WithReader("Deserialize", reader)
			defer s.Done()
			for i := 0; i < 5; i++ {
				fakeClock.Step(5 * time.Millisecond)
				reader.Read(make([]byte, 5))
			}
		},
		expectedEvents: []expectedEvent{
			{name: "Raw", size: 30, count: 3, duration: 21 * time.Millisecond},
			{name: "Chunking", size: 25, count: 5, duration: 15* time.Millisecond},
			{name: "Deserialize", duration: 25 * time.Millisecond},
		},
	},
	{
		name:                   "DeepNesting",
		expectedParentSpanName: "DeepNesting",
		testFunc: func(ctx context.Context,fakeClock *testingclock.FakeClock) {
			var reader io.Reader = &mockReader{ sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx, span := Start(ctx, "DeepNesting")
			span.clock = fakeClock
			defer span.End(0)
			reader = span.WrapReader(reader, "L1")
			reader = span.WrapReader(reader, "L2")
			reader = span.WrapReader(reader, "L3")

			fakeClock.Step(10 * time.Millisecond)
			reader.Read(make([]byte, 4))
		},
		expectedEvents: []expectedEvent{
			{name: "L1", size: 4, count: 1, duration: 7 * time.Millisecond},
			{name: "L2", size: 4, count: 1, duration: 0},
			{name: "L3", size: 4, count: 1, duration: 0},
		},
	},
}

type mockReader struct {
	sleep time.Duration
	clock *testingclock.FakeClock
	next  io.Reader
}

func (m *mockReader) Read(p []byte) (int, error) {
	m.clock.Step(m.sleep)
	if m.next != nil {
		return m.next.Read(p)
	}
	// Return data
	n := copy(p, []byte("1234567890"))
	if len(p) == 0 {
		return len(p), nil
	}
	// Advance data pointer? No, mockReader is simple.
	// But for Response reader we need to return different data or just data.
	// If we don't advance, we return same data.
	// Chunking reader reads 10 bytes.
	// If we return 50 bytes, we need to advance?
	// If we don't advance, we return first 10 bytes 5 times.
	// That's fine for the test, as long as we return bytes.
	return n, nil
}

func (m *mockReader) Unwrap() io.Reader {
	return m.next
}

type prefixReader struct {
	mockReader
	prefix []byte
}

func (r *prefixReader) Read(p []byte) (int, error) {
	n := copy(p, r.prefix)
	k, err := r.mockReader.Read(p[n:])
	return n + k, err
}

type chunkingReader struct {
	mockReader
	chunkSize int
	buffer    []byte
}

func (r *chunkingReader) Read(p []byte) (n int, err error) {
	r.clock.Step(r.sleep)
	for n < len(p) {
		if len(r.buffer) == 0 {
			buf := make([]byte, r.chunkSize)
			nr, er := r.next.Read(buf)
			if nr > 0 {
				r.buffer = buf[:nr]
			}
			if er != nil {
				if n == 0 && nr == 0 {
					return 0, er
				}
				err = er
				if nr == 0 {
					return n, err
				}
			}
		}

		copied := copy(p[n:], r.buffer)
		n += copied
		r.buffer = r.buffer[copied:]

		if err != nil && len(r.buffer) == 0 {
			return n, err
		}
	}
	return n, nil
}

func TestTracedWriterLogs(t *testing.T) {
	for _, scenario := range tracedWriterScenarios {
		t.Run(scenario.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			defer klog.SetOutput(nil)

			fakeClock := testingclock.NewFakeClock(time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC))
			writer := &mockWriter{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx := context.Background()

			scenario.testFunc(ctx, writer, fakeClock)

			output := buf.String()
			t.Logf("Log output:\n%s", output)
			for _, expected := range scenario.expectedEvents {
				durationMs := expected.duration.Nanoseconds() / 1e6
				expectedDurationStr := fmt.Sprintf("%dms", durationMs)
				if expected.size > 0 && expected.count > 0 {
					assert.Contains(t, output, fmt.Sprintf(`%q size:%d,count:%d %s`, expected.name, expected.size, expected.count, expectedDurationStr))
				} else {
					assert.Contains(t, output, fmt.Sprintf(`%q %s`, expected.name, expectedDurationStr))
				}
			}
		})
	}
}

func TestTracedWriterOTEL(t *testing.T) {
	for _, scenario := range tracedWriterScenarios {
		t.Run(scenario.name, func(t *testing.T) {
			fakeRecorder := tracetest.NewSpanRecorder()
			otelTracer := trace.NewTracerProvider(trace.WithSpanProcessor(fakeRecorder)).Tracer(instrumentationScope)

			fakeClock := testingclock.NewFakeClock(time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC))
			writer := &mockWriter{sleep: 7 * time.Millisecond, clock: fakeClock}
			ctx := context.Background()

			parentName := "parent otel span"
			if scenario.expectedParentSpanName != "" {
				parentName = scenario.expectedParentSpanName
			}

			ctx, span := otelTracer.Start(ctx, parentName)
			scenario.testFunc(ctx, writer, fakeClock)
			span.End()

			output := fakeRecorder.Ended()
			if len(output) != 2 {
				t.Fatalf("got %d; expected len(output) == 2", len(output))
			}
			child := output[0]
			if child.Name() != "Writing Response" && child.Name() != scenario.name {
				// Some scenarios might use different span names, but for now let's assume they use "Writing Response" or scenario name
			}

			if len(child.Events()) != len(scenario.expectedEvents) {
				t.Errorf("got events %v; expected %d events", child.Events(), len(scenario.expectedEvents))
			}
			for i, event := range child.Events() {
				expected := scenario.expectedEvents[i]
				if event.Name != expected.name {
					t.Errorf("got event %v; expected event name %s", event, expected.name)
				}
				for _, attr := range event.Attributes {
					switch attr.Key {
					case "size":
						if attr.Value.AsInt64() != expected.size {
							t.Errorf("event %s: got size %d; expected %d", event.Name, attr.Value.AsInt64(), expected.size)
						}
					case "count":
						if attr.Value.AsInt64() != expected.count {
							t.Errorf("event %s: got count %d; expected %d", event.Name, attr.Value.AsInt64(), expected.count)
						}
					case "duration":
						if attr.Value.AsString() != expected.duration.String() {
							t.Errorf("event %s: got duration %s; expected %s", event.Name, attr.Value.AsString(), expected.duration.String())
						}
					}
				}
			}

			foundLayers := false
			var expectedLayers []string
			for _, e := range scenario.expectedEvents {
				expectedLayers = append(expectedLayers, e.name)
			}

			for _, attr := range child.Attributes() {
				if attr.Key == "writer.layers" {
					foundLayers = true
					assert.Equal(t, expectedLayers, attr.Value.AsStringSlice())
				}
			}
			if len(expectedLayers) > 0 && !foundLayers {
				t.Error("writer.layers attribute not found")
			}
		})
	}
}

type tracedWriterScenario struct {
	name                   string
	expectedParentSpanName string
	testFunc               func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock)
	expectedEvents         []expectedEvent
}

var tracedWriterScenarios = []tracedWriterScenario{
	{
		name: "Response writer",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "Writing Response")
			span.clock = fakeClock // Inject fake clock for testing
			defer span.End(0)
			writer = span.WrapWriter(writer, "Raw")

			// Adds 4 bytes to 5 calls, each taking 5ms.
			writer = &prefixWriter{mockWriter: mockWriter{next: writer, sleep: 5 * time.Millisecond, clock: fakeClock}, prefix: []byte("HEAD")}
			writer = span.WrapWriter(writer, "Prefix")

			// Batches 11 calls each taking 3 miliseconds, totalling 44 bytes into 5 calls
			batchW := &chunkingWriter{chunkSize: 10, mockWriter: mockWriter{next: writer, sleep: 3 * time.Millisecond, clock: fakeClock}}
			writer = span.WrapWriter(batchW, "Chunking")
			defer batchW.Flush()

			// Called 11 times, each time taking 2ms and writing 4 bytes.
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			for i := 0; i < 11; i++ {
				fakeClock.Step(2 * time.Millisecond)
				writer.Write([]byte("data"))
			}
		},
		expectedEvents: []expectedEvent{
			{name: "Raw", size: 64, count: 5, duration: 35 * time.Millisecond},
			{name: "Prefix", size: 44, count: 5, duration: 25 * time.Millisecond},
			{name: "Chunking", size: 44, count: 11, duration: 21 * time.Millisecond}, // 33ms total - 12ms flush (bypassed) = 21ms
			{name: "Serialize", size: 0, count: 0, duration: 22 * time.Millisecond},
		},
	},
	{
		name:                   "NoWrites",
		expectedParentSpanName: "NoWrites",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "NoWrites")
			span.clock = fakeClock
			defer span.End(0)
			span.WrapWriter(writer, "Raw")
			fakeClock.Step(10 * time.Millisecond)
		},
		expectedEvents: []expectedEvent{
			{name: "Raw", size: 0, count: 0, duration: 0},
		},
	},
	{
		name:                   "SingleWrite",
		expectedParentSpanName: "SingleWrite",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "SingleWrite")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "Raw")

			fakeClock.Step(5 * time.Millisecond)
			writer.Write([]byte("hello"))
		},
		expectedEvents: []expectedEvent{
			{name: "Raw", size: 5, count: 1, duration: 7 * time.Millisecond},
		},
	},
	{
		name:                   "DeepNesting",
		expectedParentSpanName: "DeepNesting",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "DeepNesting")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "L1")
			writer = span.WrapWriter(writer, "L2")
			writer = span.WrapWriter(writer, "L3")

			fakeClock.Step(10 * time.Millisecond)
			writer.Write([]byte("data"))
		},
		expectedEvents: []expectedEvent{
			{name: "L1", size: 4, count: 1, duration: 7 * time.Millisecond},
			{name: "L2", size: 4, count: 1, duration: 0}, // Same bytes/count as L1, so 0. Exclusive time is 0.
			{name: "L3", size: 4, count: 1, duration: 0}, // Same bytes/count as L2, so 0. Exclusive time is 0.
		},
	},
	{
		name:                   "NoWritesWithNamedWriter",
		expectedParentSpanName: "NoWritesWithNamedWriter",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "NoWritesWithNamedWriter")
			span.clock = fakeClock
			defer span.End(0)
			span.WrapWriter(writer, "NamedWriter")
			fakeClock.Step(10 * time.Millisecond)
		},
		expectedEvents: []expectedEvent{
			{name: "NamedWriter", size: 0, count: 0, duration: 0},
		},
	},
	{
		name:                   "NoWritesWithSpan",
		expectedParentSpanName: "NoWritesWithSpan",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "NoWritesWithSpan")
			span.clock = fakeClock
			defer span.End(0)
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			fakeClock.Step(10 * time.Millisecond)
		},
		expectedEvents: []expectedEvent{
			{name: "Writer", size: 0, count: 0, duration: 0},
			{name: "Serialize", size: 0, count: 0, duration: 10 * time.Millisecond},
		},
	},
	{
		name:                   "NoWritesWithSpanAndNamedWriter",
		expectedParentSpanName: "NoWritesWithSpanAndNamedWriter",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "NoWritesWithSpanAndNamedWriter")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "NamedWriter")
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			fakeClock.Step(10 * time.Millisecond)
		},
		expectedEvents: []expectedEvent{
			{name: "NamedWriter", size: 0, count: 0, duration: 0},
			{name: "Serialize", size: 0, count: 0, duration: 10 * time.Millisecond},
		},
	},
	{
		name:                   "SingleWriteWithNamedWriter",
		expectedParentSpanName: "SingleWriteWithNamedWriter",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "SingleWriteWithNamedWriter")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "NamedWriter")

			fakeClock.Step(5 * time.Millisecond)
			writer.Write([]byte("hello"))
		},
		expectedEvents: []expectedEvent{
			{name: "NamedWriter", size: 5, count: 1, duration: 7 * time.Millisecond},
		},
	},
	{
		name:                   "SingleWriteWithSpan",
		expectedParentSpanName: "SingleWriteWithSpan",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "SingleWriteWithSpan")
			span.clock = fakeClock
			defer span.End(0)
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			fakeClock.Step(5 * time.Millisecond)
			writer.Write([]byte("hello"))
		},
		expectedEvents: []expectedEvent{
			{name: "Writer", size: 5, count: 1, duration: 7 * time.Millisecond},
			{name: "Serialize", size: 0, count: 0, duration: 5 * time.Millisecond},
		},
	},
	{
		name:                   "SingleWriteWithSpanAndNamedWriter",
		expectedParentSpanName: "SingleWriteWithSpanAndNamedWriter",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "SingleWriteWithSpanAndNamedWriter")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "NamedWriter")
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			fakeClock.Step(5 * time.Millisecond)
			writer.Write([]byte("hello"))
		},
		expectedEvents: []expectedEvent{
			{name: "NamedWriter", size: 5, count: 1, duration: 7 * time.Millisecond},
			{name: "Serialize", size: 0, count: 0, duration: 5 * time.Millisecond},
		},
	},
	{
		name:                   "SingleWriteWithMultipleLayers",
		expectedParentSpanName: "SingleWriteWithMultipleLayers",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "SingleWriteWithMultipleLayers")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "NamedWriter")
			writer = &mockWriter{sleep: 3 * time.Millisecond, clock: fakeClock, next: writer}
			writer = span.WrapWriter(writer, "NamedWriter2")
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			fakeClock.Step(5 * time.Millisecond)
			writer.Write([]byte("hello"))
		},
		expectedEvents: []expectedEvent{
			{name: "NamedWriter", size: 5, count: 1, duration: 7 * time.Millisecond},
			{name: "NamedWriter2", size: 5, count: 1, duration: 3 * time.Millisecond},
			{name: "Serialize", size: 0, count: 0, duration: 5 * time.Millisecond},
		},
	},
	{
		name:                   "MultipleWrites",
		expectedParentSpanName: "MultipleWrites",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "MultipleWrites")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "NamedWriter")
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			for i := 0; i < 5; i++ {
				fakeClock.Step(5 * time.Millisecond)
				writer.Write([]byte("hello"))
			}
		},
		expectedEvents: []expectedEvent{
			{name: "NamedWriter", size: 25, count: 5, duration: 35 * time.Millisecond},
			{name: "Serialize", size: 0, count: 0, duration: 25 * time.Millisecond},
		},
	},
	{
		name:                   "MultipleWritesWithMultipleLayers",
		expectedParentSpanName: "MultipleWritesWithMultipleLayers",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "MultipleWritesWithMultipleLayers")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "NamedWriter")
			writer = &mockWriter{sleep: 3 * time.Millisecond, clock: fakeClock, next: writer}
			writer = span.WrapWriter(writer, "NamedWriter2")
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			for i := 0; i < 5; i++ {
				fakeClock.Step(5 * time.Millisecond)
				writer.Write([]byte("hello"))
			}
		},
		expectedEvents: []expectedEvent{
			{name: "NamedWriter", size: 25, count: 5, duration: 35 * time.Millisecond},
			{name: "NamedWriter2", size: 25, count: 5, duration: 15 * time.Millisecond},
			{name: "Serialize", size: 0, count: 0, duration: 25 * time.Millisecond},
		},
	},
	{
		name:                   "PrefixWriter",
		expectedParentSpanName: "PrefixWriter",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "PrefixWriter")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "Raw")
			writer = &prefixWriter{mockWriter: mockWriter{next: writer, sleep: 3 * time.Millisecond, clock: fakeClock}, prefix: []byte("HEAD")}
			writer = span.WrapWriter(writer, "Prefix")
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			for i := 0; i < 5; i++ {
				fakeClock.Step(5 * time.Millisecond)
				writer.Write([]byte("hello"))
			}
		},
		expectedEvents: []expectedEvent{
			{name: "Raw", size: 45, count: 5, duration: 35 * time.Millisecond},
			{name: "Prefix", size: 25, count: 5, duration: 15 * time.Millisecond},
			{name: "Serialize", size: 0, count: 0, duration: 25 * time.Millisecond},
		},
	},
	{
		name:                   "ChunkingWriter",
		expectedParentSpanName: "ChunkingWriter",
		testFunc: func(ctx context.Context, writer io.Writer, fakeClock *testingclock.FakeClock) {
			ctx, span := Start(ctx, "ChunkingWriter")
			span.clock = fakeClock
			defer span.End(0)
			writer = span.WrapWriter(writer, "Raw")
			writer = &chunkingWriter{chunkSize: 10, mockWriter: mockWriter{next: writer, sleep: 3 * time.Millisecond, clock: fakeClock}}
			writer = span.WrapWriter(writer, "Chunking")
			writer, s := span.WithWriter("Serialize", writer)
			defer s.Done()
			for i := 0; i < 5; i++ {
				fakeClock.Step(5 * time.Millisecond)
				writer.Write([]byte("hello"))
			}
			// Note: chunkingWriter in this test setup doesn't flush automatically at end unless we call Flush.
			// The loop writes 5 * 5 = 25 bytes.
			// Chunk size 10.
			// 1. "hello" (5). Buf 5.
			// 2. "hello" (5). Buf 10. Flush 10. Buf 0.
			// 3. "hello" (5). Buf 5.
			// 4. "hello" (5). Buf 10. Flush 10. Buf 0.
			// 5. "hello" (5). Buf 5.
			// Total flushes: 2. Total bytes written to Raw: 20.
			// Raw duration: 2 * 7ms = 14ms.
			// Chunking duration: 5 * 3ms + 2 * 7ms = 15 + 14 = 29ms.
			// Chunking exclusive: 29 - 14 = 15ms.
			// Serialize duration: 5 * 5ms = 25ms.
		},
		expectedEvents: []expectedEvent{
			{name: "Raw", size: 20, count: 2, duration: 14 * time.Millisecond},
			{name: "Chunking", size: 25, count: 5, duration: 15 * time.Millisecond},
			{name: "Serialize", size: 0, count: 0, duration: 25 * time.Millisecond},
		},
	},
}

type mockWriter struct {
	sleep time.Duration
	clock *testingclock.FakeClock
	next  io.Writer
}

func (m *mockWriter) Write(p []byte) (int, error) {
	m.clock.Step(m.sleep)
	if m.next != nil {
		return m.next.Write(p)
	}
	return len(p), nil
}

func (m *mockWriter) Unwrap() io.Writer {
	return m.next
}

type prefixWriter struct {
	mockWriter
	prefix []byte
}

func (w *prefixWriter) Write(p []byte) (int, error) {
	_, err := w.mockWriter.Write(append(w.prefix, p...))
	return len(p), err
}

type chunkingWriter struct {
	mockWriter
	chunkSize int
	buffer    []byte
}

func (w *chunkingWriter) Write(p []byte) (int, error) {
	w.clock.Step(w.sleep)
	w.buffer = append(w.buffer, p...)
	if len(w.buffer) >= w.chunkSize {
		return len(p), w.Flush()
	}
	return len(p), nil
}

func (w *chunkingWriter) Flush() error {
	if len(w.buffer) == 0 {
		return nil
	}
	toWrite := w.chunkSize
	if len(w.buffer) < w.chunkSize {
		toWrite = len(w.buffer)
	}
	_, err := w.next.Write(w.buffer[:toWrite])
	if len(w.buffer) > toWrite {
		w.buffer = w.buffer[toWrite:]
	} else {
		w.buffer = w.buffer[:0]
	}
	return err
}
