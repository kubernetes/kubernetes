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
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/scheme"
)

func TestBufferedEnvelopeDecoderReadCodecs(t *testing.T) {
	for _, info := range scheme.Codecs.SupportedMediaTypes() {
		if info.MediaType != runtime.ContentTypeJSON && info.MediaType != runtime.ContentTypeProtobuf {
			continue
		}
		t.Run(info.MediaType, func(t *testing.T) {
			var wire bytes.Buffer
			encoder := NewEncoder(streaming.NewEncoder(info.StreamSerializer.Framer.NewFrameWriter(&wire), info.StreamSerializer.Serializer), info.Serializer)
			types := []watch.EventType{watch.Added, watch.Modified, watch.Deleted, watch.Bookmark, watch.Error}
			objects := []runtime.Object{
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "added"}},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "modified"}},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "deleted"}},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "42"}},
				&metav1.Status{Message: "watch failed"},
			}
			for i := range types {
				kind := "Pod"
				if types[i] == watch.Error {
					kind = "Status"
				}
				objects[i].GetObjectKind().SetGroupVersionKind(v1.SchemeGroupVersion.WithKind(kind))
				if err := encoder.Encode(&watch.Event{Type: types[i], Object: objects[i]}); err != nil {
					t.Fatal(err)
				}
			}
			stream := streaming.NewDecoder(info.StreamSerializer.Framer.NewFrameReader(io.NopCloser(&wire)), info.StreamSerializer.Serializer)
			d, finished := newBufferedEnvelopeDecoderForTest(t, stream, len(types)+1)
			waitForBufferedEnvelopeDecoder(t, finished)
			objectDecoder := scheme.Codecs.WithoutConversion().DecoderToVersion(info.Serializer, v1.SchemeGroupVersion)
			processor := &concurrentOrderedEventProcessing{embeddedDecoder: objectDecoder}
			for i, action := range types {
				result := <-d.incomingEnvelopesCh
				if result.err != nil || result.envelope == nil || result.envelope.Type != string(action) {
					t.Fatalf("result %d: %#v", i, result)
				}
				processed := processor.processEnvelope(result)
				if processed.err != nil || processed.event == nil || processed.event.Type != action {
					t.Fatalf("processed result %d: %#v", i, processed)
				}
				obj := processed.event.Object
				obj.GetObjectKind().SetGroupVersionKind(schema.GroupVersionKind{})
				objects[i].GetObjectKind().SetGroupVersionKind(schema.GroupVersionKind{})
				if !reflect.DeepEqual(obj, objects[i]) {
					t.Fatalf("result %d: got %#v, want %#v", i, obj, objects[i])
				}
			}
			if result := <-d.incomingEnvelopesCh; !errors.Is(result.err, io.EOF) {
				t.Fatalf("terminal result: %#v, want EOF", result)
			}
			if _, ok := <-d.incomingEnvelopesCh; ok {
				t.Fatal("queue remained open after EOF")
			}
		})
	}
}

func TestConcurrentOrderedEventProcessingProcessEnvelopeErrors(t *testing.T) {
	t.Run("envelope error", func(t *testing.T) {
		processor := &concurrentOrderedEventProcessing{}
		for _, err := range []error{io.EOF, errors.New("stream failed")} {
			result := processor.processEnvelope(bufferedEnvelopeResult{err: err})
			if result.err != err || result.event != nil {
				t.Fatalf("result: %#v, want original error %v and no event", result, err)
			}
		}
	})
	t.Run("embedded object error", func(t *testing.T) {
		processor := &concurrentOrderedEventProcessing{embeddedDecoder: scheme.Codecs.UniversalDeserializer()}
		result := processor.processEnvelope(bufferedEnvelopeResult{
			envelope: &metav1.WatchEvent{
				Type:   string(watch.Added),
				Object: runtime.RawExtension{Raw: []byte("{invalid")},
			},
		})
		if result.err == nil || !strings.Contains(result.err.Error(), "unable to decode watch event:") || result.event != nil {
			t.Fatalf("result: %#v, want decoding error and no event", result)
		}
	})
}

type concurrentReaderTestStream struct {
	streaming.Decoder
	decode func(runtime.Object) (runtime.Object, error)
	close  func() error
}

func (s *concurrentReaderTestStream) Decode(_ *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	obj, err := s.decode(into)
	return obj, nil, err
}

func (s *concurrentReaderTestStream) Close() error {
	if s.close != nil {
		return s.close()
	}
	return nil
}

func TestBufferedEnvelopeDecoderReadErrors(t *testing.T) {
	streamErr := errors.New("stream failed")
	for _, tc := range []struct {
		name   string
		result runtime.Object
		action string
		err    error
		want   string
	}{
		{name: "stream error", action: "ADDED", err: streamErr, want: "stream failed"},
		{name: "wrong object", result: &v1.Pod{}, action: "ADDED", want: "unable to decode to metav1.WatchEvent"},
		{name: "invalid type", action: "INVALID", want: "invalid watch event type"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			stream := &concurrentReaderTestStream{decode: func(into runtime.Object) (runtime.Object, error) {
				calls++
				envelope := into.(*metav1.WatchEvent)
				envelope.Type = "ADDED"
				envelope.Object.Raw = []byte("payload")
				if calls == 1 {
					return into, nil
				}
				envelope.Type = tc.action
				if tc.result != nil {
					return tc.result, tc.err
				}
				return into, tc.err
			}}
			d, finished := newBufferedEnvelopeDecoderForTest(t, stream, 2)
			waitForBufferedEnvelopeDecoder(t, finished)
			if result := <-d.incomingEnvelopesCh; result.err != nil || result.envelope == nil || result.envelope.Type != string(watch.Added) {
				t.Fatalf("first result: %#v", result)
			}
			result := <-d.incomingEnvelopesCh
			if result.err == nil || !strings.Contains(result.err.Error(), tc.want) {
				t.Fatalf("error=%v, want %q", result.err, tc.want)
			}
			if tc.err != nil && !errors.Is(result.err, tc.err) {
				t.Fatalf("stream error identity lost: %v", result.err)
			}
			if result.envelope != nil {
				t.Fatalf("terminal result retained partial data: %#v", result)
			}
			if _, ok := <-d.incomingEnvelopesCh; ok || calls != 2 {
				t.Fatalf("reader continued after terminal error: open=%v, calls=%d", ok, calls)
			}
		})
	}
}

func TestBufferedEnvelopeDecoderReadCancellation(t *testing.T) {
	t.Run("blocked queue", func(t *testing.T) {
		started := make(chan struct{})
		stream := &concurrentReaderTestStream{decode: func(into runtime.Object) (runtime.Object, error) {
			into.(*metav1.WatchEvent).Type = "ADDED"
			close(started)
			return into, nil
		}}
		d, finished := newBufferedEnvelopeDecoderForTest(t, stream, 0)
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("stream decode did not start")
		}
		d.close()
		waitForBufferedEnvelopeDecoder(t, finished)
		if _, ok := <-d.incomingEnvelopesCh; ok {
			t.Fatal("queue remained open after cancellation")
		}
	})
	t.Run("blocked stream", func(t *testing.T) {
		reader, writer := io.Pipe()
		defer writer.Close()
		d, finished := newBufferedEnvelopeDecoderForTest(t, streaming.NewDecoder(reader, getDecoder()), 0)
		d.close()
		waitForBufferedEnvelopeDecoder(t, finished)
	})
}

func newBufferedEnvelopeDecoderForTest(t *testing.T, stream streaming.Decoder, capacity int) (*bufferedEnvelopeDecoder, <-chan struct{}) {
	t.Helper()
	d := &bufferedEnvelopeDecoder{envelopeDecoder: stream, incomingEnvelopesCh: make(chan bufferedEnvelopeResult, capacity), doneCh: make(chan struct{})}
	finished := make(chan struct{})
	t.Cleanup(func() {
		d.close()
		waitForBufferedEnvelopeDecoder(t, finished)
	})
	go func() {
		defer close(finished)
		d.read()
	}()
	return d, finished
}

func waitForBufferedEnvelopeDecoder(t *testing.T, finished <-chan struct{}) {
	t.Helper()
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("receive loop did not stop")
	}
}

type concurrentProcessingTestDecoder struct {
	runtime.Decoder
	decode func([]byte) (runtime.Object, error)
}

func (d *concurrentProcessingTestDecoder) Decode(data []byte, _ *schema.GroupVersionKind, _ runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	obj, err := d.decode(data)
	return obj, nil, err
}

func TestConcurrentOrderedEventProcessingScheduleOrdering(t *testing.T) {
	firstReleaseCh := make(chan struct{}, 1)
	var wg sync.WaitGroup
	t.Cleanup(func() {
		close(firstReleaseCh)
		wg.Wait()
	})
	wantObject := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "first"}}
	wantErr := errors.New("second object failed")
	incomingEnvelopesCh := make(chan bufferedEnvelopeResult, 3)
	for _, name := range []string{"first", "second"} {
		incomingEnvelopesCh <- bufferedEnvelopeResult{envelope: &metav1.WatchEvent{
			Type: string(watch.Added), Object: runtime.RawExtension{Raw: []byte(name)},
		}}
	}
	incomingEnvelopesCh <- bufferedEnvelopeResult{err: io.EOF}
	close(incomingEnvelopesCh)
	p := &concurrentOrderedEventProcessing{
		incomingEnvelopesCh: incomingEnvelopesCh,
		processingResultsCh: make(chan chan *concurrentOrderedEventProcessingResult, 3),
		doneCh:              make(chan struct{}),
		embeddedDecoder: &concurrentProcessingTestDecoder{decode: func(data []byte) (runtime.Object, error) {
			if string(data) == "first" {
				<-firstReleaseCh
				return wantObject, nil
			}
			return nil, wantErr
		}},
	}
	finishedCh := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		p.scheduleEventProcessing(&wg)
		close(finishedCh)
	}()
	select {
	case <-finishedCh:
	case <-time.After(5 * time.Second):
		t.Fatal("scheduler did not stop after the envelope channel closed")
	}
	firstCh := <-p.processingResultsCh
	secondCh := <-p.processingResultsCh
	terminalCh := <-p.processingResultsCh
	select {
	case result := <-secondCh:
		if result.err == nil || result.err.Error() != "unable to decode watch event: "+wantErr.Error() || result.event != nil {
			t.Fatalf("second result: %#v, want embedded decoding error", result)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("second worker did not finish while the first was blocked")
	}
	select {
	case result := <-terminalCh:
		if result.err != io.EOF || result.event != nil {
			t.Fatalf("terminal result: %#v, want EOF", result)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("terminal worker did not finish")
	}
	firstReleaseCh <- struct{}{}
	select {
	case result := <-firstCh:
		if result.err != nil || result.event == nil || result.event.Type != watch.Added || result.event.Object != wantObject {
			t.Fatalf("first result: %#v, want Added with original object", result)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("first worker did not finish")
	}
}

func TestConcurrentOrderedEventProcessingScheduleCancellation(t *testing.T) {
	for _, fullQueue := range []bool{false, true} {
		t.Run(fmt.Sprintf("full queue=%t", fullQueue), func(t *testing.T) {
			incomingEnvelopesCh := make(chan bufferedEnvelopeResult)
			doneCh := make(chan struct{})
			p := &concurrentOrderedEventProcessing{
				incomingEnvelopesCh: incomingEnvelopesCh,
				processingResultsCh: make(chan chan *concurrentOrderedEventProcessingResult, 1),
				doneCh:              doneCh,
			}
			finishedCh := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				p.scheduleEventProcessing(&wg)
				close(finishedCh)
			}()
			if fullQueue {
				p.processingResultsCh <- make(chan *concurrentOrderedEventProcessingResult, 1)
				select {
				case incomingEnvelopesCh <- bufferedEnvelopeResult{err: io.EOF}:
				case <-time.After(5 * time.Second):
					close(doneCh)
					t.Fatal("scheduler did not receive the envelope")
				}
				select {
				case incomingEnvelopesCh <- bufferedEnvelopeResult{err: io.EOF}:
					t.Error("scheduler read another envelope while the results channel was full")
				default:
				}
			}
			close(doneCh)
			select {
			case <-finishedCh:
			case <-time.After(5 * time.Second):
				t.Fatal("scheduler did not stop after cancellation")
			}
			wg.Wait()
		})
	}
}

func TestConcurrentDecoderConstruction(t *testing.T) {
	stream := &concurrentReaderTestStream{decode: func(into runtime.Object) (runtime.Object, error) {
		return nil, io.EOF
	}}
	objectDecoder := getDecoder()
	d := NewConcurrentDecoder(stream, objectDecoder)
	defer d.Close()
	if d.concurrentOrderedEventProcessing.embeddedDecoder != objectDecoder {
		t.Fatal("constructor did not retain the embedded object decoder")
	}
	if d.concurrentOrderedEventProcessing.incomingEnvelopesCh != d.bufferedEnvelopeDecoder.incomingEnvelopesCh {
		t.Fatal("event processor does not use the envelope reader's queue")
	}
	if d.concurrentOrderedEventProcessing.doneCh == nil || d.concurrentOrderedEventProcessing.doneCh == d.bufferedEnvelopeDecoder.doneCh {
		t.Fatal("event processor does not have its own cancellation channel")
	}
	if capacity := cap(d.concurrentOrderedEventProcessing.processingResultsCh); capacity != defaultConcurrentEventProcessingConcurrency-1 {
		t.Fatalf("processing results channel capacity=%d, want %d", capacity, defaultConcurrentEventProcessingConcurrency-1)
	}
	if capacity := cap(d.bufferedEnvelopeDecoder.incomingEnvelopesCh); capacity != defaultBufferedEnvelopeBufferSize {
		t.Fatalf("queue capacity=%d, want %d", capacity, defaultBufferedEnvelopeBufferSize)
	}
	if eventType, obj, err := d.Decode(); eventType != "" || obj != nil || err != io.EOF {
		t.Fatalf("Decode() = (%q, %v, %v), want empty type, nil, EOF", eventType, obj, err)
	}
	finishedCh := make(chan error, 1)
	go func() {
		_, _, err := d.Decode()
		finishedCh <- err
	}()
	select {
	case err := <-finishedCh:
		if err != io.EOF {
			t.Fatalf("second Decode() error = %v, want EOF", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("second Decode blocked after EOF")
	}
}

func TestConcurrentDecoderClose(t *testing.T) {
	t.Run("completed results", func(t *testing.T) {
		calls := 0
		stream := &concurrentReaderTestStream{decode: func(into runtime.Object) (runtime.Object, error) {
			calls++
			if calls > 1 {
				return nil, io.EOF
			}
			into.(*metav1.WatchEvent).Type = string(watch.Added)
			return into, nil
		}}
		objectDecoder := &concurrentProcessingTestDecoder{decode: func([]byte) (runtime.Object, error) {
			return &v1.Pod{}, nil
		}}
		d := NewConcurrentDecoder(stream, objectDecoder)
		defer d.Close()
		d.wg.Wait()
		if got := len(d.concurrentOrderedEventProcessing.processingResultsCh); got != 2 {
			t.Fatalf("buffered result channels=%d, want 2", got)
		}
		d.Close()
		if got := len(d.concurrentOrderedEventProcessing.processingResultsCh); got != 0 {
			t.Fatalf("Close retained %d completed result channels", got)
		}
		if eventType, obj, err := d.Decode(); eventType != "" || obj != nil || err != io.EOF {
			t.Fatalf("Decode after Close = (%q, %v, %v), want empty type, nil, EOF", eventType, obj, err)
		}
	})
	t.Run("waits for worker", func(t *testing.T) {
		startedCh := make(chan struct{})
		releaseCh := make(chan struct{})
		exitedCh := make(chan struct{})
		streamClosedCh := make(chan struct{})
		var releaseOnce sync.Once
		release := func() { releaseOnce.Do(func() { close(releaseCh) }) }
		defer release()
		calls := 0
		stream := &concurrentReaderTestStream{
			decode: func(into runtime.Object) (runtime.Object, error) {
				calls++
				if calls > 1 {
					return nil, io.EOF
				}
				into.(*metav1.WatchEvent).Type = string(watch.Added)
				return into, nil
			},
			close: func() error {
				close(streamClosedCh)
				return nil
			},
		}
		objectDecoder := &concurrentProcessingTestDecoder{decode: func([]byte) (runtime.Object, error) {
			close(startedCh)
			defer close(exitedCh)
			<-releaseCh
			return &v1.Pod{}, nil
		}}
		d := NewConcurrentDecoder(stream, objectDecoder)
		t.Cleanup(func() {
			release()
			d.Close()
		})
		select {
		case <-startedCh:
		case <-time.After(5 * time.Second):
			t.Fatal("worker did not start")
		}
		decodedCh := make(chan error, 1)
		go func() {
			_, _, err := d.Decode()
			decodedCh <- err
		}()
		closedCh := make(chan struct{})
		go func() {
			d.Close()
			close(closedCh)
		}()
		select {
		case <-streamClosedCh:
		case <-time.After(5 * time.Second):
			t.Fatal("Close did not close the envelope stream before waiting")
		}
		select {
		case err := <-decodedCh:
			if err != io.EOF {
				t.Fatalf("Decode() error = %v, want EOF", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("Decode did not stop while Close waited for the worker")
		}
		select {
		case <-closedCh:
			t.Fatal("Close returned while the worker was still decoding")
		case <-time.After(20 * time.Millisecond):
		}
		release()
		select {
		case <-closedCh:
		case <-time.After(5 * time.Second):
			t.Fatal("Close did not return after the worker finished")
		}
		select {
		case <-exitedCh:
		default:
			t.Fatal("Close returned before embedded decoding finished")
		}
		waitForConcurrentDecoderQueueClosure(t, d)
	})
	t.Run("blocked stream", func(t *testing.T) {
		reader, writer := io.Pipe()
		defer writer.Close()
		d := NewConcurrentDecoder(streaming.NewDecoder(reader, getDecoder()), getDecoder())
		defer d.Close()
		d.Close()
		d.Close()
		waitForConcurrentDecoderQueueClosure(t, d)
	})
	t.Run("full queue", func(t *testing.T) {
		envelopeCount := defaultBufferedEnvelopeBufferSize + defaultConcurrentEventProcessingConcurrency + 1
		read := make(chan struct{}, envelopeCount)
		stream := &concurrentReaderTestStream{decode: func(into runtime.Object) (runtime.Object, error) {
			into.(*metav1.WatchEvent).Type = "ADDED"
			into.(*metav1.WatchEvent).Object.Raw = []byte("buffered payload")
			select {
			case read <- struct{}{}:
			default:
			}
			return into, nil
		}}
		objectDecoder := &concurrentProcessingTestDecoder{decode: func([]byte) (runtime.Object, error) {
			return &v1.Pod{}, nil
		}}
		d := NewConcurrentDecoder(stream, objectDecoder)
		defer d.Close()
		for i := 0; i < envelopeCount; i++ {
			select {
			case <-read:
			case <-time.After(5 * time.Second):
				t.Fatalf("reader stalled before filling queue at envelope %d", i)
			}
		}
		if got := len(d.bufferedEnvelopeDecoder.incomingEnvelopesCh); got != defaultBufferedEnvelopeBufferSize {
			t.Fatalf("buffered envelopes=%d, want %d", got, defaultBufferedEnvelopeBufferSize)
		}
		if got := len(d.concurrentOrderedEventProcessing.processingResultsCh); got != defaultConcurrentEventProcessingConcurrency-1 {
			t.Fatalf("buffered result channels=%d, want %d", got, defaultConcurrentEventProcessingConcurrency-1)
		}
		d.Close()
		if got := len(d.bufferedEnvelopeDecoder.incomingEnvelopesCh); got != 0 {
			t.Fatalf("Close retained %d envelopes", got)
		}
		if got := len(d.concurrentOrderedEventProcessing.processingResultsCh); got != 0 {
			t.Fatalf("Close retained %d result channels", got)
		}
		waitForConcurrentDecoderQueueClosure(t, d)
		if eventType, obj, err := d.Decode(); eventType != "" || obj != nil || err != io.EOF {
			t.Fatalf("Decode after Close = (%q, %v, %v), want empty type, nil, EOF", eventType, obj, err)
		}
	})
}

func TestConcurrentDecoderDecode(t *testing.T) {
	p := &concurrentOrderedEventProcessing{
		processingResultsCh: make(chan chan *concurrentOrderedEventProcessingResult, 2),
		doneCh:              make(chan struct{}),
	}
	d := &ConcurrentDecoder{concurrentOrderedEventProcessing: p}
	firstCh := make(chan *concurrentOrderedEventProcessingResult, 1)
	secondCh := make(chan *concurrentOrderedEventProcessingResult, 1)
	p.processingResultsCh <- firstCh
	p.processingResultsCh <- secondCh

	wantErr := errors.New("decode embedded object")
	secondCh <- &concurrentOrderedEventProcessingResult{err: wantErr}
	obj := &v1.Pod{}
	firstCh <- &concurrentOrderedEventProcessingResult{event: &watch.Event{Type: watch.Added, Object: obj}}

	eventType, got, err := d.Decode()
	if eventType != watch.Added || got != obj || err != nil {
		t.Fatalf("Decode() = (%q, %v, %v), want Added, original object, nil", eventType, got, err)
	}
	eventType, got, err = d.Decode()
	if eventType != "" || got != nil || err != wantErr {
		t.Fatalf("Decode() = (%q, %v, %v), want empty type, nil, %v", eventType, got, err, wantErr)
	}
}

func TestConcurrentDecoderDecodeCancellation(t *testing.T) {
	t.Run("populated queue", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			doneCh := make(chan struct{})
			p := &concurrentOrderedEventProcessing{
				processingResultsCh: make(chan chan *concurrentOrderedEventProcessingResult, 1),
				doneCh:              doneCh,
			}
			d := &ConcurrentDecoder{concurrentOrderedEventProcessing: p}
			resultCh := make(chan *concurrentOrderedEventProcessingResult, 1)
			resultCh <- &concurrentOrderedEventProcessingResult{event: &watch.Event{Type: watch.Added, Object: &v1.Pod{}}}
			p.processingResultsCh <- resultCh
			close(doneCh)
			if eventType, obj, err := d.Decode(); eventType != "" || obj != nil || err != io.EOF {
				t.Fatalf("Decode after cancellation = (%q, %v, %v), want empty type, nil, EOF", eventType, obj, err)
			}
		}
	})
	for name, waitingForResult := range map[string]bool{"waiting for channel": false, "waiting for result": true} {
		t.Run(name, func(t *testing.T) {
			doneCh := make(chan struct{})
			p := &concurrentOrderedEventProcessing{
				processingResultsCh: make(chan chan *concurrentOrderedEventProcessingResult),
				doneCh:              doneCh,
			}
			d := &ConcurrentDecoder{concurrentOrderedEventProcessing: p}
			finishedCh := make(chan error, 1)
			go func() {
				_, _, err := d.Decode()
				finishedCh <- err
			}()
			if waitingForResult {
				select {
				case p.processingResultsCh <- make(chan *concurrentOrderedEventProcessingResult):
				case <-time.After(5 * time.Second):
					close(doneCh)
					t.Fatal("Decode did not receive the result channel")
				}
			}
			close(doneCh)
			select {
			case err := <-finishedCh:
				if err != io.EOF {
					t.Fatalf("Decode() error = %v, want EOF", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("Decode did not stop after cancellation")
			}
		})
	}
}

func TestConcurrentDecoderDecodeClosedResults(t *testing.T) {
	p := &concurrentOrderedEventProcessing{
		processingResultsCh: make(chan chan *concurrentOrderedEventProcessingResult, 1),
		doneCh:              make(chan struct{}),
	}
	resultCh := make(chan *concurrentOrderedEventProcessingResult, 1)
	obj := &v1.Pod{}
	resultCh <- &concurrentOrderedEventProcessingResult{event: &watch.Event{Type: watch.Added, Object: obj}}
	p.processingResultsCh <- resultCh
	close(p.processingResultsCh)
	d := &ConcurrentDecoder{concurrentOrderedEventProcessing: p}
	if eventType, got, err := d.Decode(); eventType != watch.Added || got != obj || err != nil {
		t.Fatalf("Decode() = (%q, %v, %v), want Added, original object, nil", eventType, got, err)
	}
	for i := 0; i < 2; i++ {
		if eventType, got, err := d.Decode(); eventType != "" || got != nil || err != io.EOF {
			t.Fatalf("Decode after channel closure = (%q, %v, %v), want empty type, nil, EOF", eventType, got, err)
		}
	}
}

func waitForConcurrentDecoderQueueClosure(t *testing.T, d *ConcurrentDecoder) {
	t.Helper()
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	for {
		select {
		case _, ok := <-d.bufferedEnvelopeDecoder.incomingEnvelopesCh:
			if !ok {
				return
			}
		case <-timer.C:
			t.Fatal("envelope reader did not close its queue")
		}
	}
}
