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
	"errors"
	"fmt"
	"io"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
)

func TestConcurrentDecoderWithFixedWorkersAndOutputQueueReadAhead(t *testing.T) {
	count := 2 * defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency
	terminal := errors.New("stream ended")
	objectDecoder := &concurrentProcessingTestDecoder{decode: func(data []byte) (runtime.Object, error) {
		return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: string(data)}}, nil
	}}
	d := NewConcurrentDecoderWithFixedWorkersAndOutputQueue(outputQueueTestStream(count, terminal), objectDecoder)
	defer d.Close()
	done := make(chan struct{})
	go func() {
		d.wg.Wait()
		close(done)
	}()
	waitForOutputQueueTest(t, done)
	if got := len(d.fixedWorkersBufferedConcurrentOrderedEventProcessing.outputResultsCh); got != count+1 {
		t.Fatalf("buffered results = %d, want %d events plus stream error", got, count)
	}
	for i := 0; i < count; i++ {
		action, obj, err := d.Decode()
		if err != nil || action != watch.Added {
			t.Fatalf("event %d: type=%q object=%v error=%v", i, action, obj, err)
		}
		if got := obj.(*v1.Pod).Name; got != fmt.Sprint(i) {
			t.Fatalf("event %d: name=%q", i, got)
		}
	}
	if _, _, err := d.Decode(); err != terminal {
		t.Fatalf("stream error = %v, want %v", err, terminal)
	}
	for i := 0; i < 2; i++ {
		if _, _, err := d.Decode(); err != io.EOF {
			t.Fatalf("Decode after stream error = %v, want EOF", err)
		}
	}
}

func TestConcurrentDecoderWithFixedWorkersAndOutputQueueOrdering(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	secondFinished := make(chan struct{})
	wantErr := errors.New("invalid object")
	objectDecoder := &concurrentProcessingTestDecoder{decode: func(data []byte) (runtime.Object, error) {
		if string(data) == "0" {
			<-release
			return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "first"}}, nil
		}
		close(secondFinished)
		return nil, wantErr
	}}
	d := NewConcurrentDecoderWithFixedWorkersAndOutputQueue(outputQueueTestStream(2, io.EOF), objectDecoder)
	defer d.Close()
	defer unblock()
	waitForOutputQueueTest(t, secondFinished)
	if got := len(d.fixedWorkersBufferedConcurrentOrderedEventProcessing.outputResultsCh); got != 0 {
		t.Fatalf("buffered results = %d while the first decode is blocked, want 0", got)
	}
	unblock()
	action, obj, err := d.Decode()
	if err != nil || action != watch.Added || obj.(*v1.Pod).Name != "first" {
		t.Fatalf("first result = (%q, %v, %v), want Added first", action, obj, err)
	}
	if action, obj, err := d.Decode(); action != "" || obj != nil || err == nil || err.Error() != "unable to decode watch event: "+wantErr.Error() {
		t.Fatalf("second result = (%q, %v, %v), want embedded decoding error", action, obj, err)
	}
	if _, _, err := d.Decode(); err != io.EOF {
		t.Fatalf("terminal error = %v, want EOF", err)
	}
}

func TestConcurrentDecoderWithFixedWorkersAndOutputQueueClose(t *testing.T) {
	t.Run("full output queue", func(t *testing.T) {
		count := defaultFixedWorkersBufferedConcurrentOutputBufferSize + defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency + 1
		decoded := make(chan struct{}, count)
		objectDecoder := &concurrentProcessingTestDecoder{decode: func([]byte) (runtime.Object, error) {
			decoded <- struct{}{}
			return &v1.Pod{}, nil
		}}
		d := NewConcurrentDecoderWithFixedWorkersAndOutputQueue(outputQueueTestStream(count, io.EOF), objectDecoder)
		defer d.Close()
		for i := 0; i < count-1; i++ {
			waitForOutputQueueTest(t, decoded)
		}
		if got := len(d.fixedWorkersBufferedConcurrentOrderedEventProcessing.outputResultsCh); got != defaultFixedWorkersBufferedConcurrentOutputBufferSize {
			t.Fatalf("output queue length = %d, want %d", got, defaultFixedWorkersBufferedConcurrentOutputBufferSize)
		}
		done := make(chan struct{})
		go func() {
			d.Close()
			close(done)
		}()
		waitForOutputQueueTest(t, done)
		if got := len(d.fixedWorkersBufferedConcurrentOrderedEventProcessing.outputResultsCh); got != 0 {
			t.Fatalf("Close retained %d output results", got)
		}
		if action, obj, err := d.Decode(); action != "" || obj != nil || err != io.EOF {
			t.Fatalf("Decode after Close = (%q, %v, %v), want empty type, nil, EOF", action, obj, err)
		}
	})
	t.Run("in-flight decode", func(t *testing.T) {
		started := make(chan struct{})
		release := make(chan struct{})
		var releaseOnce sync.Once
		unblock := func() { releaseOnce.Do(func() { close(release) }) }
		objectDecoder := &concurrentProcessingTestDecoder{decode: func([]byte) (runtime.Object, error) {
			close(started)
			<-release
			return &v1.Pod{}, nil
		}}
		d := NewConcurrentDecoderWithFixedWorkersAndOutputQueue(outputQueueTestStream(1, io.EOF), objectDecoder)
		defer d.Close()
		defer unblock()
		waitForOutputQueueTest(t, started)
		decodeDone := make(chan struct{})
		go func() {
			defer close(decodeDone)
			if _, _, err := d.Decode(); err != io.EOF {
				t.Errorf("Decode on Close = %v, want EOF", err)
			}
		}()
		closeDone := make(chan struct{})
		go func() {
			d.Close()
			close(closeDone)
		}()
		waitForOutputQueueTest(t, d.fixedWorkersBufferedConcurrentOrderedEventProcessing.doneCh)
		waitForOutputQueueTest(t, decodeDone)
		select {
		case <-closeDone:
			t.Fatal("Close returned before the in-flight object decode finished")
		default:
		}
		unblock()
		waitForOutputQueueTest(t, closeDone)
	})
	t.Run("blocked Decode", func(t *testing.T) {
		closed := make(chan struct{})
		stream := &concurrentReaderTestStream{
			decode: func(runtime.Object) (runtime.Object, error) { <-closed; return nil, io.EOF },
			close:  func() error { close(closed); return nil },
		}
		d := NewConcurrentDecoderWithFixedWorkersAndOutputQueue(stream, getDecoder())
		defer d.Close()
		done := make(chan struct{})
		go func() {
			defer close(done)
			if _, _, err := d.Decode(); err != io.EOF {
				t.Errorf("Decode on Close = %v, want EOF", err)
			}
		}()
		d.Close()
		waitForOutputQueueTest(t, done)
	})
}

func TestConcurrentDecoderWithFixedWorkersAndOutputQueueWorkerLimit(t *testing.T) {
	const workers = defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency
	const count = 3 * workers
	started := make(chan struct{}, count)
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	objectDecoder := &concurrentProcessingTestDecoder{decode: func(data []byte) (runtime.Object, error) {
		started <- struct{}{}
		<-release
		return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: string(data)}}, nil
	}}
	d := NewConcurrentDecoderWithFixedWorkersAndOutputQueue(outputQueueTestStream(count, io.EOF), objectDecoder)
	defer d.Close()
	defer unblock()
	for i := 0; i < workers; i++ {
		waitForOutputQueueTest(t, started)
	}
	select {
	case <-started:
		t.Fatalf("more than %d object decodes started while all workers were blocked", workers)
	case <-time.After(100 * time.Millisecond):
	}
	unblock()
	done := make(chan struct{})
	go func() {
		d.wg.Wait()
		close(done)
	}()
	waitForOutputQueueTest(t, done)
	for i := 0; i < count; i++ {
		action, obj, err := d.Decode()
		if err != nil || action != watch.Added {
			t.Fatalf("event %d: type=%q object=%v error=%v", i, action, obj, err)
		}
		if got := obj.(*v1.Pod).Name; got != fmt.Sprint(i) {
			t.Fatalf("event %d: name=%q", i, got)
		}
	}
	if _, _, err := d.Decode(); err != io.EOF {
		t.Fatalf("terminal error = %v, want EOF", err)
	}
}
