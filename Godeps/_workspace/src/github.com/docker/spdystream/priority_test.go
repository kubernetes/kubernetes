package spdystream

import (
	"sync"
	"testing"
	"time"

	"golang.org/x/net/spdy"
)

func TestPriorityQueueOrdering(t *testing.T) {
	queue := NewPriorityFrameQueue(150)
	data1 := &spdy.DataFrame{}
	data2 := &spdy.DataFrame{}
	data3 := &spdy.DataFrame{}
	data4 := &spdy.DataFrame{}
	queue.Push(data1, 2)
	queue.Push(data2, 1)
	queue.Push(data3, 1)
	queue.Push(data4, 0)

	if queue.Pop() != data4 {
		t.Fatalf("Wrong order, expected data4 first")
	}
	if queue.Pop() != data2 {
		t.Fatalf("Wrong order, expected data2 second")
	}
	if queue.Pop() != data3 {
		t.Fatalf("Wrong order, expected data3 third")
	}
	if queue.Pop() != data1 {
		t.Fatalf("Wrong order, expected data1 fourth")
	}

	// Insert 50 Medium priority frames
	for i := spdy.StreamId(50); i < 100; i++ {
		queue.Push(&spdy.DataFrame{StreamId: i}, 1)
	}
	// Insert 50 low priority frames
	for i := spdy.StreamId(100); i < 150; i++ {
		queue.Push(&spdy.DataFrame{StreamId: i}, 2)
	}
	// Insert 50 high priority frames
	for i := spdy.StreamId(0); i < 50; i++ {
		queue.Push(&spdy.DataFrame{StreamId: i}, 0)
	}

	for i := spdy.StreamId(0); i < 150; i++ {
		frame := queue.Pop()
		if frame.(*spdy.DataFrame).StreamId != i {
			t.Fatalf("Wrong frame\nActual: %d\nExpecting: %d", frame.(*spdy.DataFrame).StreamId, i)
		}
	}
}

func TestPriorityQueueSync(t *testing.T) {
	queue := NewPriorityFrameQueue(150)
	var wg sync.WaitGroup
	insertRange := func(start, stop spdy.StreamId, priority uint8) {
		for i := start; i < stop; i++ {
			queue.Push(&spdy.DataFrame{StreamId: i}, priority)
		}
		wg.Done()
	}
	wg.Add(3)
	go insertRange(spdy.StreamId(100), spdy.StreamId(150), 2)
	go insertRange(spdy.StreamId(0), spdy.StreamId(50), 0)
	go insertRange(spdy.StreamId(50), spdy.StreamId(100), 1)

	wg.Wait()
	for i := spdy.StreamId(0); i < 150; i++ {
		frame := queue.Pop()
		if frame.(*spdy.DataFrame).StreamId != i {
			t.Fatalf("Wrong frame\nActual: %d\nExpecting: %d", frame.(*spdy.DataFrame).StreamId, i)
		}
	}
}

func TestPriorityQueueBlocking(t *testing.T) {
	queue := NewPriorityFrameQueue(15)
	for i := 0; i < 15; i++ {
		queue.Push(&spdy.DataFrame{}, 2)
	}
	doneChan := make(chan bool)
	go func() {
		queue.Push(&spdy.DataFrame{}, 2)
		close(doneChan)
	}()
	select {
	case <-doneChan:
		t.Fatalf("Push succeeded, expected to block")
	case <-time.After(time.Millisecond):
		break
	}

	queue.Pop()

	select {
	case <-doneChan:
		break
	case <-time.After(time.Millisecond):
		t.Fatalf("Push should have succeeded, but timeout reached")
	}

	for i := 0; i < 15; i++ {
		queue.Pop()
	}
}
