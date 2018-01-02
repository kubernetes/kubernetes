package logger

import (
	"context"
	"strconv"
	"testing"
	"time"
)

type mockLogger struct{ c chan *Message }

func (l *mockLogger) Log(msg *Message) error {
	l.c <- msg
	return nil
}

func (l *mockLogger) Name() string {
	return "mock"
}

func (l *mockLogger) Close() error {
	return nil
}

func TestRingLogger(t *testing.T) {
	mockLog := &mockLogger{make(chan *Message)} // no buffer on this channel
	ring := newRingLogger(mockLog, Info{}, 1)
	defer ring.setClosed()

	// this should never block
	ring.Log(&Message{Line: []byte("1")})
	ring.Log(&Message{Line: []byte("2")})
	ring.Log(&Message{Line: []byte("3")})

	select {
	case msg := <-mockLog.c:
		if string(msg.Line) != "1" {
			t.Fatalf("got unexpected msg: %q", string(msg.Line))
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("timeout reading log message")
	}

	select {
	case msg := <-mockLog.c:
		t.Fatalf("expected no more messages in the queue, got: %q", string(msg.Line))
	default:
	}
}

func TestRingCap(t *testing.T) {
	r := newRing(5)
	for i := 0; i < 10; i++ {
		// queue messages with "0" to "10"
		// the "5" to "10" messages should be dropped since we only allow 5 bytes in the buffer
		if err := r.Enqueue(&Message{Line: []byte(strconv.Itoa(i))}); err != nil {
			t.Fatal(err)
		}
	}

	// should have messages in the queue for "5" to "10"
	for i := 0; i < 5; i++ {
		m, err := r.Dequeue()
		if err != nil {
			t.Fatal(err)
		}
		if string(m.Line) != strconv.Itoa(i) {
			t.Fatalf("got unexpected message for iter %d: %s", i, string(m.Line))
		}
	}

	// queue a message that's bigger than the buffer cap
	if err := r.Enqueue(&Message{Line: []byte("hello world")}); err != nil {
		t.Fatal(err)
	}

	// queue another message that's bigger than the buffer cap
	if err := r.Enqueue(&Message{Line: []byte("eat a banana")}); err != nil {
		t.Fatal(err)
	}

	m, err := r.Dequeue()
	if err != nil {
		t.Fatal(err)
	}
	if string(m.Line) != "hello world" {
		t.Fatalf("got unexpected message: %s", string(m.Line))
	}
	if len(r.queue) != 0 {
		t.Fatalf("expected queue to be empty, got: %d", len(r.queue))
	}
}

func TestRingClose(t *testing.T) {
	r := newRing(1)
	if err := r.Enqueue(&Message{Line: []byte("hello")}); err != nil {
		t.Fatal(err)
	}
	r.Close()
	if err := r.Enqueue(&Message{}); err != errClosed {
		t.Fatalf("expected errClosed, got: %v", err)
	}
	if len(r.queue) != 1 {
		t.Fatal("expected empty queue")
	}
	if m, err := r.Dequeue(); err == nil || m != nil {
		t.Fatal("expected err on Dequeue after close")
	}

	ls := r.Drain()
	if len(ls) != 1 {
		t.Fatalf("expected one message: %v", ls)
	}
	if string(ls[0].Line) != "hello" {
		t.Fatalf("got unexpected message: %s", string(ls[0].Line))
	}
}

func TestRingDrain(t *testing.T) {
	r := newRing(5)
	for i := 0; i < 5; i++ {
		if err := r.Enqueue(&Message{Line: []byte(strconv.Itoa(i))}); err != nil {
			t.Fatal(err)
		}
	}

	ls := r.Drain()
	if len(ls) != 5 {
		t.Fatal("got unexpected length after drain")
	}

	for i := 0; i < 5; i++ {
		if string(ls[i].Line) != strconv.Itoa(i) {
			t.Fatalf("got unexpected message at position %d: %s", i, string(ls[i].Line))
		}
	}
	if r.sizeBytes != 0 {
		t.Fatalf("expected buffer size to be 0 after drain, got: %d", r.sizeBytes)
	}

	ls = r.Drain()
	if len(ls) != 0 {
		t.Fatalf("expected 0 messages on 2nd drain: %v", ls)
	}

}

type nopLogger struct{}

func (nopLogger) Name() string       { return "nopLogger" }
func (nopLogger) Close() error       { return nil }
func (nopLogger) Log(*Message) error { return nil }

func BenchmarkRingLoggerThroughputNoReceiver(b *testing.B) {
	mockLog := &mockLogger{make(chan *Message)}
	defer mockLog.Close()
	l := NewRingLogger(mockLog, Info{}, -1)
	msg := &Message{Line: []byte("hello humans and everyone else!")}
	b.SetBytes(int64(len(msg.Line)))

	for i := 0; i < b.N; i++ {
		if err := l.Log(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRingLoggerThroughputWithReceiverDelay0(b *testing.B) {
	l := NewRingLogger(nopLogger{}, Info{}, -1)
	msg := &Message{Line: []byte("hello humans and everyone else!")}
	b.SetBytes(int64(len(msg.Line)))

	for i := 0; i < b.N; i++ {
		if err := l.Log(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func consumeWithDelay(delay time.Duration, c <-chan *Message) (cancel func()) {
	started := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		close(started)
		ticker := time.NewTicker(delay)
		for range ticker.C {
			select {
			case <-ctx.Done():
				ticker.Stop()
				return
			case <-c:
			}
		}
	}()
	<-started
	return cancel
}

func BenchmarkRingLoggerThroughputConsumeDelay1(b *testing.B) {
	mockLog := &mockLogger{make(chan *Message)}
	defer mockLog.Close()
	l := NewRingLogger(mockLog, Info{}, -1)
	msg := &Message{Line: []byte("hello humans and everyone else!")}
	b.SetBytes(int64(len(msg.Line)))

	cancel := consumeWithDelay(1*time.Millisecond, mockLog.c)
	defer cancel()

	for i := 0; i < b.N; i++ {
		if err := l.Log(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRingLoggerThroughputConsumeDelay10(b *testing.B) {
	mockLog := &mockLogger{make(chan *Message)}
	defer mockLog.Close()
	l := NewRingLogger(mockLog, Info{}, -1)
	msg := &Message{Line: []byte("hello humans and everyone else!")}
	b.SetBytes(int64(len(msg.Line)))

	cancel := consumeWithDelay(10*time.Millisecond, mockLog.c)
	defer cancel()

	for i := 0; i < b.N; i++ {
		if err := l.Log(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRingLoggerThroughputConsumeDelay50(b *testing.B) {
	mockLog := &mockLogger{make(chan *Message)}
	defer mockLog.Close()
	l := NewRingLogger(mockLog, Info{}, -1)
	msg := &Message{Line: []byte("hello humans and everyone else!")}
	b.SetBytes(int64(len(msg.Line)))

	cancel := consumeWithDelay(50*time.Millisecond, mockLog.c)
	defer cancel()

	for i := 0; i < b.N; i++ {
		if err := l.Log(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRingLoggerThroughputConsumeDelay100(b *testing.B) {
	mockLog := &mockLogger{make(chan *Message)}
	defer mockLog.Close()
	l := NewRingLogger(mockLog, Info{}, -1)
	msg := &Message{Line: []byte("hello humans and everyone else!")}
	b.SetBytes(int64(len(msg.Line)))

	cancel := consumeWithDelay(100*time.Millisecond, mockLog.c)
	defer cancel()

	for i := 0; i < b.N; i++ {
		if err := l.Log(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRingLoggerThroughputConsumeDelay300(b *testing.B) {
	mockLog := &mockLogger{make(chan *Message)}
	defer mockLog.Close()
	l := NewRingLogger(mockLog, Info{}, -1)
	msg := &Message{Line: []byte("hello humans and everyone else!")}
	b.SetBytes(int64(len(msg.Line)))

	cancel := consumeWithDelay(300*time.Millisecond, mockLog.c)
	defer cancel()

	for i := 0; i < b.N; i++ {
		if err := l.Log(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRingLoggerThroughputConsumeDelay500(b *testing.B) {
	mockLog := &mockLogger{make(chan *Message)}
	defer mockLog.Close()
	l := NewRingLogger(mockLog, Info{}, -1)
	msg := &Message{Line: []byte("hello humans and everyone else!")}
	b.SetBytes(int64(len(msg.Line)))

	cancel := consumeWithDelay(500*time.Millisecond, mockLog.c)
	defer cancel()

	for i := 0; i < b.N; i++ {
		if err := l.Log(msg); err != nil {
			b.Fatal(err)
		}
	}
}
