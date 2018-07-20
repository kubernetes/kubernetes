package logger

import (
	"errors"
	"sync"
	"sync/atomic"

	"github.com/sirupsen/logrus"
)

const (
	defaultRingMaxSize = 1e6 // 1MB
)

// RingLogger is a ring buffer that implements the Logger interface.
// This is used when lossy logging is OK.
type RingLogger struct {
	buffer    *messageRing
	l         Logger
	logInfo   Info
	closeFlag int32
}

type ringWithReader struct {
	*RingLogger
}

func (r *ringWithReader) ReadLogs(cfg ReadConfig) *LogWatcher {
	reader, ok := r.l.(LogReader)
	if !ok {
		// something is wrong if we get here
		panic("expected log reader")
	}
	return reader.ReadLogs(cfg)
}

func newRingLogger(driver Logger, logInfo Info, maxSize int64) *RingLogger {
	l := &RingLogger{
		buffer:  newRing(maxSize),
		l:       driver,
		logInfo: logInfo,
	}
	go l.run()
	return l
}

// NewRingLogger creates a new Logger that is implemented as a RingBuffer wrapping
// the passed in logger.
func NewRingLogger(driver Logger, logInfo Info, maxSize int64) Logger {
	if maxSize < 0 {
		maxSize = defaultRingMaxSize
	}
	l := newRingLogger(driver, logInfo, maxSize)
	if _, ok := driver.(LogReader); ok {
		return &ringWithReader{l}
	}
	return l
}

// Log queues messages into the ring buffer
func (r *RingLogger) Log(msg *Message) error {
	if r.closed() {
		return errClosed
	}
	return r.buffer.Enqueue(msg)
}

// Name returns the name of the underlying logger
func (r *RingLogger) Name() string {
	return r.l.Name()
}

func (r *RingLogger) closed() bool {
	return atomic.LoadInt32(&r.closeFlag) == 1
}

func (r *RingLogger) setClosed() {
	atomic.StoreInt32(&r.closeFlag, 1)
}

// Close closes the logger
func (r *RingLogger) Close() error {
	r.setClosed()
	r.buffer.Close()
	// empty out the queue
	var logErr bool
	for _, msg := range r.buffer.Drain() {
		if logErr {
			// some error logging a previous message, so re-insert to message pool
			// and assume log driver is hosed
			PutMessage(msg)
			continue
		}

		if err := r.l.Log(msg); err != nil {
			logrus.WithField("driver", r.l.Name()).WithField("container", r.logInfo.ContainerID).Errorf("Error writing log message: %v", r.l)
			logErr = true
		}
	}
	return r.l.Close()
}

// run consumes messages from the ring buffer and forwards them to the underling
// logger.
// This is run in a goroutine when the RingLogger is created
func (r *RingLogger) run() {
	for {
		if r.closed() {
			return
		}
		msg, err := r.buffer.Dequeue()
		if err != nil {
			// buffer is closed
			return
		}
		if err := r.l.Log(msg); err != nil {
			logrus.WithField("driver", r.l.Name()).WithField("container", r.logInfo.ContainerID).Errorf("Error writing log message: %v", r.l)
		}
	}
}

type messageRing struct {
	mu sync.Mutex
	// signals callers of `Dequeue` to wake up either on `Close` or when a new `Message` is added
	wait *sync.Cond

	sizeBytes int64 // current buffer size
	maxBytes  int64 // max buffer size size
	queue     []*Message
	closed    bool
}

func newRing(maxBytes int64) *messageRing {
	queueSize := 1000
	if maxBytes == 0 || maxBytes == 1 {
		// With 0 or 1 max byte size, the maximum size of the queue would only ever be 1
		// message long.
		queueSize = 1
	}

	r := &messageRing{queue: make([]*Message, 0, queueSize), maxBytes: maxBytes}
	r.wait = sync.NewCond(&r.mu)
	return r
}

// Enqueue adds a message to the buffer queue
// If the message is too big for the buffer it drops the oldest messages to make room
// If there are no messages in the queue and the message is still too big, it adds the message anyway.
func (r *messageRing) Enqueue(m *Message) error {
	mSize := int64(len(m.Line))

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return errClosed
	}
	if mSize+r.sizeBytes > r.maxBytes && len(r.queue) > 0 {
		r.wait.Signal()
		r.mu.Unlock()
		return nil
	}

	r.queue = append(r.queue, m)
	r.sizeBytes += mSize
	r.wait.Signal()
	r.mu.Unlock()
	return nil
}

// Dequeue pulls a message off the queue
// If there are no messages, it waits for one.
// If the buffer is closed, it will return immediately.
func (r *messageRing) Dequeue() (*Message, error) {
	r.mu.Lock()
	for len(r.queue) == 0 && !r.closed {
		r.wait.Wait()
	}

	if r.closed {
		r.mu.Unlock()
		return nil, errClosed
	}

	msg := r.queue[0]
	r.queue = r.queue[1:]
	r.sizeBytes -= int64(len(msg.Line))
	r.mu.Unlock()
	return msg, nil
}

var errClosed = errors.New("closed")

// Close closes the buffer ensuring no new messages can be added.
// Any callers waiting to dequeue a message will be woken up.
func (r *messageRing) Close() {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}

	r.closed = true
	r.wait.Broadcast()
	r.mu.Unlock()
}

// Drain drains all messages from the queue.
// This can be used after `Close()` to get any remaining messages that were in queue.
func (r *messageRing) Drain() []*Message {
	r.mu.Lock()
	ls := make([]*Message, 0, len(r.queue))
	ls = append(ls, r.queue...)
	r.sizeBytes = 0
	r.queue = r.queue[:0]
	r.mu.Unlock()
	return ls
}
