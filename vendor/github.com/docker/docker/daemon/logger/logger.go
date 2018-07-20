// Package logger defines interfaces that logger drivers implement to
// log messages.
//
// The other half of a logger driver is the implementation of the
// factory, which holds the contextual instance information that
// allows multiple loggers of the same type to perform different
// actions, such as logging to different locations.
package logger

import (
	"sync"
	"time"

	"github.com/docker/docker/api/types/backend"
)

// ErrReadLogsNotSupported is returned when the underlying log driver does not support reading
type ErrReadLogsNotSupported struct{}

func (ErrReadLogsNotSupported) Error() string {
	return "configured logging driver does not support reading"
}

// NotImplemented makes this error implement the `NotImplemented` interface from api/errdefs
func (ErrReadLogsNotSupported) NotImplemented() {}

const (
	logWatcherBufferSize = 4096
)

var messagePool = &sync.Pool{New: func() interface{} { return &Message{Line: make([]byte, 0, 256)} }}

// NewMessage returns a new message from the message sync.Pool
func NewMessage() *Message {
	return messagePool.Get().(*Message)
}

// PutMessage puts the specified message back n the message pool.
// The message fields are reset before putting into the pool.
func PutMessage(msg *Message) {
	msg.reset()
	messagePool.Put(msg)
}

// Message is datastructure that represents piece of output produced by some
// container.  The Line member is a slice of an array whose contents can be
// changed after a log driver's Log() method returns.
//
// Message is subtyped from backend.LogMessage because there is a lot of
// internal complexity around the Message type that should not be exposed
// to any package not explicitly importing the logger type.
//
// Any changes made to this struct must also be updated in the `reset` function
type Message backend.LogMessage

// reset sets the message back to default values
// This is used when putting a message back into the message pool.
// Any changes to the `Message` struct should be reflected here.
func (m *Message) reset() {
	m.Line = m.Line[:0]
	m.Source = ""
	m.Attrs = nil
	m.Partial = false

	m.Err = nil
}

// AsLogMessage returns a pointer to the message as a pointer to
// backend.LogMessage, which is an identical type with a different purpose
func (m *Message) AsLogMessage() *backend.LogMessage {
	return (*backend.LogMessage)(m)
}

// Logger is the interface for docker logging drivers.
type Logger interface {
	Log(*Message) error
	Name() string
	Close() error
}

// SizedLogger is the interface for logging drivers that can control
// the size of buffer used for their messages.
type SizedLogger interface {
	Logger
	BufSize() int
}

// ReadConfig is the configuration passed into ReadLogs.
type ReadConfig struct {
	Since  time.Time
	Until  time.Time
	Tail   int
	Follow bool
}

// LogReader is the interface for reading log messages for loggers that support reading.
type LogReader interface {
	// Read logs from underlying logging backend
	ReadLogs(ReadConfig) *LogWatcher
}

// LogWatcher is used when consuming logs read from the LogReader interface.
type LogWatcher struct {
	// For sending log messages to a reader.
	Msg chan *Message
	// For sending error messages that occur while while reading logs.
	Err           chan error
	closeOnce     sync.Once
	closeNotifier chan struct{}
}

// NewLogWatcher returns a new LogWatcher.
func NewLogWatcher() *LogWatcher {
	return &LogWatcher{
		Msg:           make(chan *Message, logWatcherBufferSize),
		Err:           make(chan error, 1),
		closeNotifier: make(chan struct{}),
	}
}

// Close notifies the underlying log reader to stop.
func (w *LogWatcher) Close() {
	// only close if not already closed
	w.closeOnce.Do(func() {
		close(w.closeNotifier)
	})
}

// WatchClose returns a channel receiver that receives notification
// when the watcher has been closed. This should only be called from
// one goroutine.
func (w *LogWatcher) WatchClose() <-chan struct{} {
	return w.closeNotifier
}

// Capability defines the list of capabilties that a driver can implement
// These capabilities are not required to be a logging driver, however do
// determine how a logging driver can be used
type Capability struct {
	// Determines if a log driver can read back logs
	ReadLogs bool
}

// MarshalFunc is a func that marshals a message into an arbitrary format
type MarshalFunc func(*Message) ([]byte, error)
