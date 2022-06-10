/*
Copyright 2019 The Kubernetes Authors.
Copyright 2020 Intel Coporation.

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

// Package testinglogger contains an implementation of the logr interface
// which is logging through a function like testing.TB.Log function.
// Therefore it can be used in standard Go tests and Gingko test suites
// to ensure that output is associated with the currently running test.
//
// In addition, the log data is captured in a buffer and can be used by the
// test to verify that the code under test is logging as expected. To get
// access to that data, cast the LogSink into the Underlier type and retrieve
// it:
//
//    logger := ktesting.NewLogger(...)
//    if testingLogger, ok := logger.GetSink().(ktesting.Underlier); ok {
//        t := testingLogger.GetUnderlying()
//        buffer := testingLogger.GetBuffer()
//        text := buffer.String()
//        log := buffer.Data()
//
// Serialization of the structured log parameters is done in the same way
// as for klog.InfoS.
//
// Experimental
//
// Notice: This package is EXPERIMENTAL and may be changed or removed in a
// later release.
package ktesting

import (
	"bytes"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"

	"k8s.io/klog/v2/internal/serialize"
	"k8s.io/klog/v2/internal/verbosity"
)

// TL is the relevant subset of testing.TB.
//
// Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type TL interface {
	Helper()
	Log(args ...interface{})
}

// NopTL implements TL with empty stubs. It can be used when only capturing
// output in memory is relevant.
//
// Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type NopTL struct{}

func (n NopTL) Helper()                 {}
func (n NopTL) Log(args ...interface{}) {}

var _TL = NopTL{}

// NewLogger constructs a new logger for the given test interface.
//
// Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
func NewLogger(t TL, c *Config) logr.Logger {
	return logr.New(&tlogger{
		t:      t,
		prefix: "",
		values: nil,
		config: c,
		buffer: new(buffer),
	})
}

// Buffer stores log entries as formatted text and structured data.
// It is safe to use this concurrently.
//
// Experimental
//
// Notice: This interface is EXPERIMENTAL and may be changed or removed in a
// later release.
type Buffer interface {
	// String returns the log entries in a format that is similar to the
	// klog text output.
	String() string

	// Data returns the log entries as structs.
	Data() Log
}

// Log contains log entries in the order in which they were generated.
//
// Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type Log []LogEntry

// DeepCopy returns a copy of the log. The error instance and key/value
// pairs remain shared.
//
// Experimental
//
// Notice: This function is EXPERIMENTAL and may be changed or removed in a
// later release.
func (l Log) DeepCopy() Log {
	log := make(Log, 0, len(l))
	log = append(log, l...)
	return log
}

// LogEntry represents all information captured for a log entry.
//
// Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type LogEntry struct {
	// Timestamp stores the time when the log entry was created.
	Timestamp time.Time

	// Type is either LogInfo or LogError.
	Type LogType

	// Prefix contains the WithName strings concatenated with a slash.
	Prefix string

	// Message is the fixed log message string.
	Message string

	// Verbosity is always 0 for LogError.
	Verbosity int

	// Err is always nil for LogInfo. It may or may not be
	// nil for LogError.
	Err error

	// WithKVList are the concatenated key/value pairs from WithValues
	// calls. It's guaranteed to have an even number of entries because
	// the logger ensures that when WithValues is called.
	WithKVList []interface{}

	// ParameterKVList are the key/value pairs passed into the call,
	// without any validation.
	ParameterKVList []interface{}
}

// LogType determines whether a log entry was created with an Error or Info
// call.
//
// Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type LogType string

const (
	// LogError is the special value used for Error log entries.
	//
	// Experimental
	//
	// Notice: This value is EXPERIMENTAL and may be changed or removed in
	// a later release.
	LogError = LogType("ERROR")

	// LogInfo is the special value used for Info log entries.
	//
	// Experimental
	//
	// Notice: This value is EXPERIMENTAL and may be changed or removed in
	// a later release.
	LogInfo = LogType("INFO")
)

// Underlier is implemented by the LogSink of this logger. It provides access
// to additional APIs that are normally hidden behind the Logger API.
//
// Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type Underlier interface {
	// GetUnderlying returns the testing instance that logging goes to.
	GetUnderlying() TL

	// GetBuffer grants access to the in-memory copy of the log entries.
	GetBuffer() Buffer
}

type buffer struct {
	mutex sync.Mutex
	text  strings.Builder
	log   Log
}

func (b *buffer) String() string {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	return b.text.String()
}

func (b *buffer) Data() Log {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	return b.log.DeepCopy()
}

type tlogger struct {
	t      TL
	prefix string
	values []interface{}
	config *Config
	buffer *buffer
}

func (l *tlogger) Init(info logr.RuntimeInfo) {
}

func (l *tlogger) GetCallStackHelper() func() {
	return l.t.Helper
}

func (l *tlogger) Info(level int, msg string, kvList ...interface{}) {
	l.t.Helper()
	buffer := &bytes.Buffer{}
	merged := serialize.MergeKVs(l.values, kvList)
	serialize.KVListFormat(buffer, merged...)
	l.log(LogInfo, msg, level, buffer, nil, kvList)
}

func (l *tlogger) Enabled(level int) bool {
	return l.config.vstate.Enabled(verbosity.Level(level), 1)
}

func (l *tlogger) Error(err error, msg string, kvList ...interface{}) {
	l.t.Helper()
	buffer := &bytes.Buffer{}
	if err != nil {
		serialize.KVListFormat(buffer, "err", err)
	}
	merged := serialize.MergeKVs(l.values, kvList)
	serialize.KVListFormat(buffer, merged...)
	l.log(LogError, msg, 0, buffer, err, kvList)
}

func (l *tlogger) log(what LogType, msg string, level int, buffer *bytes.Buffer, err error, kvList []interface{}) {
	l.t.Helper()
	args := []interface{}{what}
	if l.prefix != "" {
		args = append(args, l.prefix+":")
	}
	args = append(args, msg)
	if buffer.Len() > 0 {
		// Skip leading space inserted by serialize.KVListFormat.
		args = append(args, string(buffer.Bytes()[1:]))
	}
	l.t.Log(args...)

	l.buffer.mutex.Lock()
	defer l.buffer.mutex.Unlock()

	// Store as text.
	l.buffer.text.WriteString(string(what))
	for i := 1; i < len(args); i++ {
		l.buffer.text.WriteByte(' ')
		l.buffer.text.WriteString(args[i].(string))
	}
	lastArg := args[len(args)-1].(string)
	if lastArg[len(lastArg)-1] != '\n' {
		l.buffer.text.WriteByte('\n')
	}

	// Store as raw data.
	l.buffer.log = append(l.buffer.log,
		LogEntry{
			Timestamp:       time.Now(),
			Type:            what,
			Prefix:          l.prefix,
			Message:         msg,
			Verbosity:       level,
			Err:             err,
			WithKVList:      l.values,
			ParameterKVList: kvList,
		},
	)
}

// WithName returns a new logr.Logger with the specified name appended.  klogr
// uses '/' characters to separate name elements.  Callers should not pass '/'
// in the provided name string, but this library does not actually enforce that.
func (l *tlogger) WithName(name string) logr.LogSink {
	new := *l
	if len(l.prefix) > 0 {
		new.prefix = l.prefix + "/"
	}
	new.prefix += name
	return &new
}

func (l *tlogger) WithValues(kvList ...interface{}) logr.LogSink {
	new := *l
	new.values = serialize.WithValues(l.values, kvList)
	return &new
}

func (l *tlogger) GetUnderlying() TL {
	return l.t
}

func (l *tlogger) GetBuffer() Buffer {
	return l.buffer
}

var _ logr.LogSink = &tlogger{}
var _ logr.CallStackHelperLogSink = &tlogger{}
var _ Underlier = &tlogger{}
