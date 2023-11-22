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
//	logger := ktesting.NewLogger(...)
//	if testingLogger, ok := logger.GetSink().(ktesting.Underlier); ok {
//	    t := testingLogger.GetUnderlying()
//	    buffer := testingLogger.GetBuffer()
//	    text := buffer.String()
//	    log := buffer.Data()
//
// Serialization of the structured log parameters is done in the same way
// as for klog.InfoS.
package ktesting

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/internal/buffer"
	"k8s.io/klog/v2/internal/dbg"
	"k8s.io/klog/v2/internal/serialize"
	"k8s.io/klog/v2/internal/severity"
	"k8s.io/klog/v2/internal/verbosity"
)

// TL is the relevant subset of testing.TB.
type TL interface {
	Helper()
	Log(args ...interface{})
}

// NopTL implements TL with empty stubs. It can be used when only capturing
// output in memory is relevant.
type NopTL struct{}

func (n NopTL) Helper()            {}
func (n NopTL) Log(...interface{}) {}

var _ TL = NopTL{}

// BufferTL implements TL with an in-memory buffer.
type BufferTL struct {
	strings.Builder
}

func (n *BufferTL) Helper() {}
func (n *BufferTL) Log(args ...interface{}) {
	n.Builder.WriteString(fmt.Sprintln(args...))
}

var _ TL = &BufferTL{}

// NewLogger constructs a new logger for the given test interface.
//
// Beware that testing.T does not support logging after the test that
// it was created for has completed. If a test leaks goroutines
// and those goroutines log something after test completion,
// that output will be printed via the global klog logger with
// `<test name> leaked goroutine` as prefix.
//
// Verbosity can be modified at any time through the Config.V and
// Config.VModule API.
func NewLogger(t TL, c *Config) logr.Logger {
	l := tlogger{
		shared: &tloggerShared{
			t:      t,
			config: c,
		},
	}
	if c.co.anyToString != nil {
		l.shared.formatter.AnyToStringHook = c.co.anyToString
	}

	type testCleanup interface {
		Cleanup(func())
		Name() string
	}

	// Stopping the logging is optional and only done (and required)
	// for testing.T/B/F.
	if tb, ok := t.(testCleanup); ok {
		tb.Cleanup(l.shared.stop)
		l.shared.testName = tb.Name()
	}
	return logr.New(l)
}

// Buffer stores log entries as formatted text and structured data.
// It is safe to use this concurrently.
type Buffer interface {
	// String returns the log entries in a format that is similar to the
	// klog text output.
	String() string

	// Data returns the log entries as structs.
	Data() Log
}

// Log contains log entries in the order in which they were generated.
type Log []LogEntry

// DeepCopy returns a copy of the log. The error instance and key/value
// pairs remain shared.
func (l Log) DeepCopy() Log {
	log := make(Log, 0, len(l))
	log = append(log, l...)
	return log
}

// LogEntry represents all information captured for a log entry.
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
type LogType string

const (
	// LogError is the special value used for Error log entries.
	LogError = LogType("ERROR")

	// LogInfo is the special value used for Info log entries.
	LogInfo = LogType("INFO")
)

// Underlier is implemented by the LogSink of this logger. It provides access
// to additional APIs that are normally hidden behind the Logger API.
type Underlier interface {
	// GetUnderlying returns the testing instance that logging goes to.
	// It returns nil when the test has completed already.
	GetUnderlying() TL

	// GetBuffer grants access to the in-memory copy of the log entries.
	GetBuffer() Buffer
}

type logBuffer struct {
	mutex sync.Mutex
	text  strings.Builder
	log   Log
}

func (b *logBuffer) String() string {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	return b.text.String()
}

func (b *logBuffer) Data() Log {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	return b.log.DeepCopy()
}

// tloggerShared holds values that are the same for all LogSink instances. It
// gets referenced by pointer in the tlogger struct.
type tloggerShared struct {
	// mutex protects access to t.
	mutex sync.Mutex

	// t gets cleared when the test is completed.
	t TL

	// We warn once when a leaked goroutine is detected because
	// it logs after test completion.
	goroutineWarningDone bool

	formatter serialize.Formatter
	testName  string
	config    *Config
	buffer    logBuffer
	callDepth int
}

func (ls *tloggerShared) stop() {
	ls.mutex.Lock()
	defer ls.mutex.Unlock()
	ls.t = nil
}

// tlogger is the actual LogSink implementation.
type tlogger struct {
	shared *tloggerShared
	prefix string
	values []interface{}
}

func (l tlogger) fallbackLogger() logr.Logger {
	logger := klog.Background().WithValues(l.values...).WithName(l.shared.testName + " leaked goroutine")
	if l.prefix != "" {
		logger = logger.WithName(l.prefix)
	}
	// Skip direct caller (= Error or Info) plus the logr wrapper.
	logger = logger.WithCallDepth(l.shared.callDepth + 1)

	if !l.shared.goroutineWarningDone {
		logger.WithCallDepth(1).Error(nil, "WARNING: test kept at least one goroutine running after test completion", "callstack", string(dbg.Stacks(false)))
		l.shared.goroutineWarningDone = true
	}
	return logger
}

func (l tlogger) Init(info logr.RuntimeInfo) {
	l.shared.callDepth = info.CallDepth
}

func (l tlogger) GetCallStackHelper() func() {
	l.shared.mutex.Lock()
	defer l.shared.mutex.Unlock()
	if l.shared.t == nil {
		return func() {}
	}

	return l.shared.t.Helper
}

func (l tlogger) Info(level int, msg string, kvList ...interface{}) {
	l.shared.mutex.Lock()
	defer l.shared.mutex.Unlock()
	if l.shared.t == nil {
		l.fallbackLogger().V(level).Info(msg, kvList...)
		return
	}

	l.shared.t.Helper()
	buf := buffer.GetBuffer()
	l.shared.formatter.MergeAndFormatKVs(&buf.Buffer, l.values, kvList)
	l.log(LogInfo, msg, level, buf, nil, kvList)
}

func (l tlogger) Enabled(level int) bool {
	return l.shared.config.vstate.Enabled(verbosity.Level(level), 1)
}

func (l tlogger) Error(err error, msg string, kvList ...interface{}) {
	l.shared.mutex.Lock()
	defer l.shared.mutex.Unlock()
	if l.shared.t == nil {
		l.fallbackLogger().Error(err, msg, kvList...)
		return
	}

	l.shared.t.Helper()
	buf := buffer.GetBuffer()
	if err != nil {
		l.shared.formatter.KVFormat(&buf.Buffer, "err", err)
	}
	l.shared.formatter.MergeAndFormatKVs(&buf.Buffer, l.values, kvList)
	l.log(LogError, msg, 0, buf, err, kvList)
}

func (l tlogger) log(what LogType, msg string, level int, buf *buffer.Buffer, err error, kvList []interface{}) {
	l.shared.t.Helper()
	s := severity.InfoLog
	if what == LogError {
		s = severity.ErrorLog
	}
	args := []interface{}{buf.SprintHeader(s, time.Now())}
	if l.prefix != "" {
		args = append(args, l.prefix+":")
	}
	args = append(args, msg)
	if buf.Len() > 0 {
		// Skip leading space inserted by serialize.KVListFormat.
		args = append(args, string(buf.Bytes()[1:]))
	}
	l.shared.t.Log(args...)

	if !l.shared.config.co.bufferLogs {
		return
	}

	l.shared.buffer.mutex.Lock()
	defer l.shared.buffer.mutex.Unlock()

	// Store as text.
	l.shared.buffer.text.WriteString(string(what))
	for i := 1; i < len(args); i++ {
		l.shared.buffer.text.WriteByte(' ')
		l.shared.buffer.text.WriteString(args[i].(string))
	}
	lastArg := args[len(args)-1].(string)
	if lastArg[len(lastArg)-1] != '\n' {
		l.shared.buffer.text.WriteByte('\n')
	}

	// Store as raw data.
	l.shared.buffer.log = append(l.shared.buffer.log,
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
func (l tlogger) WithName(name string) logr.LogSink {
	if len(l.prefix) > 0 {
		l.prefix = l.prefix + "/"
	}
	l.prefix += name
	return l
}

func (l tlogger) WithValues(kvList ...interface{}) logr.LogSink {
	l.values = serialize.WithValues(l.values, kvList)
	return l
}

func (l tlogger) GetUnderlying() TL {
	return l.shared.t
}

func (l tlogger) GetBuffer() Buffer {
	return &l.shared.buffer
}

var _ logr.LogSink = &tlogger{}
var _ logr.CallStackHelperLogSink = &tlogger{}
var _ Underlier = &tlogger{}
