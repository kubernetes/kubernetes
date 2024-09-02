/*
Copyright 2019 The Kubernetes Authors.
Copyright 2020 Intel Corporation.

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

// Package textlogger contains an implementation of the logr interface which is
// producing the exact same output as klog. It does not route output through
// klog (i.e. ignores [k8s.io/klog/v2.InitFlags]). Instead, all settings must be
// configured through its own [NewConfig] and [Config.AddFlags].
package textlogger

import (
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"

	"k8s.io/klog/v2/internal/buffer"
	"k8s.io/klog/v2/internal/serialize"
	"k8s.io/klog/v2/internal/severity"
	"k8s.io/klog/v2/internal/verbosity"
)

var (
	// TimeNow is used to retrieve the current time. May be changed for testing.
	TimeNow = time.Now
)

const (
	// nameKey is used to log the `WithName` values as an additional attribute.
	nameKey = "logger"
)

// NewLogger constructs a new logger.
//
// Verbosity can be modified at any time through the Config.V and
// Config.VModule API.
func NewLogger(c *Config) logr.Logger {
	return logr.New(&tlogger{
		values: nil,
		config: c,
	})
}

type tlogger struct {
	callDepth int

	// hasPrefix is true if the first entry in values is the special
	// nameKey key/value. Such an entry gets added and later updated in
	// WithName.
	hasPrefix bool

	values []interface{}
	groups string
	config *Config
}

func (l *tlogger) Init(info logr.RuntimeInfo) {
	l.callDepth = info.CallDepth
}

func (l *tlogger) WithCallDepth(depth int) logr.LogSink {
	newLogger := *l
	newLogger.callDepth += depth
	return &newLogger
}

func (l *tlogger) Enabled(level int) bool {
	return l.config.vstate.Enabled(verbosity.Level(level), 1+l.callDepth)
}

func (l *tlogger) Info(_ int, msg string, kvList ...interface{}) {
	l.print(nil, severity.InfoLog, msg, kvList)
}

func (l *tlogger) Error(err error, msg string, kvList ...interface{}) {
	l.print(err, severity.ErrorLog, msg, kvList)
}

func (l *tlogger) print(err error, s severity.Severity, msg string, kvList []interface{}) {
	// Determine caller.
	// +1 for this frame, +1 for Info/Error.
	skip := l.callDepth + 2
	file, line := l.config.co.unwind(skip)
	if file == "" {
		file = "???"
		line = 1
	} else if slash := strings.LastIndex(file, "/"); slash >= 0 {
		file = file[slash+1:]
	}
	l.printWithInfos(file, line, time.Now(), err, s, msg, kvList)
}

func runtimeBacktrace(skip int) (string, int) {
	_, file, line, ok := runtime.Caller(skip + 1)
	if !ok {
		return "", 0
	}
	return file, line
}

func (l *tlogger) printWithInfos(file string, line int, now time.Time, err error, s severity.Severity, msg string, kvList []interface{}) {
	// Only create a new buffer if we don't have one cached.
	b := buffer.GetBuffer()
	defer buffer.PutBuffer(b)

	// Format header.
	if l.config.co.fixedTime != nil {
		now = *l.config.co.fixedTime
	}
	b.FormatHeader(s, file, line, now)

	// The message is always quoted, even if it contains line breaks.
	// If developers want multi-line output, they should use a small, fixed
	// message and put the multi-line output into a value.
	b.WriteString(strconv.Quote(msg))
	if err != nil {
		serialize.KVFormat(&b.Buffer, "err", err)
	}
	serialize.MergeAndFormatKVs(&b.Buffer, l.values, kvList)
	if b.Len() == 0 || b.Bytes()[b.Len()-1] != '\n' {
		b.WriteByte('\n')
	}
	_, _ = l.config.co.output.Write(b.Bytes())
}

func (l *tlogger) WriteKlogBuffer(data []byte) {
	_, _ = l.config.co.output.Write(data)
}

// WithName returns a new logr.Logger with the specified name appended.  klogr
// uses '/' characters to separate name elements.  Callers should not pass '/'
// in the provided name string, but this library does not actually enforce that.
func (l *tlogger) WithName(name string) logr.LogSink {
	clone := *l
	if l.hasPrefix {
		// Copy slice and modify value. No length checks and type
		// assertions are needed because hasPrefix is only true if the
		// first two elements exist and are key/value strings.
		v := make([]interface{}, 0, len(l.values))
		v = append(v, l.values...)
		prefix, _ := v[1].(string)
		v[1] = prefix + "." + name
		clone.values = v
	} else {
		// Preprend new key/value pair.
		v := make([]interface{}, 0, 2+len(l.values))
		v = append(v, nameKey, name)
		v = append(v, l.values...)
		clone.values = v
		clone.hasPrefix = true
	}
	return &clone
}

func (l *tlogger) WithValues(kvList ...interface{}) logr.LogSink {
	clone := *l
	clone.values = serialize.WithValues(l.values, kvList)
	return &clone
}

// KlogBufferWriter is implemented by the textlogger LogSink.
type KlogBufferWriter interface {
	// WriteKlogBuffer takes a pre-formatted buffer prepared by klog and
	// writes it unchanged to the output stream. Can be used with
	// klog.WriteKlogBuffer when setting a logger through
	// klog.SetLoggerWithOptions.
	WriteKlogBuffer([]byte)
}

var _ logr.LogSink = &tlogger{}
var _ logr.CallDepthLogSink = &tlogger{}
var _ KlogBufferWriter = &tlogger{}
