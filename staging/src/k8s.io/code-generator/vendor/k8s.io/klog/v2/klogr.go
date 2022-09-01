/*
Copyright 2021 The Kubernetes Authors.

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

package klog

import (
	"github.com/go-logr/logr"

	"k8s.io/klog/v2/internal/serialize"
)

// NewKlogr returns a logger that is functionally identical to
// klogr.NewWithOptions(klogr.FormatKlog), i.e. it passes through to klog. The
// difference is that it uses a simpler implementation.
func NewKlogr() Logger {
	return New(&klogger{})
}

// klogger is a subset of klogr/klogr.go. It had to be copied to break an
// import cycle (klogr wants to use klog, and klog wants to use klogr).
type klogger struct {
	level     int
	callDepth int
	prefix    string
	values    []interface{}
}

func (l *klogger) Init(info logr.RuntimeInfo) {
	l.callDepth += info.CallDepth
}

func (l klogger) Info(level int, msg string, kvList ...interface{}) {
	merged := serialize.MergeKVs(l.values, kvList)
	if l.prefix != "" {
		msg = l.prefix + ": " + msg
	}
	V(Level(level)).InfoSDepth(l.callDepth+1, msg, merged...)
}

func (l klogger) Enabled(level int) bool {
	return V(Level(level)).Enabled()
}

func (l klogger) Error(err error, msg string, kvList ...interface{}) {
	merged := serialize.MergeKVs(l.values, kvList)
	if l.prefix != "" {
		msg = l.prefix + ": " + msg
	}
	ErrorSDepth(l.callDepth+1, err, msg, merged...)
}

// WithName returns a new logr.Logger with the specified name appended.  klogr
// uses '/' characters to separate name elements.  Callers should not pass '/'
// in the provided name string, but this library does not actually enforce that.
func (l klogger) WithName(name string) logr.LogSink {
	if len(l.prefix) > 0 {
		l.prefix = l.prefix + "/"
	}
	l.prefix += name
	return &l
}

func (l klogger) WithValues(kvList ...interface{}) logr.LogSink {
	l.values = serialize.WithValues(l.values, kvList)
	return &l
}

func (l klogger) WithCallDepth(depth int) logr.LogSink {
	l.callDepth += depth
	return &l
}

var _ logr.LogSink = &klogger{}
var _ logr.CallDepthLogSink = &klogger{}
