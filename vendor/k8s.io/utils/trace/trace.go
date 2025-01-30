/*
Copyright 2015 The Kubernetes Authors.

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

package trace

import (
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

var klogV = func(lvl klog.Level) bool {
	return klog.V(lvl).Enabled()
}

// Field is a key value pair that provides additional details about the trace.
type Field struct {
	Key   string
	Value interface{}
}

func (f Field) format() string {
	return fmt.Sprintf("%s:%v", f.Key, f.Value)
}

func writeFields(b *bytes.Buffer, l []Field) {
	for i, f := range l {
		b.WriteString(f.format())
		if i < len(l)-1 {
			b.WriteString(",")
		}
	}
}

func writeTraceItemSummary(b *bytes.Buffer, msg string, totalTime time.Duration, startTime time.Time, fields []Field) {
	b.WriteString(fmt.Sprintf("%q ", msg))
	if len(fields) > 0 {
		writeFields(b, fields)
		b.WriteString(" ")
	}

	b.WriteString(fmt.Sprintf("%vms (%v)", durationToMilliseconds(totalTime), startTime.Format("15:04:05.000")))
}

func durationToMilliseconds(timeDuration time.Duration) int64 {
	return timeDuration.Nanoseconds() / 1e6
}

type traceItem interface {
	// rLock must be called before invoking time or writeItem.
	rLock()
	// rUnlock must be called after processing the item is complete.
	rUnlock()

	// time returns when the trace was recorded as completed.
	time() time.Time
	// writeItem outputs the traceItem to the buffer. If stepThreshold is non-nil, only output the
	// traceItem if its the duration exceeds the stepThreshold.
	// Each line of output is prefixed by formatter to visually indent nested items.
	writeItem(b *bytes.Buffer, formatter string, startTime time.Time, stepThreshold *time.Duration)
}

type traceStep struct {
	stepTime time.Time
	msg      string
	fields   []Field
}

// rLock doesn't need to do anything because traceStep instances are immutable.
func (s traceStep) rLock()   {}
func (s traceStep) rUnlock() {}

func (s traceStep) time() time.Time {
	return s.stepTime
}

func (s traceStep) writeItem(b *bytes.Buffer, formatter string, startTime time.Time, stepThreshold *time.Duration) {
	stepDuration := s.stepTime.Sub(startTime)
	if stepThreshold == nil || *stepThreshold == 0 || stepDuration >= *stepThreshold || klogV(4) {
		b.WriteString(fmt.Sprintf("%s---", formatter))
		writeTraceItemSummary(b, s.msg, stepDuration, s.stepTime, s.fields)
	}
}

// Trace keeps track of a set of "steps" and allows us to log a specific
// step if it took longer than its share of the total allowed time
type Trace struct {
	// constant fields
	name        string
	fields      []Field
	startTime   time.Time
	parentTrace *Trace
	// fields guarded by a lock
	lock       sync.RWMutex
	threshold  *time.Duration
	endTime    *time.Time
	traceItems []traceItem
}

func (t *Trace) rLock() {
	t.lock.RLock()
}

func (t *Trace) rUnlock() {
	t.lock.RUnlock()
}

func (t *Trace) time() time.Time {
	if t.endTime != nil {
		return *t.endTime
	}
	return t.startTime // if the trace is incomplete, don't assume an end time
}

func (t *Trace) writeItem(b *bytes.Buffer, formatter string, startTime time.Time, stepThreshold *time.Duration) {
	if t.durationIsWithinThreshold() || klogV(4) {
		b.WriteString(fmt.Sprintf("%v[", formatter))
		writeTraceItemSummary(b, t.name, t.TotalTime(), t.startTime, t.fields)
		if st := t.calculateStepThreshold(); st != nil {
			stepThreshold = st
		}
		t.writeTraceSteps(b, formatter+" ", stepThreshold)
		b.WriteString("]")
		return
	}
	// If the trace should not be written, still check for nested traces that should be written
	for _, s := range t.traceItems {
		if nestedTrace, ok := s.(*Trace); ok {
			nestedTrace.writeItem(b, formatter, startTime, stepThreshold)
		}
	}
}

// New creates a Trace with the specified name. The name identifies the operation to be traced. The
// Fields add key value pairs to provide additional details about the trace, such as operation inputs.
func New(name string, fields ...Field) *Trace {
	return &Trace{name: name, startTime: time.Now(), fields: fields}
}

// Step adds a new step with a specific message. Call this at the end of an execution step to record
// how long it took. The Fields add key value pairs to provide additional details about the trace
// step.
func (t *Trace) Step(msg string, fields ...Field) {
	t.lock.Lock()
	defer t.lock.Unlock()
	if t.traceItems == nil {
		// traces almost always have less than 6 steps, do this to avoid more than a single allocation
		t.traceItems = make([]traceItem, 0, 6)
	}
	t.traceItems = append(t.traceItems, traceStep{stepTime: time.Now(), msg: msg, fields: fields})
}

// Nest adds a nested trace with the given message and fields and returns it.
// As a convenience, if the receiver is nil, returns a top level trace. This allows
// one to call FromContext(ctx).Nest without having to check if the trace
// in the context is nil.
func (t *Trace) Nest(msg string, fields ...Field) *Trace {
	newTrace := New(msg, fields...)
	if t != nil {
		newTrace.parentTrace = t
		t.lock.Lock()
		t.traceItems = append(t.traceItems, newTrace)
		t.lock.Unlock()
	}
	return newTrace
}

// Log is used to dump all the steps in the Trace. It also logs the nested trace messages using indentation.
// If the Trace is nested it is not immediately logged. Instead, it is logged when the trace it is nested within
// is logged.
func (t *Trace) Log() {
	endTime := time.Now()
	t.lock.Lock()
	t.endTime = &endTime
	t.lock.Unlock()
	// an explicit logging request should dump all the steps out at the higher level
	if t.parentTrace == nil && klogV(2) { // We don't start logging until Log or LogIfLong is called on the root trace
		t.logTrace()
	}
}

// LogIfLong only logs the trace if the duration of the trace exceeds the threshold.
// Only steps that took longer than their share or the given threshold are logged.
// If klog is at verbosity level 4 or higher and the trace took longer than the threshold,
// all substeps and subtraces are logged. Otherwise, only those which took longer than
// their own threshold.
// If the Trace is nested it is not immediately logged. Instead, it is logged when the trace it
// is nested within is logged.
func (t *Trace) LogIfLong(threshold time.Duration) {
	t.lock.Lock()
	t.threshold = &threshold
	t.lock.Unlock()
	t.Log()
}

// logTopLevelTraces finds all traces in a hierarchy of nested traces that should be logged but do not have any
// parents that will be logged, due to threshold limits, and logs them as top level traces.
func (t *Trace) logTrace() {
	t.lock.RLock()
	defer t.lock.RUnlock()
	if t.durationIsWithinThreshold() {
		var buffer bytes.Buffer
		traceNum := rand.Int31()

		totalTime := t.endTime.Sub(t.startTime)
		buffer.WriteString(fmt.Sprintf("Trace[%d]: %q ", traceNum, t.name))
		if len(t.fields) > 0 {
			writeFields(&buffer, t.fields)
			buffer.WriteString(" ")
		}

		// if any step took more than it's share of the total allowed time, it deserves a higher log level
		buffer.WriteString(fmt.Sprintf("(%v) (total time: %vms):", t.startTime.Format("02-Jan-2006 15:04:05.000"), totalTime.Milliseconds()))
		stepThreshold := t.calculateStepThreshold()
		t.writeTraceSteps(&buffer, fmt.Sprintf("\nTrace[%d]: ", traceNum), stepThreshold)
		buffer.WriteString(fmt.Sprintf("\nTrace[%d]: [%v] [%v] END\n", traceNum, t.endTime.Sub(t.startTime), totalTime))

		klog.Info(buffer.String())
		return
	}

	// If the trace should not be logged, still check if nested traces should be logged
	for _, s := range t.traceItems {
		if nestedTrace, ok := s.(*Trace); ok {
			nestedTrace.logTrace()
		}
	}
}

func (t *Trace) writeTraceSteps(b *bytes.Buffer, formatter string, stepThreshold *time.Duration) {
	lastStepTime := t.startTime
	for _, stepOrTrace := range t.traceItems {
		stepOrTrace.rLock()
		stepOrTrace.writeItem(b, formatter, lastStepTime, stepThreshold)
		lastStepTime = stepOrTrace.time()
		stepOrTrace.rUnlock()
	}
}

func (t *Trace) durationIsWithinThreshold() bool {
	if t.endTime == nil { // we don't assume incomplete traces meet the threshold
		return false
	}
	return t.threshold == nil || *t.threshold == 0 || t.endTime.Sub(t.startTime) >= *t.threshold
}

// TotalTime can be used to figure out how long it took since the Trace was created
func (t *Trace) TotalTime() time.Duration {
	return time.Since(t.startTime)
}

// calculateStepThreshold returns a threshold for the individual steps of a trace, or nil if there is no threshold and
// all steps should be written.
func (t *Trace) calculateStepThreshold() *time.Duration {
	if t.threshold == nil {
		return nil
	}
	lenTrace := len(t.traceItems) + 1
	traceThreshold := *t.threshold
	for _, s := range t.traceItems {
		nestedTrace, ok := s.(*Trace)
		if ok {
			nestedTrace.lock.RLock()
			if nestedTrace.threshold != nil {
				traceThreshold = traceThreshold - *nestedTrace.threshold
				lenTrace--
			}
			nestedTrace.lock.RUnlock()
		}
	}

	// the limit threshold is used when the threshold(
	//remaining after subtracting that of the child trace) is getting very close to zero to prevent unnecessary logging
	limitThreshold := *t.threshold / 4
	if traceThreshold < limitThreshold {
		traceThreshold = limitThreshold
		lenTrace = len(t.traceItems) + 1
	}

	stepThreshold := traceThreshold / time.Duration(lenTrace)
	return &stepThreshold
}

// ContextTraceKey provides a common key for traces in context.Context values.
type ContextTraceKey struct{}

// FromContext returns the trace keyed by ContextTraceKey in the context values, if one
// is present, or nil If there is no trace in the Context.
// It is safe to call Nest() on the returned value even if it is nil because ((*Trace)nil).Nest returns a top level
// trace.
func FromContext(ctx context.Context) *Trace {
	if v, ok := ctx.Value(ContextTraceKey{}).(*Trace); ok {
		return v
	}
	return nil
}

// ContextWithTrace returns a context with trace included in the context values, keyed by ContextTraceKey.
func ContextWithTrace(ctx context.Context, trace *Trace) context.Context {
	return context.WithValue(ctx, ContextTraceKey{}, trace)
}
