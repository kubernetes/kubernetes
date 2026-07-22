/*
Copyright 2026 The Kubernetes Authors.

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

package etcd3

import (
	"bytes"
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"k8s.io/apiserver/pkg/storage/etcd3/metrics"
	"k8s.io/klog/v2"
	utiltrace "k8s.io/utils/trace"
)

type locklessStep struct {
	stepTime   time.Time
	msg        string
	attributes []attribute.KeyValue
}

type locklessTraceLog struct {
	startTime  time.Time
	name       string
	traceID    string
	attributes []attribute.KeyValue
	steps      []locklessStep
}

func newLocklessTraceLog(ctx context.Context, name string, attributes ...attribute.KeyValue) *locklessTraceLog {
	traceID := ""
	if trace := utiltrace.FromContext(ctx); trace != nil {
		traceID = fmt.Sprintf("%d", trace.Num())
	}
	return &locklessTraceLog{
		startTime:  time.Now(),
		name:       name,
		traceID:    traceID,
		attributes: attributes,
		steps:      make([]locklessStep, 0, 3),
	}
}

func (t *locklessTraceLog) AddEvent(msg string, attributes ...attribute.KeyValue) {
	t.steps = append(t.steps, locklessStep{
		stepTime:   time.Now(),
		msg:        msg,
		attributes: attributes,
	})
}

func (t *locklessTraceLog) End(threshold time.Duration) {
	if !klog.V(2).Enabled() {
		return
	}
	endTime := time.Now()
	totalTime := endTime.Sub(t.startTime)
	if totalTime < threshold {
		return
	}

	var buffer bytes.Buffer
	buffer.WriteString(fmt.Sprintf("Trace[%s]: %q ", t.traceID, t.name))
	for i, attr := range t.attributes {
		buffer.WriteString(string(attr.Key))
		buffer.WriteString(":")
		buffer.WriteString(fmt.Sprintf("%v", attr.Value.AsInterface()))
		if i < len(t.attributes)-1 {
			buffer.WriteString(",")
		}
	}
	buffer.WriteString(" ")
	buffer.WriteString(fmt.Sprintf("(%s) (total time: %dms):", t.startTime.Format("02-Jan-2006 15:04:05.000"), totalTime.Milliseconds()))

	lastTime := t.startTime
	for _, step := range t.steps {
		duration := step.stepTime.Sub(lastTime)
		buffer.WriteString(fmt.Sprintf("\nTrace[%s]: ---%q ", t.traceID, step.msg))
		for i, attr := range step.attributes {
			buffer.WriteString(string(attr.Key))
			buffer.WriteString(":")
			buffer.WriteString(fmt.Sprintf("%v", attr.Value.AsInterface()))
			if i < len(step.attributes)-1 {
				buffer.WriteString(",")
			}
		}
		if len(step.attributes) > 0 {
			buffer.WriteString(" ")
		}
		buffer.WriteString(fmt.Sprintf("%dms (%s)", duration.Milliseconds(), step.stepTime.Format("15:04:05.000")))
		lastTime = step.stepTime
	}
	buffer.WriteString(fmt.Sprintf("\nTrace[%s]: [%v] [%v] END\n", t.traceID, endTime.Sub(t.startTime), totalTime))

	klogStartTime := time.Now()
	klog.Info(buffer.String())
	metrics.RecordLocklessTraceWriteDuration(time.Since(klogStartTime))
}
