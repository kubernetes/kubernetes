/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"bytes"
	"fmt"
	"time"

	"github.com/golang/glog"
)

type traceStep struct {
	stepTime time.Time
	msg      string
}

type Trace struct {
	name      string
	startTime time.Time
	steps     []*traceStep
}

func NewTrace(name string) *Trace {
	return &Trace{name, time.Now(), make([]*traceStep, 0)}
}

func (t *Trace) Step(msg string) {
	t.steps = append(t.steps, &traceStep{time.Now(), msg})
}

func (t *Trace) Log() {
	endTime := time.Now()
	var buffer bytes.Buffer

	buffer.WriteString(fmt.Sprintf("Trace %q (started %v):\n", t.name, t.startTime))
	lastStepTime := t.startTime
	for _, step := range t.steps {
		buffer.WriteString(fmt.Sprintf("[%v] [%v] %v\n", step.stepTime.Sub(t.startTime), step.stepTime.Sub(lastStepTime), step.msg))
		lastStepTime = step.stepTime
	}
	buffer.WriteString(fmt.Sprintf("[%v] [%v] END\n", endTime.Sub(t.startTime), endTime.Sub(lastStepTime)))
	glog.Info(buffer.String())
}

func (t *Trace) LogIfLong(threshold time.Duration) {
	if time.Since(t.startTime) >= threshold {
		t.Log()
	}
}

func (t *Trace) TotalTime() time.Duration {
	return time.Since(t.startTime)
}
