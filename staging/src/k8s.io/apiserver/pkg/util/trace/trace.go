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
	"fmt"
	"math/rand"
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
	steps     []traceStep
}

func New(name string) *Trace {
	return &Trace{name, time.Now(), nil}
}

func (t *Trace) Step(msg string) {
	if t.steps == nil {
		// traces almost always have less than 6 steps, do this to avoid more than a single allocation
		t.steps = make([]traceStep, 0, 6)
	}
	t.steps = append(t.steps, traceStep{time.Now(), msg})
}

func (t *Trace) Log() {
	// an explicit logging request should dump all the steps out at the higher level
	t.logWithStepThreshold(0)
}

func (t *Trace) logWithStepThreshold(stepThreshold time.Duration) {
	var buffer bytes.Buffer
	tracenum := rand.Int31()
	endTime := time.Now()

	totalTime := endTime.Sub(t.startTime)
	buffer.WriteString(fmt.Sprintf("Trace[%d]: %q (started: %v) (total time: %v):\n", tracenum, t.name, t.startTime, totalTime))
	lastStepTime := t.startTime
	for _, step := range t.steps {
		stepDuration := step.stepTime.Sub(lastStepTime)
		if stepThreshold == 0 || stepDuration > stepThreshold || glog.V(4) {
			buffer.WriteString(fmt.Sprintf("Trace[%d]: [%v] [%v] %v\n", tracenum, step.stepTime.Sub(t.startTime), stepDuration, step.msg))
		}
		lastStepTime = step.stepTime
	}
	stepDuration := endTime.Sub(lastStepTime)
	if stepThreshold == 0 || stepDuration > stepThreshold || glog.V(4) {
		buffer.WriteString(fmt.Sprintf("Trace[%d]: [%v] [%v] END\n", tracenum, endTime.Sub(t.startTime), stepDuration))
	}

	glog.Info(buffer.String())
}

func (t *Trace) LogIfLong(threshold time.Duration) {
	if time.Since(t.startTime) >= threshold {
		// if any step took more than it's share of the total allowed time, it deserves a higher log level
		stepThreshold := threshold / time.Duration(len(t.steps)+1)
		t.logWithStepThreshold(stepThreshold)
	}
}

func (t *Trace) TotalTime() time.Duration {
	return time.Since(t.startTime)
}
