/*
Copyright 2022 The Kubernetes Authors.

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

package tracelog

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

// TraceLog keeps track of a set of log "modules" and allows us to log
// with a specific frequency
// TODO: move this package to kubernetes/utils
type TraceLog struct {
	name            string
	lock            sync.RWMutex
	traceLogModules map[string]*time.Time
	frequency       time.Duration
	logger          func(msg string, keysAndValues ...interface{})
}

// NewTraceLog creates a TraceLog with the specified name. The name identifies the operation to be traced.
// The frequency identifies how often the logs should output.
func NewTraceLog(name string, frequency time.Duration) *TraceLog {
	return &TraceLog{
		name:            name,
		traceLogModules: make(map[string]*time.Time),
		frequency:       frequency,
		logger:          klog.InfoS,
	}
}

// LogWithFrequency is used to log higher verbosity with a slow frequency,
// which can make sure that logs are not too verbose, while allowing
// troubleshooting without changing verbosity to higher level.
func (t *TraceLog) LogWithFrequency(module string, msg string, keysAndValues ...interface{}) {
	now := time.Now()
	lastLogTime := t.getLastLogTime(module)
	if lastLogTime == nil || now.Sub(*lastLogTime) >= t.frequency {
		t.logger(fmt.Sprintf("Trace[%s][%s]: %s", t.name, module, msg), keysAndValues...)
		t.lock.Lock()
		defer t.lock.Unlock()
		t.traceLogModules[module] = &now
	}
}

// CleanLogModule delete the module of trace log
func (t *TraceLog) CleanLogModule(module string) {
	t.lock.Lock()
	defer t.lock.Unlock()
	delete(t.traceLogModules, module)
}

// getLastLogTime get the module of trace log
func (t *TraceLog) getLastLogTime(module string) *time.Time {
	t.lock.RLock()
	defer t.lock.RUnlock()
	return t.traceLogModules[module]
}
