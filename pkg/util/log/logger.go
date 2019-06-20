/*
Copyright 2019 The Kubernetes Authors.

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

package log

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/klog"
)

type logEntry struct {
	count       int
	countLogged int
	timestamp   time.Time
}

// Verbose is a boolean type that implements Infof (like Printf) etc.
// See the documentation of V for more information.
type Verbose klog.Verbose

var messages = make(map[string]*logEntry)
var lock = sync.RWMutex{}

func init() {
	go cleanupOldMessagesPeriodically()
}

// V reports whether verbosity at the call site is at least the requested level.
// See the documentation of klog.V for usage.
func V(level klog.Level) Verbose {
	return Verbose(klog.V(level))
}

// InfoInfreqf is equivalent to the global Infof function, guarded by the value of v,
// but identical log messages will not be logged if sooner than an reasonable time window.
// That time window increases each time the message actually gets logged.
// See the documentation of klog.V for usage.
func (v Verbose) InfoInfreqf(format string, args ...interface{}) {
	if v {
		message := fmt.Sprintf(format, args...)
		if message = getMessageInfrequently(message); message != "" {
			klog.Infof(message)
		}
	}
}

// ErrorInfreqf logs to the ERROR, WARNING, and INFO logs, but identical log messages
// will not be logged if sooner than an reasonable time window. That time window increases
// each time the message actually gets logged.
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func ErrorInfreqf(format string, args ...interface{}) {
	message := fmt.Sprintf(format, args...)
	if message = getMessageInfrequently(message); message != "" {
		klog.Errorf(message)
	}
}

func getMessageInfrequently(message string) string {
	entry, ok := messages[message]
	if !ok {
		lock.Lock()
		defer lock.Unlock()
		messages[message] = &logEntry{count: 1, countLogged: 1, timestamp: time.Now()}
		return message
	}

	entry.count++
	now := time.Now()
	if delta := now.Sub(entry.timestamp); delta >= 5*time.Minute || delta >= time.Duration(entry.countLogged*10)*time.Second {
		entry.countLogged++
		entry.timestamp = now
		return fmt.Sprintf("(x%d) %s", entry.count, message)
	}
	return ""
}

func cleanupOldMessagesPeriodically() {
	for {
		cleanOldMessages()
		time.Sleep(10 * time.Minute)
	}
}

func cleanOldMessages() {
	now := time.Now()
	lock.Lock()
	defer lock.Unlock()
	for key, entry := range messages {
		if delta := now.Sub(entry.timestamp); delta >= 10*time.Minute {
			delete(messages, key)
		}
	}
}
