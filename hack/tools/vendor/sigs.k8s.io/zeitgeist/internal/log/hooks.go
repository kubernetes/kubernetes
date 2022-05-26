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

package log

import (
	"fmt"
	"runtime"
	"strings"

	"github.com/sirupsen/logrus"
)

type FileNameHook struct {
	field      string
	skipPrefix []string
	formatter  logrus.Formatter
	Formatter  func(file, function string, line int) string
}

type wrapper struct {
	old  logrus.Formatter
	hook *FileNameHook
}

// NewFilenameHook creates a new default FileNameHook
func NewFilenameHook() *FileNameHook {
	return &FileNameHook{
		field:      "file",
		skipPrefix: []string{"log/", "logrus/", "logrus@"},
		Formatter: func(file, function string, line int) string {
			return fmt.Sprintf("%s:%d", file, line)
		},
	}
}

// Levels returns the levels for which the hook is activated. This contains
// currently only the DebugLevel
func (f *FileNameHook) Levels() []logrus.Level {
	return []logrus.Level{logrus.DebugLevel}
}

// Fire executes the hook for every logrus entry
func (f *FileNameHook) Fire(entry *logrus.Entry) error {
	if f.formatter != entry.Logger.Formatter {
		f.formatter = &wrapper{entry.Logger.Formatter, f}
	}
	entry.Logger.Formatter = f.formatter
	return nil
}

// Format returns the log format including the caller as field
func (w *wrapper) Format(entry *logrus.Entry) ([]byte, error) {
	field := entry.WithField(
		w.hook.field,
		w.hook.Formatter(w.hook.findCaller()),
	)
	field.Level = entry.Level
	field.Message = entry.Message
	return w.old.Format(field)
}

// findCaller returns the file, function and line number for the current call
func (f *FileNameHook) findCaller() (file, function string, line int) {
	var pc uintptr
	// The maximum amount of frames to be iterated
	const maxFrames = 10
	for i := 0; i < maxFrames; i++ {
		// The amount of frames to be skipped to land at the actual caller
		const skipFrames = 5
		pc, file, line = caller(skipFrames + i)
		if !f.shouldSkipPrefix(file) {
			break
		}
	}
	if pc != 0 {
		frames := runtime.CallersFrames([]uintptr{pc})
		frame, _ := frames.Next()
		function = frame.Function
	}

	return file, function, line
}

// caller reports file and line number information about function invocations
// on the calling goroutine's stack. The argument skip is the number of stack
// frames to ascend, with 0 identifying the caller of Caller.
func caller(skip int) (pc uintptr, file string, line int) {
	ok := false
	pc, file, line, ok = runtime.Caller(skip)
	if !ok {
		return 0, "", 0
	}

	n := 0
	for i := len(file) - 1; i > 0; i-- {
		if file[i] == '/' {
			n++
			if n >= 2 {
				file = file[i+1:]
				break
			}
		}
	}

	return pc, file, line
}

// shouldSkipPrefix returns true if the hook should be skipped, otherwise false
func (f *FileNameHook) shouldSkipPrefix(file string) bool {
	for i := range f.skipPrefix {
		if strings.HasPrefix(file, f.skipPrefix[i]) {
			return true
		}
	}
	return false
}
