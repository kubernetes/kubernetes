package log
// Copyright 2013, CoreOS, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// author: David Fisher <ddf1991@gmail.com>
// based on previous package by: Cong Ding <dinggnu@gmail.com>

import (
	"bitbucket.org/kardianos/osext"
	"os"
	"path"
	"time"
)

// Logger is user-immutable immutable struct which can log to several outputs
type Logger struct {
	sinks   []Sink // the sinks this logger will log to
	verbose bool   // gather expensive logging data?
	prefix  string // static field available to all log sinks under this logger

	created    time.Time // time when this logger was created
	seq        uint64    // sequential number of log message, starting at 1
	executable string    // executable name
}

// New creates a new Logger which logs to all the supplied sinks.  The prefix
// argument is passed to all loggers under the field "prefix" with every log
// message.  If verbose is true, more expensive runtime fields will be computed
// and passed to loggers.  These fields are funcname, lineno, pathname, and
// filename.
func New(prefix string, verbose bool, sinks ...Sink) *Logger {
	return &Logger{
		sinks:   sinks,
		verbose: verbose,
		prefix:  prefix,

		created:    time.Now(),
		seq:        0,
		executable: getExecutableName(),
	}
}

func getExecutableName() string {
	executablePath, err := osext.Executable()
	if err != nil {
		return "(UNKNOWN)"
	} else {
		return path.Base(executablePath)
	}
}

// NewSimple(sinks...) is equivalent to New("", false, sinks...)
func NewSimple(sinks ...Sink) *Logger {
	return New("", false, sinks...)
}

var defaultLogger *Logger

func init() {
	defaultLogger = NewSimple(CombinedSink(os.Stdout, BasicFormat, BasicFields))
}
