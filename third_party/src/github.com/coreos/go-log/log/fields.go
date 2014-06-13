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
	"os"
	"path"
	"runtime"
	"strings"
	"sync/atomic"
	"time"
)

type Fields map[string]interface{}

func (logger *Logger) fieldValues() Fields {
	now := time.Now()
	fields := Fields{
		"prefix":     logger.prefix,               // static field available to all sinks
		"seq":        logger.nextSeq(),            // auto-incrementing sequence number
		"start_time": logger.created,              // start time of the logger
		"time":       now.Format(time.StampMilli), // formatted time of log entry
		"full_time":  now,                         // time of log entry
		"rtime":      time.Since(logger.created),  // relative time of log entry since started
		"pid":        os.Getpid(),                 // process id
		"executable": logger.executable,           // executable filename
	}

	if logger.verbose {
		setVerboseFields(fields)
	}
	return fields
}

func (logger *Logger) nextSeq() uint64 {
	return atomic.AddUint64(&logger.seq, 1)
}

func setVerboseFields(fields Fields) {
	callers := make([]uintptr, 10)
	n := runtime.Callers(3, callers) // starts in (*Logger).Log or similar
	callers = callers[:n]

	for _, pc := range callers {
		f := runtime.FuncForPC(pc)
		if !strings.Contains(f.Name(), "logger.(*Logger)") {
			fields["funcname"] = f.Name()
			pathname, lineno := f.FileLine(pc)
			fields["lineno"] = lineno
			fields["pathname"] = pathname
			fields["filename"] = path.Base(pathname)
			return
		}
	}
}
