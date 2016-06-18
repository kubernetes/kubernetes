// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package capnslog

import (
	"bufio"
	"fmt"
	"io"
	"runtime"
	"strings"
	"time"
)

type Formatter interface {
	Format(pkg string, level LogLevel, depth int, entries ...interface{})
	Flush()
}

func NewStringFormatter(w io.Writer) *StringFormatter {
	return &StringFormatter{
		w: bufio.NewWriter(w),
	}
}

type StringFormatter struct {
	w *bufio.Writer
}

func (s *StringFormatter) Format(pkg string, l LogLevel, i int, entries ...interface{}) {
	now := time.Now().UTC()
	s.w.WriteString(now.Format(time.RFC3339))
	s.w.WriteByte(' ')
	writeEntries(s.w, pkg, l, i, entries...)
	s.Flush()
}

func writeEntries(w *bufio.Writer, pkg string, _ LogLevel, _ int, entries ...interface{}) {
	if pkg != "" {
		w.WriteString(pkg + ": ")
	}
	str := fmt.Sprint(entries...)
	endsInNL := strings.HasSuffix(str, "\n")
	w.WriteString(str)
	if !endsInNL {
		w.WriteString("\n")
	}
}

func (s *StringFormatter) Flush() {
	s.w.Flush()
}

func NewPrettyFormatter(w io.Writer, debug bool) Formatter {
	return &PrettyFormatter{
		w:     bufio.NewWriter(w),
		debug: debug,
	}
}

type PrettyFormatter struct {
	w     *bufio.Writer
	debug bool
}

func (c *PrettyFormatter) Format(pkg string, l LogLevel, depth int, entries ...interface{}) {
	now := time.Now()
	ts := now.Format("2006-01-02 15:04:05")
	c.w.WriteString(ts)
	ms := now.Nanosecond() / 1000
	c.w.WriteString(fmt.Sprintf(".%06d", ms))
	if c.debug {
		_, file, line, ok := runtime.Caller(depth) // It's always the same number of frames to the user's call.
		if !ok {
			file = "???"
			line = 1
		} else {
			slash := strings.LastIndex(file, "/")
			if slash >= 0 {
				file = file[slash+1:]
			}
		}
		if line < 0 {
			line = 0 // not a real line number
		}
		c.w.WriteString(fmt.Sprintf(" [%s:%d]", file, line))
	}
	c.w.WriteString(fmt.Sprint(" ", l.Char(), " | "))
	writeEntries(c.w, pkg, l, depth, entries...)
	c.Flush()
}

func (c *PrettyFormatter) Flush() {
	c.w.Flush()
}
