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
	now := time.Now()
	y, m, d := now.Date()
	h, min, sec := now.Clock()
	s.w.WriteString(fmt.Sprintf("%d/%02d/%d %02d:%02d:%02d ", y, m, d, h, min, sec))
	s.writeEntries(pkg, l, i, entries...)
}

func (s *StringFormatter) writeEntries(pkg string, _ LogLevel, _ int, entries ...interface{}) {
	if pkg != "" {
		s.w.WriteString(pkg + ": ")
	}
	str := fmt.Sprint(entries...)
	endsInNL := strings.HasSuffix(str, "\n")
	s.w.WriteString(str)
	if !endsInNL {
		s.w.WriteString("\n")
	}
	s.Flush()
}

func (s *StringFormatter) Flush() {
	s.w.Flush()
}
