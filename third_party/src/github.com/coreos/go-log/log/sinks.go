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
	"fmt"
	"io"
	"sync"
)

const AsyncBuffer = 100

type Sink interface {
	Log(Fields)
}

type nullSink struct{}

func (sink *nullSink) Log(fields Fields) {}

func NullSink() Sink {
	return &nullSink{}
}

type writerSink struct {
	lock   sync.Mutex
	out    io.Writer
	format string
	fields []string
}

func (sink *writerSink) Log(fields Fields) {
	vals := make([]interface{}, len(sink.fields))
	for i, field := range sink.fields {
		var ok bool
		vals[i], ok = fields[field]
		if !ok {
			vals[i] = "???"
		}
	}

	sink.lock.Lock()
	defer sink.lock.Unlock()
	fmt.Fprintf(sink.out, sink.format, vals...)
}

func WriterSink(out io.Writer, format string, fields []string) Sink {
	return &writerSink{
		out:    out,
		format: format,
		fields: fields,
	}
}

type combinedSink struct {
	sinks []Sink
}

func (sink *combinedSink) Log(fields Fields) {
	for _, s := range sink.sinks {
		s.Log(fields)
	}
}

type priorityFilter struct {
	priority Priority
	target   Sink
}

func (filter *priorityFilter) Log(fields Fields) {
	// lower priority values indicate more important messages
	if fields["priority"].(Priority) <= filter.priority {
		filter.target.Log(fields)
	}
}

func PriorityFilter(priority Priority, target Sink) Sink {
	return &priorityFilter{
		priority: priority,
		target:   target,
	}
}
