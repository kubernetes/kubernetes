// +build windows

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
	"io"
)

func CombinedSink(writer io.Writer, format string, fields []string) Sink {
	sinks := make([]Sink, 0)
	sinks = append(sinks, WriterSink(writer, format, fields))

	return &combinedSink{
		sinks: sinks,
	}
}
