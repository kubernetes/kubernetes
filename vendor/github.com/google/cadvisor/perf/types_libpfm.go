//go:build libpfm && cgo
// +build libpfm,cgo

// Copyright 2020 Google Inc. All Rights Reserved.
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

// Types related to handling perf events that are missing from unix package.
package perf

import "C"
import (
	"io"
	"unsafe"
)

// GroupReadFormat allows to read perf event's values for grouped events.
// See https://man7.org/linux/man-pages/man2/perf_event_open.2.html section "Reading results" with PERF_FORMAT_GROUP specified.
type GroupReadFormat struct {
	Nr          uint64 /* The number of events */
	TimeEnabled uint64 /* if PERF_FORMAT_TOTAL_TIME_ENABLED */
	TimeRunning uint64 /* if PERF_FORMAT_TOTAL_TIME_RUNNING */
}

type Values struct {
	Value uint64 /* The value of the event */
	ID    uint64 /* if PERF_FORMAT_ID */
}

// pfmPerfEncodeArgT represents structure that is used to parse perf event nam
// into perf_event_attr using libpfm.
type pfmPerfEncodeArgT struct {
	attr unsafe.Pointer
	fstr unsafe.Pointer
	size C.size_t
	_    C.int // idx
	_    C.int // cpu
	_    C.int // flags
}

type readerCloser interface {
	io.Reader
	io.Closer
}
