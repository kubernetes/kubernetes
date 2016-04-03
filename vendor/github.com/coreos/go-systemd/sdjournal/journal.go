// Copyright 2015 RedHat, Inc.
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

// Package sdjournal provides a low-level Go interface to the
// systemd journal wrapped around the sd-journal C API.
//
// All public read methods map closely to the sd-journal API functions. See the
// sd-journal.h documentation[1] for information about each function.
//
// To write to the journal, see the pure-Go "journal" package
//
// [1] http://www.freedesktop.org/software/systemd/man/sd-journal.html
package sdjournal

/*
#cgo pkg-config: libsystemd-journal
#include <systemd/sd-journal.h>
#include <stdlib.h>
#include <syslog.h>
*/
import "C"
import (
	"fmt"
	"sync"
	"time"
	"unsafe"
)

// Journal entry field strings which correspond to:
// http://www.freedesktop.org/software/systemd/man/systemd.journal-fields.html
const (
	SD_JOURNAL_FIELD_SYSTEMD_UNIT = "_SYSTEMD_UNIT"
	SD_JOURNAL_FIELD_MESSAGE      = "MESSAGE"
	SD_JOURNAL_FIELD_PID          = "_PID"
	SD_JOURNAL_FIELD_UID          = "_UID"
	SD_JOURNAL_FIELD_GID          = "_GID"
	SD_JOURNAL_FIELD_HOSTNAME     = "_HOSTNAME"
	SD_JOURNAL_FIELD_MACHINE_ID   = "_MACHINE_ID"
)

// Journal event constants
const (
	SD_JOURNAL_NOP        = int(C.SD_JOURNAL_NOP)
	SD_JOURNAL_APPEND     = int(C.SD_JOURNAL_APPEND)
	SD_JOURNAL_INVALIDATE = int(C.SD_JOURNAL_INVALIDATE)
)

// A Journal is a Go wrapper of an sd_journal structure.
type Journal struct {
	cjournal *C.sd_journal
	mu       sync.Mutex
}

// A Match is a convenience wrapper to describe filters supplied to AddMatch.
type Match struct {
	Field string
	Value string
}

// String returns a string representation of a Match suitable for use with AddMatch.
func (m *Match) String() string {
	return m.Field + "=" + m.Value
}

// NewJournal returns a new Journal instance pointing to the local journal
func NewJournal() (*Journal, error) {
	j := &Journal{}
	err := C.sd_journal_open(&j.cjournal, C.SD_JOURNAL_LOCAL_ONLY)

	if err < 0 {
		return nil, fmt.Errorf("failed to open journal: %s", err)
	}

	return j, nil
}

// Close closes a journal opened with NewJournal.
func (j *Journal) Close() error {
	j.mu.Lock()
	C.sd_journal_close(j.cjournal)
	j.mu.Unlock()

	return nil
}

// AddMatch adds a match by which to filter the entries of the journal.
func (j *Journal) AddMatch(match string) error {
	m := C.CString(match)
	defer C.free(unsafe.Pointer(m))

	j.mu.Lock()
	C.sd_journal_add_match(j.cjournal, unsafe.Pointer(m), C.size_t(len(match)))
	j.mu.Unlock()

	return nil
}

// Next advances the read pointer into the journal by one entry.
func (j *Journal) Next() (int, error) {
	j.mu.Lock()
	r := C.sd_journal_next(j.cjournal)
	j.mu.Unlock()

	if r < 0 {
		return int(r), fmt.Errorf("failed to iterate journal: %d", r)
	}

	return int(r), nil
}

// NextSkip advances the read pointer by multiple entries at once,
// as specified by the skip parameter.
func (j *Journal) NextSkip(skip uint64) (uint64, error) {
	j.mu.Lock()
	r := C.sd_journal_next_skip(j.cjournal, C.uint64_t(skip))
	j.mu.Unlock()

	if r < 0 {
		return uint64(r), fmt.Errorf("failed to iterate journal: %d", r)
	}

	return uint64(r), nil
}

// Previous sets the read pointer into the journal back by one entry.
func (j *Journal) Previous() (uint64, error) {
	j.mu.Lock()
	r := C.sd_journal_previous(j.cjournal)
	j.mu.Unlock()

	if r < 0 {
		return uint64(r), fmt.Errorf("failed to iterate journal: %d", r)
	}

	return uint64(r), nil
}

// PreviousSkip sets back the read pointer by multiple entries at once,
// as specified by the skip parameter.
func (j *Journal) PreviousSkip(skip uint64) (uint64, error) {
	j.mu.Lock()
	r := C.sd_journal_previous_skip(j.cjournal, C.uint64_t(skip))
	j.mu.Unlock()

	if r < 0 {
		return uint64(r), fmt.Errorf("failed to iterate journal: %d", r)
	}

	return uint64(r), nil
}

// GetData gets the data object associated with a specific field from the
// current journal entry.
func (j *Journal) GetData(field string) (string, error) {
	f := C.CString(field)
	defer C.free(unsafe.Pointer(f))

	var d unsafe.Pointer
	var l C.size_t

	j.mu.Lock()
	err := C.sd_journal_get_data(j.cjournal, f, &d, &l)
	j.mu.Unlock()

	if err < 0 {
		return "", fmt.Errorf("failed to read message: %d", err)
	}

	msg := C.GoStringN((*C.char)(d), C.int(l))

	return msg, nil
}

// GetRealtimeUsec gets the realtime (wallclock) timestamp of the current
// journal entry.
func (j *Journal) GetRealtimeUsec() (uint64, error) {
	var usec C.uint64_t

	j.mu.Lock()
	r := C.sd_journal_get_realtime_usec(j.cjournal, &usec)
	j.mu.Unlock()

	if r < 0 {
		return 0, fmt.Errorf("error getting timestamp for entry: %d", r)
	}

	return uint64(usec), nil
}

// SeekTail may be used to seek to the end of the journal, i.e. the most recent
// available entry.
func (j *Journal) SeekTail() error {
	j.mu.Lock()
	err := C.sd_journal_seek_tail(j.cjournal)
	j.mu.Unlock()

	if err != 0 {
		return fmt.Errorf("failed to seek to tail of journal: %s", err)
	}

	return nil
}

// SeekRealtimeUsec seeks to the entry with the specified realtime (wallclock)
// timestamp, i.e. CLOCK_REALTIME.
func (j *Journal) SeekRealtimeUsec(usec uint64) error {
	j.mu.Lock()
	err := C.sd_journal_seek_realtime_usec(j.cjournal, C.uint64_t(usec))
	j.mu.Unlock()

	if err != 0 {
		return fmt.Errorf("failed to seek to %d: %d", usec, int(err))
	}

	return nil
}

// Wait will synchronously wait until the journal gets changed. The maximum time
// this call sleeps may be controlled with the timeout parameter.
func (j *Journal) Wait(timeout time.Duration) int {
	to := uint64(time.Now().Add(timeout).Unix() / 1000)
	j.mu.Lock()
	r := C.sd_journal_wait(j.cjournal, C.uint64_t(to))
	j.mu.Unlock()

	return int(r)
}

// GetUsage returns the journal disk space usage, in bytes.
func (j *Journal) GetUsage() (uint64, error) {
	var out C.uint64_t
	j.mu.Lock()
	err := C.sd_journal_get_usage(j.cjournal, &out)
	j.mu.Unlock()

	if err != 0 {
		return 0, fmt.Errorf("failed to get journal disk space usage: %d", err)
	}

	return uint64(out), nil
}
