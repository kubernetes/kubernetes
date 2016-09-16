// Copyright 2015 The etcd Authors
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

// Package logutil includes utilities to facilitate logging.
package logutil

import (
	"fmt"
	"sync"
	"time"

	"github.com/coreos/pkg/capnslog"
)

var (
	defaultMergePeriod     = time.Second
	defaultTimeOutputScale = 10 * time.Millisecond

	outputInterval = time.Second
)

// line represents a log line that can be printed out
// through capnslog.PackageLogger.
type line struct {
	level capnslog.LogLevel
	str   string
}

func (l line) append(s string) line {
	return line{
		level: l.level,
		str:   l.str + " " + s,
	}
}

// status represents the merge status of a line.
type status struct {
	period time.Duration

	start time.Time // start time of latest merge period
	count int       // number of merged lines from starting
}

func (s *status) isInMergePeriod(now time.Time) bool {
	return s.period == 0 || s.start.Add(s.period).After(now)
}

func (s *status) isEmpty() bool { return s.count == 0 }

func (s *status) summary(now time.Time) string {
	ts := s.start.Round(defaultTimeOutputScale)
	took := now.Round(defaultTimeOutputScale).Sub(ts)
	return fmt.Sprintf("[merged %d repeated lines in %s]", s.count, took)
}

func (s *status) reset(now time.Time) {
	s.start = now
	s.count = 0
}

// MergeLogger supports merge logging, which merges repeated log lines
// and prints summary log lines instead.
//
// For merge logging, MergeLogger prints out the line when the line appears
// at the first time. MergeLogger holds the same log line printed within
// defaultMergePeriod, and prints out summary log line at the end of defaultMergePeriod.
// It stops merging when the line doesn't appear within the
// defaultMergePeriod.
type MergeLogger struct {
	*capnslog.PackageLogger

	mu      sync.Mutex // protect statusm
	statusm map[line]*status
}

func NewMergeLogger(logger *capnslog.PackageLogger) *MergeLogger {
	l := &MergeLogger{
		PackageLogger: logger,
		statusm:       make(map[line]*status),
	}
	go l.outputLoop()
	return l
}

func (l *MergeLogger) MergeInfo(entries ...interface{}) {
	l.merge(line{
		level: capnslog.INFO,
		str:   fmt.Sprint(entries...),
	})
}

func (l *MergeLogger) MergeInfof(format string, args ...interface{}) {
	l.merge(line{
		level: capnslog.INFO,
		str:   fmt.Sprintf(format, args...),
	})
}

func (l *MergeLogger) MergeNotice(entries ...interface{}) {
	l.merge(line{
		level: capnslog.NOTICE,
		str:   fmt.Sprint(entries...),
	})
}

func (l *MergeLogger) MergeNoticef(format string, args ...interface{}) {
	l.merge(line{
		level: capnslog.NOTICE,
		str:   fmt.Sprintf(format, args...),
	})
}

func (l *MergeLogger) MergeWarning(entries ...interface{}) {
	l.merge(line{
		level: capnslog.WARNING,
		str:   fmt.Sprint(entries...),
	})
}

func (l *MergeLogger) MergeWarningf(format string, args ...interface{}) {
	l.merge(line{
		level: capnslog.WARNING,
		str:   fmt.Sprintf(format, args...),
	})
}

func (l *MergeLogger) MergeError(entries ...interface{}) {
	l.merge(line{
		level: capnslog.ERROR,
		str:   fmt.Sprint(entries...),
	})
}

func (l *MergeLogger) MergeErrorf(format string, args ...interface{}) {
	l.merge(line{
		level: capnslog.ERROR,
		str:   fmt.Sprintf(format, args...),
	})
}

func (l *MergeLogger) merge(ln line) {
	l.mu.Lock()

	// increase count if the logger is merging the line
	if status, ok := l.statusm[ln]; ok {
		status.count++
		l.mu.Unlock()
		return
	}

	// initialize status of the line
	l.statusm[ln] = &status{
		period: defaultMergePeriod,
		start:  time.Now(),
	}
	// release the lock before IO operation
	l.mu.Unlock()
	// print out the line at its first time
	l.PackageLogger.Logf(ln.level, ln.str)
}

func (l *MergeLogger) outputLoop() {
	for now := range time.Tick(outputInterval) {
		var outputs []line

		l.mu.Lock()
		for ln, status := range l.statusm {
			if status.isInMergePeriod(now) {
				continue
			}
			if status.isEmpty() {
				delete(l.statusm, ln)
				continue
			}
			outputs = append(outputs, ln.append(status.summary(now)))
			status.reset(now)
		}
		l.mu.Unlock()

		for _, o := range outputs {
			l.PackageLogger.Logf(o.level, o.str)
		}
	}
}
