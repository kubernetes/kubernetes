// Go support for leveled logs, analogous to https://code.google.com/p/google-glog/
//
// Copyright 2013 Google Inc. All Rights Reserved.
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

// File I/O for logs.

package klog

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// MaxSize is the maximum size of a log file in bytes.
var MaxSize uint64 = 1024 * 1024 * 1800

// logDirs lists the candidate directories for new log files.
var logDirs []string

func createLogDirs() {
	if logging.logDir != "" {
		logDirs = append(logDirs, logging.logDir)
	}
	logDirs = append(logDirs, os.TempDir())
}

var (
	pid          = os.Getpid()
	program      = filepath.Base(os.Args[0])
	host         = "unknownhost"
	userName     = "unknownuser"
	userNameOnce sync.Once
)

func init() {
	if h, err := os.Hostname(); err == nil {
		host = shortHostname(h)
	}
}

// shortHostname returns its argument, truncating at the first period.
// For instance, given "www.google.com" it returns "www".
func shortHostname(hostname string) string {
	if i := strings.Index(hostname, "."); i >= 0 {
		return hostname[:i]
	}
	return hostname
}

// logName returns a new log file name containing tag, with start time t, and
// the name for the symlink for tag.
func logName(tag string, t time.Time) (name, link string) {
	name = fmt.Sprintf("%s.%s.%s.log.%s.%04d%02d%02d-%02d%02d%02d.%d",
		program,
		host,
		getUserName(),
		tag,
		t.Year(),
		t.Month(),
		t.Day(),
		t.Hour(),
		t.Minute(),
		t.Second(),
		pid)
	return name, program + "." + tag
}

var onceLogDirs sync.Once

// create creates a new log file and returns the file and its filename, which
// contains tag ("INFO", "FATAL", etc.) and t.  If the file is created
// successfully, create also attempts to update the symlink for that tag, ignoring
// errors.
// The startup argument indicates whether this is the initial startup of klog.
// If startup is true, existing files are opened for appending instead of truncated.
func create(tag string, t time.Time, startup bool) (f *os.File, filename string, err error) {
	if logging.logFile != "" {
		f, err := openOrCreate(logging.logFile, startup)
		if err == nil {
			return f, logging.logFile, nil
		}
		return nil, "", fmt.Errorf("log: unable to create log: %v", err)
	}
	onceLogDirs.Do(createLogDirs)
	if len(logDirs) == 0 {
		return nil, "", errors.New("log: no log dirs")
	}
	name, link := logName(tag, t)
	var lastErr error
	for _, dir := range logDirs {
		fname := filepath.Join(dir, name)
		f, err := openOrCreate(fname, startup)
		if err == nil {
			symlink := filepath.Join(dir, link)
			_ = os.Remove(symlink)        // ignore err
			_ = os.Symlink(name, symlink) // ignore err
			return f, fname, nil
		}
		lastErr = err
	}
	return nil, "", fmt.Errorf("log: cannot create log: %v", lastErr)
}

// The startup argument indicates whether this is the initial startup of klog.
// If startup is true, existing files are opened for appending instead of truncated.
func openOrCreate(name string, startup bool) (*os.File, error) {
	if startup {
		f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
		return f, err
	}
	f, err := os.Create(name)
	return f, err
}
