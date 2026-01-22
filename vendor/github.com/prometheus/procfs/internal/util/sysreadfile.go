// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build (linux || darwin) && !appengine
// +build linux darwin
// +build !appengine

package util

import (
	"bytes"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// SysReadFile is a simplified os.ReadFile that invokes syscall.Read directly.
// https://github.com/prometheus/node_exporter/pull/728/files
//
// Note that this function will not read files larger than 128 bytes.
func SysReadFile(file string) (string, error) {
	f, err := os.Open(file)
	if err != nil {
		return "", err
	}
	defer f.Close()

	// On some machines, hwmon drivers are broken and return EAGAIN.  This causes
	// Go's os.ReadFile implementation to poll forever.
	//
	// Since we either want to read data or bail immediately, do the simplest
	// possible read using syscall directly.
	const sysFileBufferSize = 128
	b := make([]byte, sysFileBufferSize)
	n, err := syscall.Read(int(f.Fd()), b)
	if err != nil {
		return "", err
	}

	return string(bytes.TrimSpace(b[:n])), nil
}

// SysReadUintFromFile reads a file using SysReadFile and attempts to parse a uint64 from it.
func SysReadUintFromFile(path string) (uint64, error) {
	data, err := SysReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}

// SysReadIntFromFile reads a file using SysReadFile and attempts to parse a int64 from it.
func SysReadIntFromFile(path string) (int64, error) {
	data, err := SysReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
}
