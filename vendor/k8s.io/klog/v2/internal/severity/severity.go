// Copyright 2013 Google Inc. All Rights Reserved.
// Copyright 2022 The Kubernetes Authors.
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

// Package severity provides definitions for klog severity (info, warning, ...)
package severity

import (
	"strings"
)

// severity identifies the sort of log: info, warning etc. The binding to flag.Value
// is handled in klog.go
type Severity int32 // sync/atomic int32

// These constants identify the log levels in order of increasing severity.
// A message written to a high-severity log file is also written to each
// lower-severity log file.
const (
	InfoLog Severity = iota
	WarningLog
	ErrorLog
	FatalLog
	NumSeverity = 4
)

// Char contains one shortcut letter per severity level.
const Char = "IWEF"

// Name contains one name per severity level.
var Name = []string{
	InfoLog:    "INFO",
	WarningLog: "WARNING",
	ErrorLog:   "ERROR",
	FatalLog:   "FATAL",
}

// ByName looks up a severity level by name.
func ByName(s string) (Severity, bool) {
	s = strings.ToUpper(s)
	for i, name := range Name {
		if name == s {
			return Severity(i), true
		}
	}
	return 0, false
}
