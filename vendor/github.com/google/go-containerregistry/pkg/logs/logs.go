// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package logs exposes the loggers used by this library.
package logs

import (
	"io"
	"log"
)

var (
	// Warn is used to log non-fatal errors.
	Warn = log.New(io.Discard, "", log.LstdFlags)

	// Progress is used to log notable, successful events.
	Progress = log.New(io.Discard, "", log.LstdFlags)

	// Debug is used to log information that is useful for debugging.
	Debug = log.New(io.Discard, "", log.LstdFlags)
)

// Enabled checks to see if the logger's writer is set to something other
// than io.Discard. This allows callers to avoid expensive operations
// that will end up in /dev/null anyway.
func Enabled(l *log.Logger) bool {
	return l.Writer() != io.Discard
}
