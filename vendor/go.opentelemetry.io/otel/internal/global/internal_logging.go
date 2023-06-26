// Copyright The OpenTelemetry Authors
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

package global // import "go.opentelemetry.io/otel/internal/global"

import (
	"log"
	"os"
	"sync/atomic"
	"unsafe"

	"github.com/go-logr/logr"
	"github.com/go-logr/stdr"
)

// globalLogger is the logging interface used within the otel api and sdk provide details of the internals.
//
// The default logger uses stdr which is backed by the standard `log.Logger`
// interface. This logger will only show messages at the Error Level.
var globalLogger unsafe.Pointer

func init() {
	SetLogger(stdr.New(log.New(os.Stderr, "", log.LstdFlags|log.Lshortfile)))
}

// SetLogger overrides the globalLogger with l.
//
// To see Warn messages use a logger with `l.V(1).Enabled() == true`
// To see Info messages use a logger with `l.V(4).Enabled() == true`
// To see Debug messages use a logger with `l.V(8).Enabled() == true`.
func SetLogger(l logr.Logger) {
	atomic.StorePointer(&globalLogger, unsafe.Pointer(&l))
}

func getLogger() logr.Logger {
	return *(*logr.Logger)(atomic.LoadPointer(&globalLogger))
}

// Info prints messages about the general state of the API or SDK.
// This should usually be less than 5 messages a minute.
func Info(msg string, keysAndValues ...interface{}) {
	getLogger().V(4).Info(msg, keysAndValues...)
}

// Error prints messages about exceptional states of the API or SDK.
func Error(err error, msg string, keysAndValues ...interface{}) {
	getLogger().Error(err, msg, keysAndValues...)
}

// Debug prints messages about all internal changes in the API or SDK.
func Debug(msg string, keysAndValues ...interface{}) {
	getLogger().V(8).Info(msg, keysAndValues...)
}

// Warn prints messages about warnings in the API or SDK.
// Not an error but is likely more important than an informational event.
func Warn(msg string, keysAndValues ...interface{}) {
	getLogger().V(1).Info(msg, keysAndValues...)
}
