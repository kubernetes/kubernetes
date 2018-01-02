// Copyright 2015 The rkt Authors
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

package logger

import "log"

// Logger mimics golang's testing logging functionality as an interface.
type Logger interface {
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Skip(args ...interface{})
	Skipf(format string, args ...interface{})
}

func Error(args ...interface{})                 { logger.Error(args...) }
func Errorf(format string, args ...interface{}) { logger.Errorf(format, args...) }
func Fatal(args ...interface{})                 { logger.Fatal(args...) }
func Fatalf(format string, args ...interface{}) { logger.Fatalf(format, args...) }
func Log(args ...interface{})                   { logger.Log(args...) }
func Logf(format string, args ...interface{})   { logger.Logf(format, args...) }
func Skip(args ...interface{})                  { logger.Skip(args...) }
func Skipf(format string, args ...interface{})  { logger.Skipf(format, args...) }

var logger Logger = DefaultLogger{}

func SetLogger(l Logger) { logger = l }

// Compatibility type to forward messages to Go's log package
type DefaultLogger struct{}

func (dl DefaultLogger) Error(args ...interface{})                 { log.Print(args...) }
func (dl DefaultLogger) Errorf(format string, args ...interface{}) { log.Printf(format, args...) }
func (dl DefaultLogger) Fatal(args ...interface{})                 { log.Fatal(args...) }
func (dl DefaultLogger) Fatalf(format string, args ...interface{}) { log.Fatalf(format, args...) }
func (dl DefaultLogger) Log(args ...interface{})                   { log.Print(args...) }
func (dl DefaultLogger) Logf(format string, args ...interface{})   { log.Printf(format, args...) }
func (dl DefaultLogger) Skip(args ...interface{})                  { log.Print(args...) }
func (dl DefaultLogger) Skipf(format string, args ...interface{})  { log.Printf(format, args...) }
