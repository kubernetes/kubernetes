// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package log

import "log"

// Logger provides an abstract interface for logging from Reporters.
// Applications can provide their own implementation of this interface to adapt
// reporters logging to whatever logging library they prefer (stdlib log,
// logrus, go-logging, etc).
type Logger interface {
	// Error logs a message at error priority
	Error(msg string)

	// Infof logs a message at info priority
	Infof(msg string, args ...interface{})
}

// StdLogger is implementation of the Logger interface that delegates to default `log` package
var StdLogger = &stdLogger{}

type stdLogger struct{}

func (l *stdLogger) Error(msg string) {
	log.Printf("ERROR: %s", msg)
}

// Infof logs a message at info priority
func (l *stdLogger) Infof(msg string, args ...interface{}) {
	log.Printf(msg, args...)
}

// NullLogger is implementation of the Logger interface that delegates to default `log` package
var NullLogger = &nullLogger{}

type nullLogger struct{}

func (l *nullLogger) Error(msg string)                      {}
func (l *nullLogger) Infof(msg string, args ...interface{}) {}
