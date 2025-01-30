/*
Copyright 2019 The logr Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package stdr implements github.com/go-logr/logr.Logger in terms of
// Go's standard log package.
package stdr

import (
	"log"
	"os"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
)

// The global verbosity level.  See SetVerbosity().
var globalVerbosity int

// SetVerbosity sets the global level against which all info logs will be
// compared.  If this is greater than or equal to the "V" of the logger, the
// message will be logged.  A higher value here means more logs will be written.
// The previous verbosity value is returned.  This is not concurrent-safe -
// callers must be sure to call it from only one goroutine.
func SetVerbosity(v int) int {
	old := globalVerbosity
	globalVerbosity = v
	return old
}

// New returns a logr.Logger which is implemented by Go's standard log package,
// or something like it.  If std is nil, this will use a default logger
// instead.
//
// Example: stdr.New(log.New(os.Stderr, "", log.LstdFlags|log.Lshortfile)))
func New(std StdLogger) logr.Logger {
	return NewWithOptions(std, Options{})
}

// NewWithOptions returns a logr.Logger which is implemented by Go's standard
// log package, or something like it.  See New for details.
func NewWithOptions(std StdLogger, opts Options) logr.Logger {
	if std == nil {
		// Go's log.Default() is only available in 1.16 and higher.
		std = log.New(os.Stderr, "", log.LstdFlags)
	}

	if opts.Depth < 0 {
		opts.Depth = 0
	}

	fopts := funcr.Options{
		LogCaller: funcr.MessageClass(opts.LogCaller),
	}

	sl := &logger{
		Formatter: funcr.NewFormatter(fopts),
		std:       std,
	}

	// For skipping our own logger.Info/Error.
	sl.Formatter.AddCallDepth(1 + opts.Depth)

	return logr.New(sl)
}

// Options carries parameters which influence the way logs are generated.
type Options struct {
	// Depth biases the assumed number of call frames to the "true" caller.
	// This is useful when the calling code calls a function which then calls
	// stdr (e.g. a logging shim to another API).  Values less than zero will
	// be treated as zero.
	Depth int

	// LogCaller tells stdr to add a "caller" key to some or all log lines.
	// Go's log package has options to log this natively, too.
	LogCaller MessageClass

	// TODO: add an option to log the date/time
}

// MessageClass indicates which category or categories of messages to consider.
type MessageClass int

const (
	// None ignores all message classes.
	None MessageClass = iota
	// All considers all message classes.
	All
	// Info only considers info messages.
	Info
	// Error only considers error messages.
	Error
)

// StdLogger is the subset of the Go stdlib log.Logger API that is needed for
// this adapter.
type StdLogger interface {
	// Output is the same as log.Output and log.Logger.Output.
	Output(calldepth int, logline string) error
}

type logger struct {
	funcr.Formatter
	std StdLogger
}

var _ logr.LogSink = &logger{}
var _ logr.CallDepthLogSink = &logger{}

func (l logger) Enabled(level int) bool {
	return globalVerbosity >= level
}

func (l logger) Info(level int, msg string, kvList ...interface{}) {
	prefix, args := l.FormatInfo(level, msg, kvList)
	if prefix != "" {
		args = prefix + ": " + args
	}
	_ = l.std.Output(l.Formatter.GetDepth()+1, args)
}

func (l logger) Error(err error, msg string, kvList ...interface{}) {
	prefix, args := l.FormatError(err, msg, kvList)
	if prefix != "" {
		args = prefix + ": " + args
	}
	_ = l.std.Output(l.Formatter.GetDepth()+1, args)
}

func (l logger) WithName(name string) logr.LogSink {
	l.Formatter.AddName(name)
	return &l
}

func (l logger) WithValues(kvList ...interface{}) logr.LogSink {
	l.Formatter.AddValues(kvList)
	return &l
}

func (l logger) WithCallDepth(depth int) logr.LogSink {
	l.Formatter.AddCallDepth(depth)
	return &l
}

// Underlier exposes access to the underlying logging implementation.  Since
// callers only have a logr.Logger, they have to know which implementation is
// in use, so this interface is less of an abstraction and more of way to test
// type conversion.
type Underlier interface {
	GetUnderlying() StdLogger
}

// GetUnderlying returns the StdLogger underneath this logger.  Since StdLogger
// is itself an interface, the result may or may not be a Go log.Logger.
func (l logger) GetUnderlying() StdLogger {
	return l.std
}
