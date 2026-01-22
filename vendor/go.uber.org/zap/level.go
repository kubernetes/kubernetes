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

package zap

import (
	"sync/atomic"

	"go.uber.org/zap/internal"
	"go.uber.org/zap/zapcore"
)

const (
	// DebugLevel logs are typically voluminous, and are usually disabled in
	// production.
	DebugLevel = zapcore.DebugLevel
	// InfoLevel is the default logging priority.
	InfoLevel = zapcore.InfoLevel
	// WarnLevel logs are more important than Info, but don't need individual
	// human review.
	WarnLevel = zapcore.WarnLevel
	// ErrorLevel logs are high-priority. If an application is running smoothly,
	// it shouldn't generate any error-level logs.
	ErrorLevel = zapcore.ErrorLevel
	// DPanicLevel logs are particularly important errors. In development the
	// logger panics after writing the message.
	DPanicLevel = zapcore.DPanicLevel
	// PanicLevel logs a message, then panics.
	PanicLevel = zapcore.PanicLevel
	// FatalLevel logs a message, then calls os.Exit(1).
	FatalLevel = zapcore.FatalLevel
)

// LevelEnablerFunc is a convenient way to implement zapcore.LevelEnabler with
// an anonymous function.
//
// It's particularly useful when splitting log output between different
// outputs (e.g., standard error and standard out). For sample code, see the
// package-level AdvancedConfiguration example.
type LevelEnablerFunc func(zapcore.Level) bool

// Enabled calls the wrapped function.
func (f LevelEnablerFunc) Enabled(lvl zapcore.Level) bool { return f(lvl) }

// An AtomicLevel is an atomically changeable, dynamic logging level. It lets
// you safely change the log level of a tree of loggers (the root logger and
// any children created by adding context) at runtime.
//
// The AtomicLevel itself is an http.Handler that serves a JSON endpoint to
// alter its level.
//
// AtomicLevels must be created with the NewAtomicLevel constructor to allocate
// their internal atomic pointer.
type AtomicLevel struct {
	l *atomic.Int32
}

var _ internal.LeveledEnabler = AtomicLevel{}

// NewAtomicLevel creates an AtomicLevel with InfoLevel and above logging
// enabled.
func NewAtomicLevel() AtomicLevel {
	lvl := AtomicLevel{l: new(atomic.Int32)}
	lvl.l.Store(int32(InfoLevel))
	return lvl
}

// NewAtomicLevelAt is a convenience function that creates an AtomicLevel
// and then calls SetLevel with the given level.
func NewAtomicLevelAt(l zapcore.Level) AtomicLevel {
	a := NewAtomicLevel()
	a.SetLevel(l)
	return a
}

// ParseAtomicLevel parses an AtomicLevel based on a lowercase or all-caps ASCII
// representation of the log level. If the provided ASCII representation is
// invalid an error is returned.
//
// This is particularly useful when dealing with text input to configure log
// levels.
func ParseAtomicLevel(text string) (AtomicLevel, error) {
	a := NewAtomicLevel()
	l, err := zapcore.ParseLevel(text)
	if err != nil {
		return a, err
	}

	a.SetLevel(l)
	return a, nil
}

// Enabled implements the zapcore.LevelEnabler interface, which allows the
// AtomicLevel to be used in place of traditional static levels.
func (lvl AtomicLevel) Enabled(l zapcore.Level) bool {
	return lvl.Level().Enabled(l)
}

// Level returns the minimum enabled log level.
func (lvl AtomicLevel) Level() zapcore.Level {
	return zapcore.Level(int8(lvl.l.Load()))
}

// SetLevel alters the logging level.
func (lvl AtomicLevel) SetLevel(l zapcore.Level) {
	lvl.l.Store(int32(l))
}

// String returns the string representation of the underlying Level.
func (lvl AtomicLevel) String() string {
	return lvl.Level().String()
}

// UnmarshalText unmarshals the text to an AtomicLevel. It uses the same text
// representations as the static zapcore.Levels ("debug", "info", "warn",
// "error", "dpanic", "panic", and "fatal").
func (lvl *AtomicLevel) UnmarshalText(text []byte) error {
	if lvl.l == nil {
		lvl.l = &atomic.Int32{}
	}

	var l zapcore.Level
	if err := l.UnmarshalText(text); err != nil {
		return err
	}

	lvl.SetLevel(l)
	return nil
}

// MarshalText marshals the AtomicLevel to a byte slice. It uses the same
// text representation as the static zapcore.Levels ("debug", "info", "warn",
// "error", "dpanic", "panic", and "fatal").
func (lvl AtomicLevel) MarshalText() (text []byte, err error) {
	return lvl.Level().MarshalText()
}
