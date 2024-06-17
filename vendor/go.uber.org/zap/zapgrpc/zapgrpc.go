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

// Package zapgrpc provides a logger that is compatible with grpclog.
package zapgrpc // import "go.uber.org/zap/zapgrpc"

import (
	"fmt"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// See https://github.com/grpc/grpc-go/blob/v1.35.0/grpclog/loggerv2.go#L77-L86
const (
	grpcLvlInfo int = iota
	grpcLvlWarn
	grpcLvlError
	grpcLvlFatal
)

var (
	// _grpcToZapLevel maps gRPC log levels to zap log levels.
	// See https://pkg.go.dev/go.uber.org/zap@v1.16.0/zapcore#Level
	_grpcToZapLevel = map[int]zapcore.Level{
		grpcLvlInfo:  zapcore.InfoLevel,
		grpcLvlWarn:  zapcore.WarnLevel,
		grpcLvlError: zapcore.ErrorLevel,
		grpcLvlFatal: zapcore.FatalLevel,
	}
)

// An Option overrides a Logger's default configuration.
type Option interface {
	apply(*Logger)
}

type optionFunc func(*Logger)

func (f optionFunc) apply(log *Logger) {
	f(log)
}

// WithDebug configures a Logger to print at zap's DebugLevel instead of
// InfoLevel.
// It only affects the Printf, Println and Print methods, which are only used in the gRPC v1 grpclog.Logger API.
//
// Deprecated: use grpclog.SetLoggerV2() for v2 API.
func WithDebug() Option {
	return optionFunc(func(logger *Logger) {
		logger.print = &printer{
			enab:   logger.levelEnabler,
			level:  zapcore.DebugLevel,
			print:  logger.delegate.Debug,
			printf: logger.delegate.Debugf,
		}
	})
}

// withWarn redirects the fatal level to the warn level, which makes testing
// easier. This is intentionally unexported.
func withWarn() Option {
	return optionFunc(func(logger *Logger) {
		logger.fatal = &printer{
			enab:   logger.levelEnabler,
			level:  zapcore.WarnLevel,
			print:  logger.delegate.Warn,
			printf: logger.delegate.Warnf,
		}
	})
}

// NewLogger returns a new Logger.
func NewLogger(l *zap.Logger, options ...Option) *Logger {
	logger := &Logger{
		delegate:     l.Sugar(),
		levelEnabler: l.Core(),
	}
	logger.print = &printer{
		enab:   logger.levelEnabler,
		level:  zapcore.InfoLevel,
		print:  logger.delegate.Info,
		printf: logger.delegate.Infof,
	}
	logger.fatal = &printer{
		enab:   logger.levelEnabler,
		level:  zapcore.FatalLevel,
		print:  logger.delegate.Fatal,
		printf: logger.delegate.Fatalf,
	}
	for _, option := range options {
		option.apply(logger)
	}
	return logger
}

// printer implements Print, Printf, and Println operations for a Zap level.
//
// We use it to customize Debug vs Info, and Warn vs Fatal for Print and Fatal
// respectively.
type printer struct {
	enab   zapcore.LevelEnabler
	level  zapcore.Level
	print  func(...interface{})
	printf func(string, ...interface{})
}

func (v *printer) Print(args ...interface{}) {
	v.print(args...)
}

func (v *printer) Printf(format string, args ...interface{}) {
	v.printf(format, args...)
}

func (v *printer) Println(args ...interface{}) {
	if v.enab.Enabled(v.level) {
		v.print(sprintln(args))
	}
}

// Logger adapts zap's Logger to be compatible with grpclog.LoggerV2 and the deprecated grpclog.Logger.
type Logger struct {
	delegate     *zap.SugaredLogger
	levelEnabler zapcore.LevelEnabler
	print        *printer
	fatal        *printer
	// printToDebug bool
	// fatalToWarn  bool
}

// Print implements grpclog.Logger.
//
// Deprecated: use [Logger.Info].
func (l *Logger) Print(args ...interface{}) {
	l.print.Print(args...)
}

// Printf implements grpclog.Logger.
//
// Deprecated: use [Logger.Infof].
func (l *Logger) Printf(format string, args ...interface{}) {
	l.print.Printf(format, args...)
}

// Println implements grpclog.Logger.
//
// Deprecated: use [Logger.Info].
func (l *Logger) Println(args ...interface{}) {
	l.print.Println(args...)
}

// Info implements grpclog.LoggerV2.
func (l *Logger) Info(args ...interface{}) {
	l.delegate.Info(args...)
}

// Infoln implements grpclog.LoggerV2.
func (l *Logger) Infoln(args ...interface{}) {
	if l.levelEnabler.Enabled(zapcore.InfoLevel) {
		l.delegate.Info(sprintln(args))
	}
}

// Infof implements grpclog.LoggerV2.
func (l *Logger) Infof(format string, args ...interface{}) {
	l.delegate.Infof(format, args...)
}

// Warning implements grpclog.LoggerV2.
func (l *Logger) Warning(args ...interface{}) {
	l.delegate.Warn(args...)
}

// Warningln implements grpclog.LoggerV2.
func (l *Logger) Warningln(args ...interface{}) {
	if l.levelEnabler.Enabled(zapcore.WarnLevel) {
		l.delegate.Warn(sprintln(args))
	}
}

// Warningf implements grpclog.LoggerV2.
func (l *Logger) Warningf(format string, args ...interface{}) {
	l.delegate.Warnf(format, args...)
}

// Error implements grpclog.LoggerV2.
func (l *Logger) Error(args ...interface{}) {
	l.delegate.Error(args...)
}

// Errorln implements grpclog.LoggerV2.
func (l *Logger) Errorln(args ...interface{}) {
	if l.levelEnabler.Enabled(zapcore.ErrorLevel) {
		l.delegate.Error(sprintln(args))
	}
}

// Errorf implements grpclog.LoggerV2.
func (l *Logger) Errorf(format string, args ...interface{}) {
	l.delegate.Errorf(format, args...)
}

// Fatal implements grpclog.LoggerV2.
func (l *Logger) Fatal(args ...interface{}) {
	l.fatal.Print(args...)
}

// Fatalln implements grpclog.LoggerV2.
func (l *Logger) Fatalln(args ...interface{}) {
	l.fatal.Println(args...)
}

// Fatalf implements grpclog.LoggerV2.
func (l *Logger) Fatalf(format string, args ...interface{}) {
	l.fatal.Printf(format, args...)
}

// V implements grpclog.LoggerV2.
func (l *Logger) V(level int) bool {
	return l.levelEnabler.Enabled(_grpcToZapLevel[level])
}

func sprintln(args []interface{}) string {
	s := fmt.Sprintln(args...)
	// Drop the new line character added by Sprintln
	return s[:len(s)-1]
}
