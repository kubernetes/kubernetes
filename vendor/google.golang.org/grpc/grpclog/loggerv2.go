/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpclog

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"

	"google.golang.org/grpc/internal/grpclog"
)

// LoggerV2 does underlying logging work for grpclog.
type LoggerV2 interface {
	// Info logs to INFO log. Arguments are handled in the manner of fmt.Print.
	Info(args ...any)
	// Infoln logs to INFO log. Arguments are handled in the manner of fmt.Println.
	Infoln(args ...any)
	// Infof logs to INFO log. Arguments are handled in the manner of fmt.Printf.
	Infof(format string, args ...any)
	// Warning logs to WARNING log. Arguments are handled in the manner of fmt.Print.
	Warning(args ...any)
	// Warningln logs to WARNING log. Arguments are handled in the manner of fmt.Println.
	Warningln(args ...any)
	// Warningf logs to WARNING log. Arguments are handled in the manner of fmt.Printf.
	Warningf(format string, args ...any)
	// Error logs to ERROR log. Arguments are handled in the manner of fmt.Print.
	Error(args ...any)
	// Errorln logs to ERROR log. Arguments are handled in the manner of fmt.Println.
	Errorln(args ...any)
	// Errorf logs to ERROR log. Arguments are handled in the manner of fmt.Printf.
	Errorf(format string, args ...any)
	// Fatal logs to ERROR log. Arguments are handled in the manner of fmt.Print.
	// gRPC ensures that all Fatal logs will exit with os.Exit(1).
	// Implementations may also call os.Exit() with a non-zero exit code.
	Fatal(args ...any)
	// Fatalln logs to ERROR log. Arguments are handled in the manner of fmt.Println.
	// gRPC ensures that all Fatal logs will exit with os.Exit(1).
	// Implementations may also call os.Exit() with a non-zero exit code.
	Fatalln(args ...any)
	// Fatalf logs to ERROR log. Arguments are handled in the manner of fmt.Printf.
	// gRPC ensures that all Fatal logs will exit with os.Exit(1).
	// Implementations may also call os.Exit() with a non-zero exit code.
	Fatalf(format string, args ...any)
	// V reports whether verbosity level l is at least the requested verbose level.
	V(l int) bool
}

// SetLoggerV2 sets logger that is used in grpc to a V2 logger.
// Not mutex-protected, should be called before any gRPC functions.
func SetLoggerV2(l LoggerV2) {
	if _, ok := l.(*componentData); ok {
		panic("cannot use component logger as grpclog logger")
	}
	grpclog.Logger = l
	grpclog.DepthLogger, _ = l.(grpclog.DepthLoggerV2)
}

const (
	// infoLog indicates Info severity.
	infoLog int = iota
	// warningLog indicates Warning severity.
	warningLog
	// errorLog indicates Error severity.
	errorLog
	// fatalLog indicates Fatal severity.
	fatalLog
)

// severityName contains the string representation of each severity.
var severityName = []string{
	infoLog:    "INFO",
	warningLog: "WARNING",
	errorLog:   "ERROR",
	fatalLog:   "FATAL",
}

// loggerT is the default logger used by grpclog.
type loggerT struct {
	m          []*log.Logger
	v          int
	jsonFormat bool
}

// NewLoggerV2 creates a loggerV2 with the provided writers.
// Fatal logs will be written to errorW, warningW, infoW, followed by exit(1).
// Error logs will be written to errorW, warningW and infoW.
// Warning logs will be written to warningW and infoW.
// Info logs will be written to infoW.
func NewLoggerV2(infoW, warningW, errorW io.Writer) LoggerV2 {
	return newLoggerV2WithConfig(infoW, warningW, errorW, loggerV2Config{})
}

// NewLoggerV2WithVerbosity creates a loggerV2 with the provided writers and
// verbosity level.
func NewLoggerV2WithVerbosity(infoW, warningW, errorW io.Writer, v int) LoggerV2 {
	return newLoggerV2WithConfig(infoW, warningW, errorW, loggerV2Config{verbose: v})
}

type loggerV2Config struct {
	verbose    int
	jsonFormat bool
}

func newLoggerV2WithConfig(infoW, warningW, errorW io.Writer, c loggerV2Config) LoggerV2 {
	var m []*log.Logger
	flag := log.LstdFlags
	if c.jsonFormat {
		flag = 0
	}
	m = append(m, log.New(infoW, "", flag))
	m = append(m, log.New(io.MultiWriter(infoW, warningW), "", flag))
	ew := io.MultiWriter(infoW, warningW, errorW) // ew will be used for error and fatal.
	m = append(m, log.New(ew, "", flag))
	m = append(m, log.New(ew, "", flag))
	return &loggerT{m: m, v: c.verbose, jsonFormat: c.jsonFormat}
}

// newLoggerV2 creates a loggerV2 to be used as default logger.
// All logs are written to stderr.
func newLoggerV2() LoggerV2 {
	errorW := io.Discard
	warningW := io.Discard
	infoW := io.Discard

	logLevel := os.Getenv("GRPC_GO_LOG_SEVERITY_LEVEL")
	switch logLevel {
	case "", "ERROR", "error": // If env is unset, set level to ERROR.
		errorW = os.Stderr
	case "WARNING", "warning":
		warningW = os.Stderr
	case "INFO", "info":
		infoW = os.Stderr
	}

	var v int
	vLevel := os.Getenv("GRPC_GO_LOG_VERBOSITY_LEVEL")
	if vl, err := strconv.Atoi(vLevel); err == nil {
		v = vl
	}

	jsonFormat := strings.EqualFold(os.Getenv("GRPC_GO_LOG_FORMATTER"), "json")

	return newLoggerV2WithConfig(infoW, warningW, errorW, loggerV2Config{
		verbose:    v,
		jsonFormat: jsonFormat,
	})
}

func (g *loggerT) output(severity int, s string) {
	sevStr := severityName[severity]
	if !g.jsonFormat {
		g.m[severity].Output(2, fmt.Sprintf("%v: %v", sevStr, s))
		return
	}
	// TODO: we can also include the logging component, but that needs more
	// (API) changes.
	b, _ := json.Marshal(map[string]string{
		"severity": sevStr,
		"message":  s,
	})
	g.m[severity].Output(2, string(b))
}

func (g *loggerT) Info(args ...any) {
	g.output(infoLog, fmt.Sprint(args...))
}

func (g *loggerT) Infoln(args ...any) {
	g.output(infoLog, fmt.Sprintln(args...))
}

func (g *loggerT) Infof(format string, args ...any) {
	g.output(infoLog, fmt.Sprintf(format, args...))
}

func (g *loggerT) Warning(args ...any) {
	g.output(warningLog, fmt.Sprint(args...))
}

func (g *loggerT) Warningln(args ...any) {
	g.output(warningLog, fmt.Sprintln(args...))
}

func (g *loggerT) Warningf(format string, args ...any) {
	g.output(warningLog, fmt.Sprintf(format, args...))
}

func (g *loggerT) Error(args ...any) {
	g.output(errorLog, fmt.Sprint(args...))
}

func (g *loggerT) Errorln(args ...any) {
	g.output(errorLog, fmt.Sprintln(args...))
}

func (g *loggerT) Errorf(format string, args ...any) {
	g.output(errorLog, fmt.Sprintf(format, args...))
}

func (g *loggerT) Fatal(args ...any) {
	g.output(fatalLog, fmt.Sprint(args...))
	os.Exit(1)
}

func (g *loggerT) Fatalln(args ...any) {
	g.output(fatalLog, fmt.Sprintln(args...))
	os.Exit(1)
}

func (g *loggerT) Fatalf(format string, args ...any) {
	g.output(fatalLog, fmt.Sprintf(format, args...))
	os.Exit(1)
}

func (g *loggerT) V(l int) bool {
	return l <= g.v
}

// DepthLoggerV2 logs at a specified call frame. If a LoggerV2 also implements
// DepthLoggerV2, the below functions will be called with the appropriate stack
// depth set for trivial functions the logger may ignore.
//
// # Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type DepthLoggerV2 interface {
	LoggerV2
	// InfoDepth logs to INFO log at the specified depth. Arguments are handled in the manner of fmt.Println.
	InfoDepth(depth int, args ...any)
	// WarningDepth logs to WARNING log at the specified depth. Arguments are handled in the manner of fmt.Println.
	WarningDepth(depth int, args ...any)
	// ErrorDepth logs to ERROR log at the specified depth. Arguments are handled in the manner of fmt.Println.
	ErrorDepth(depth int, args ...any)
	// FatalDepth logs to FATAL log at the specified depth. Arguments are handled in the manner of fmt.Println.
	FatalDepth(depth int, args ...any)
}
