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
	"io"
	"io/ioutil"
	"log"
	"os"
	"strconv"
)

// LoggerV2 does underlying logging work for grpclog.
type LoggerV2 interface {
	// Info logs to INFO log. Arguments are handled in the manner of fmt.Print.
	Info(args ...interface{})
	// Infoln logs to INFO log. Arguments are handled in the manner of fmt.Println.
	Infoln(args ...interface{})
	// Infof logs to INFO log. Arguments are handled in the manner of fmt.Printf.
	Infof(format string, args ...interface{})
	// Warning logs to WARNING log. Arguments are handled in the manner of fmt.Print.
	Warning(args ...interface{})
	// Warningln logs to WARNING log. Arguments are handled in the manner of fmt.Println.
	Warningln(args ...interface{})
	// Warningf logs to WARNING log. Arguments are handled in the manner of fmt.Printf.
	Warningf(format string, args ...interface{})
	// Error logs to ERROR log. Arguments are handled in the manner of fmt.Print.
	Error(args ...interface{})
	// Errorln logs to ERROR log. Arguments are handled in the manner of fmt.Println.
	Errorln(args ...interface{})
	// Errorf logs to ERROR log. Arguments are handled in the manner of fmt.Printf.
	Errorf(format string, args ...interface{})
	// Fatal logs to ERROR log. Arguments are handled in the manner of fmt.Print.
	// gRPC ensures that all Fatal logs will exit with os.Exit(1).
	// Implementations may also call os.Exit() with a non-zero exit code.
	Fatal(args ...interface{})
	// Fatalln logs to ERROR log. Arguments are handled in the manner of fmt.Println.
	// gRPC ensures that all Fatal logs will exit with os.Exit(1).
	// Implementations may also call os.Exit() with a non-zero exit code.
	Fatalln(args ...interface{})
	// Fatalf logs to ERROR log. Arguments are handled in the manner of fmt.Printf.
	// gRPC ensures that all Fatal logs will exit with os.Exit(1).
	// Implementations may also call os.Exit() with a non-zero exit code.
	Fatalf(format string, args ...interface{})
	// V reports whether verbosity level l is at least the requested verbose level.
	V(l int) bool
}

// SetLoggerV2 sets logger that is used in grpc to a V2 logger.
// Not mutex-protected, should be called before any gRPC functions.
func SetLoggerV2(l LoggerV2) {
	logger = l
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
	m []*log.Logger
	v int
}

// NewLoggerV2 creates a loggerV2 with the provided writers.
// Fatal logs will be written to errorW, warningW, infoW, followed by exit(1).
// Error logs will be written to errorW, warningW and infoW.
// Warning logs will be written to warningW and infoW.
// Info logs will be written to infoW.
func NewLoggerV2(infoW, warningW, errorW io.Writer) LoggerV2 {
	return NewLoggerV2WithVerbosity(infoW, warningW, errorW, 0)
}

// NewLoggerV2WithVerbosity creates a loggerV2 with the provided writers and
// verbosity level.
func NewLoggerV2WithVerbosity(infoW, warningW, errorW io.Writer, v int) LoggerV2 {
	var m []*log.Logger
	m = append(m, log.New(infoW, severityName[infoLog]+": ", log.LstdFlags))
	m = append(m, log.New(io.MultiWriter(infoW, warningW), severityName[warningLog]+": ", log.LstdFlags))
	ew := io.MultiWriter(infoW, warningW, errorW) // ew will be used for error and fatal.
	m = append(m, log.New(ew, severityName[errorLog]+": ", log.LstdFlags))
	m = append(m, log.New(ew, severityName[fatalLog]+": ", log.LstdFlags))
	return &loggerT{m: m, v: v}
}

// newLoggerV2 creates a loggerV2 to be used as default logger.
// All logs are written to stderr.
func newLoggerV2() LoggerV2 {
	errorW := ioutil.Discard
	warningW := ioutil.Discard
	infoW := ioutil.Discard

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
	return NewLoggerV2WithVerbosity(infoW, warningW, errorW, v)
}

func (g *loggerT) Info(args ...interface{}) {
	g.m[infoLog].Print(args...)
}

func (g *loggerT) Infoln(args ...interface{}) {
	g.m[infoLog].Println(args...)
}

func (g *loggerT) Infof(format string, args ...interface{}) {
	g.m[infoLog].Printf(format, args...)
}

func (g *loggerT) Warning(args ...interface{}) {
	g.m[warningLog].Print(args...)
}

func (g *loggerT) Warningln(args ...interface{}) {
	g.m[warningLog].Println(args...)
}

func (g *loggerT) Warningf(format string, args ...interface{}) {
	g.m[warningLog].Printf(format, args...)
}

func (g *loggerT) Error(args ...interface{}) {
	g.m[errorLog].Print(args...)
}

func (g *loggerT) Errorln(args ...interface{}) {
	g.m[errorLog].Println(args...)
}

func (g *loggerT) Errorf(format string, args ...interface{}) {
	g.m[errorLog].Printf(format, args...)
}

func (g *loggerT) Fatal(args ...interface{}) {
	g.m[fatalLog].Fatal(args...)
	// No need to call os.Exit() again because log.Logger.Fatal() calls os.Exit().
}

func (g *loggerT) Fatalln(args ...interface{}) {
	g.m[fatalLog].Fatalln(args...)
	// No need to call os.Exit() again because log.Logger.Fatal() calls os.Exit().
}

func (g *loggerT) Fatalf(format string, args ...interface{}) {
	g.m[fatalLog].Fatalf(format, args...)
	// No need to call os.Exit() again because log.Logger.Fatal() calls os.Exit().
}

func (g *loggerT) V(l int) bool {
	return l <= g.v
}
