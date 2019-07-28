package logger

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// LevelType tells a logger the minimum level to log. When code reports a log entry,
// the LogLevel indicates the level of the log entry. The logger only records entries
// whose level is at least the level it was told to log. See the Log* constants.
// For example, if a logger is configured with LogError, then LogError, LogPanic,
// and LogFatal entries will be logged; lower level entries are ignored.
type LevelType uint32

const (
	// LogNone tells a logger not to log any entries passed to it.
	LogNone LevelType = iota

	// LogFatal tells a logger to log all LogFatal entries passed to it.
	LogFatal

	// LogPanic tells a logger to log all LogPanic and LogFatal entries passed to it.
	LogPanic

	// LogError tells a logger to log all LogError, LogPanic and LogFatal entries passed to it.
	LogError

	// LogWarning tells a logger to log all LogWarning, LogError, LogPanic and LogFatal entries passed to it.
	LogWarning

	// LogInfo tells a logger to log all LogInfo, LogWarning, LogError, LogPanic and LogFatal entries passed to it.
	LogInfo

	// LogDebug tells a logger to log all LogDebug, LogInfo, LogWarning, LogError, LogPanic and LogFatal entries passed to it.
	LogDebug
)

const (
	logNone    = "NONE"
	logFatal   = "FATAL"
	logPanic   = "PANIC"
	logError   = "ERROR"
	logWarning = "WARNING"
	logInfo    = "INFO"
	logDebug   = "DEBUG"
	logUnknown = "UNKNOWN"
)

// ParseLevel converts the specified string into the corresponding LevelType.
func ParseLevel(s string) (lt LevelType, err error) {
	switch strings.ToUpper(s) {
	case logFatal:
		lt = LogFatal
	case logPanic:
		lt = LogPanic
	case logError:
		lt = LogError
	case logWarning:
		lt = LogWarning
	case logInfo:
		lt = LogInfo
	case logDebug:
		lt = LogDebug
	default:
		err = fmt.Errorf("bad log level '%s'", s)
	}
	return
}

// String implements the stringer interface for LevelType.
func (lt LevelType) String() string {
	switch lt {
	case LogNone:
		return logNone
	case LogFatal:
		return logFatal
	case LogPanic:
		return logPanic
	case LogError:
		return logError
	case LogWarning:
		return logWarning
	case LogInfo:
		return logInfo
	case LogDebug:
		return logDebug
	default:
		return logUnknown
	}
}

// Filter defines functions for filtering HTTP request/response content.
type Filter struct {
	// URL returns a potentially modified string representation of a request URL.
	URL func(u *url.URL) string

	// Header returns a potentially modified set of values for the specified key.
	// To completely exclude the header key/values return false.
	Header func(key string, val []string) (bool, []string)

	// Body returns a potentially modified request/response body.
	Body func(b []byte) []byte
}

func (f Filter) processURL(u *url.URL) string {
	if f.URL == nil {
		return u.String()
	}
	return f.URL(u)
}

func (f Filter) processHeader(k string, val []string) (bool, []string) {
	if f.Header == nil {
		return true, val
	}
	return f.Header(k, val)
}

func (f Filter) processBody(b []byte) []byte {
	if f.Body == nil {
		return b
	}
	return f.Body(b)
}

// Writer defines methods for writing to a logging facility.
type Writer interface {
	// Writeln writes the specified message with the standard log entry header and new-line character.
	Writeln(level LevelType, message string)

	// Writef writes the specified format specifier with the standard log entry header and no new-line character.
	Writef(level LevelType, format string, a ...interface{})

	// WriteRequest writes the specified HTTP request to the logger if the log level is greater than
	// or equal to LogInfo.  The request body, if set, is logged at level LogDebug or higher.
	// Custom filters can be specified to exclude URL, header, and/or body content from the log.
	// By default no request content is excluded.
	WriteRequest(req *http.Request, filter Filter)

	// WriteResponse writes the specified HTTP response to the logger if the log level is greater than
	// or equal to LogInfo.  The response body, if set, is logged at level LogDebug or higher.
	// Custom filters can be specified to exclude URL, header, and/or body content from the log.
	// By default no respone content is excluded.
	WriteResponse(resp *http.Response, filter Filter)
}

// Instance is the default log writer initialized during package init.
// This can be replaced with a custom implementation as required.
var Instance Writer

// default log level
var logLevel = LogNone

// Level returns the value specified in AZURE_GO_AUTOREST_LOG_LEVEL.
// If no value was specified the default value is LogNone.
// Custom loggers can call this to retrieve the configured log level.
func Level() LevelType {
	return logLevel
}

func init() {
	// separated for testing purposes
	initDefaultLogger()
}

func initDefaultLogger() {
	// init with nilLogger so callers don't have to do a nil check on Default
	Instance = nilLogger{}
	llStr := strings.ToLower(os.Getenv("AZURE_GO_SDK_LOG_LEVEL"))
	if llStr == "" {
		return
	}
	var err error
	logLevel, err = ParseLevel(llStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go-autorest: failed to parse log level: %s\n", err.Error())
		return
	}
	if logLevel == LogNone {
		return
	}
	// default to stderr
	dest := os.Stderr
	lfStr := os.Getenv("AZURE_GO_SDK_LOG_FILE")
	if strings.EqualFold(lfStr, "stdout") {
		dest = os.Stdout
	} else if lfStr != "" {
		lf, err := os.Create(lfStr)
		if err == nil {
			dest = lf
		} else {
			fmt.Fprintf(os.Stderr, "go-autorest: failed to create log file, using stderr: %s\n", err.Error())
		}
	}
	Instance = fileLogger{
		logLevel: logLevel,
		mu:       &sync.Mutex{},
		logFile:  dest,
	}
}

// the nil logger does nothing
type nilLogger struct{}

func (nilLogger) Writeln(LevelType, string) {}

func (nilLogger) Writef(LevelType, string, ...interface{}) {}

func (nilLogger) WriteRequest(*http.Request, Filter) {}

func (nilLogger) WriteResponse(*http.Response, Filter) {}

// A File is used instead of a Logger so the stream can be flushed after every write.
type fileLogger struct {
	logLevel LevelType
	mu       *sync.Mutex // for synchronizing writes to logFile
	logFile  *os.File
}

func (fl fileLogger) Writeln(level LevelType, message string) {
	fl.Writef(level, "%s\n", message)
}

func (fl fileLogger) Writef(level LevelType, format string, a ...interface{}) {
	if fl.logLevel >= level {
		fl.mu.Lock()
		defer fl.mu.Unlock()
		fmt.Fprintf(fl.logFile, "%s %s", entryHeader(level), fmt.Sprintf(format, a...))
		fl.logFile.Sync()
	}
}

func (fl fileLogger) WriteRequest(req *http.Request, filter Filter) {
	if req == nil || fl.logLevel < LogInfo {
		return
	}
	b := &bytes.Buffer{}
	fmt.Fprintf(b, "%s REQUEST: %s %s\n", entryHeader(LogInfo), req.Method, filter.processURL(req.URL))
	// dump headers
	for k, v := range req.Header {
		if ok, mv := filter.processHeader(k, v); ok {
			fmt.Fprintf(b, "%s: %s\n", k, strings.Join(mv, ","))
		}
	}
	if fl.shouldLogBody(req.Header, req.Body) {
		// dump body
		body, err := ioutil.ReadAll(req.Body)
		if err == nil {
			fmt.Fprintln(b, string(filter.processBody(body)))
			if nc, ok := req.Body.(io.Seeker); ok {
				// rewind to the beginning
				nc.Seek(0, io.SeekStart)
			} else {
				// recreate the body
				req.Body = ioutil.NopCloser(bytes.NewReader(body))
			}
		} else {
			fmt.Fprintf(b, "failed to read body: %v\n", err)
		}
	}
	fl.mu.Lock()
	defer fl.mu.Unlock()
	fmt.Fprint(fl.logFile, b.String())
	fl.logFile.Sync()
}

func (fl fileLogger) WriteResponse(resp *http.Response, filter Filter) {
	if resp == nil || fl.logLevel < LogInfo {
		return
	}
	b := &bytes.Buffer{}
	fmt.Fprintf(b, "%s RESPONSE: %d %s\n", entryHeader(LogInfo), resp.StatusCode, filter.processURL(resp.Request.URL))
	// dump headers
	for k, v := range resp.Header {
		if ok, mv := filter.processHeader(k, v); ok {
			fmt.Fprintf(b, "%s: %s\n", k, strings.Join(mv, ","))
		}
	}
	if fl.shouldLogBody(resp.Header, resp.Body) {
		// dump body
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err == nil {
			fmt.Fprintln(b, string(filter.processBody(body)))
			resp.Body = ioutil.NopCloser(bytes.NewReader(body))
		} else {
			fmt.Fprintf(b, "failed to read body: %v\n", err)
		}
	}
	fl.mu.Lock()
	defer fl.mu.Unlock()
	fmt.Fprint(fl.logFile, b.String())
	fl.logFile.Sync()
}

// returns true if the provided body should be included in the log
func (fl fileLogger) shouldLogBody(header http.Header, body io.ReadCloser) bool {
	ct := header.Get("Content-Type")
	return fl.logLevel >= LogDebug && body != nil && strings.Index(ct, "application/octet-stream") == -1
}

// creates standard header for log entries, it contains a timestamp and the log level
func entryHeader(level LevelType) string {
	// this format provides a fixed number of digits so the size of the timestamp is constant
	return fmt.Sprintf("(%s) %s:", time.Now().Format("2006-01-02T15:04:05.0000000Z07:00"), level.String())
}
