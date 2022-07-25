// Go support for leveled logs, analogous to https://code.google.com/p/google-glog/
//
// Copyright 2013 Google Inc. All Rights Reserved.
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

// Package klog implements logging analogous to the Google-internal C++ INFO/ERROR/V setup.
// It provides functions Info, Warning, Error, Fatal, plus formatting variants such as
// Infof. It also provides V-style logging controlled by the -v and -vmodule=file=2 flags.
//
// Basic examples:
//
//	klog.Info("Prepare to repel boarders")
//
//	klog.Fatalf("Initialization failed: %s", err)
//
// See the documentation for the V function for an explanation of these examples:
//
//	if klog.V(2) {
//		klog.Info("Starting transaction...")
//	}
//
//	klog.V(2).Infoln("Processed", nItems, "elements")
//
// Log output is buffered and written periodically using Flush. Programs
// should call Flush before exiting to guarantee all log output is written.
//
// By default, all log statements write to standard error.
// This package provides several flags that modify this behavior.
// As a result, flag.Parse must be called before any logging is done.
//
//	-logtostderr=true
//		Logs are written to standard error instead of to files.
//              This shortcuts most of the usual output routing:
//              -alsologtostderr, -stderrthreshold and -log_dir have no
//              effect and output redirection at runtime with SetOutput is
//              ignored.
//	-alsologtostderr=false
//		Logs are written to standard error as well as to files.
//	-stderrthreshold=ERROR
//		Log events at or above this severity are logged to standard
//		error as well as to files.
//	-log_dir=""
//		Log files will be written to this directory instead of the
//		default temporary directory.
//
//	Other flags provide aids to debugging.
//
//	-log_backtrace_at=""
//		When set to a file and line number holding a logging statement,
//		such as
//			-log_backtrace_at=gopherflakes.go:234
//		a stack trace will be written to the Info log whenever execution
//		hits that statement. (Unlike with -vmodule, the ".go" must be
//		present.)
//	-v=0
//		Enable V-leveled logging at the specified level.
//	-vmodule=""
//		The syntax of the argument is a comma-separated list of pattern=N,
//		where pattern is a literal file name (minus the ".go" suffix) or
//		"glob" pattern and N is a V level. For instance,
//			-vmodule=gopher*=3
//		sets the V level to 3 in all Go files whose names begin "gopher".
//
package klog

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	stdLog "log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"

	"k8s.io/klog/v2/internal/buffer"
	"k8s.io/klog/v2/internal/clock"
	"k8s.io/klog/v2/internal/dbg"
	"k8s.io/klog/v2/internal/serialize"
	"k8s.io/klog/v2/internal/severity"
)

// severityValue identifies the sort of log: info, warning etc. It also implements
// the flag.Value interface. The -stderrthreshold flag is of type severity and
// should be modified only through the flag.Value interface. The values match
// the corresponding constants in C++.
type severityValue struct {
	severity.Severity
}

// get returns the value of the severity.
func (s *severityValue) get() severity.Severity {
	return severity.Severity(atomic.LoadInt32((*int32)(&s.Severity)))
}

// set sets the value of the severity.
func (s *severityValue) set(val severity.Severity) {
	atomic.StoreInt32((*int32)(&s.Severity), int32(val))
}

// String is part of the flag.Value interface.
func (s *severityValue) String() string {
	return strconv.FormatInt(int64(s.Severity), 10)
}

// Get is part of the flag.Getter interface.
func (s *severityValue) Get() interface{} {
	return s.Severity
}

// Set is part of the flag.Value interface.
func (s *severityValue) Set(value string) error {
	var threshold severity.Severity
	// Is it a known name?
	if v, ok := severity.ByName(value); ok {
		threshold = v
	} else {
		v, err := strconv.ParseInt(value, 10, 32)
		if err != nil {
			return err
		}
		threshold = severity.Severity(v)
	}
	logging.stderrThreshold.set(threshold)
	return nil
}

// OutputStats tracks the number of output lines and bytes written.
type OutputStats struct {
	lines int64
	bytes int64
}

// Lines returns the number of lines written.
func (s *OutputStats) Lines() int64 {
	return atomic.LoadInt64(&s.lines)
}

// Bytes returns the number of bytes written.
func (s *OutputStats) Bytes() int64 {
	return atomic.LoadInt64(&s.bytes)
}

// Stats tracks the number of lines of output and number of bytes
// per severity level. Values must be read with atomic.LoadInt64.
var Stats struct {
	Info, Warning, Error OutputStats
}

var severityStats = [severity.NumSeverity]*OutputStats{
	severity.InfoLog:    &Stats.Info,
	severity.WarningLog: &Stats.Warning,
	severity.ErrorLog:   &Stats.Error,
}

// Level is exported because it appears in the arguments to V and is
// the type of the v flag, which can be set programmatically.
// It's a distinct type because we want to discriminate it from logType.
// Variables of type level are only changed under logging.mu.
// The -v flag is read only with atomic ops, so the state of the logging
// module is consistent.

// Level is treated as a sync/atomic int32.

// Level specifies a level of verbosity for V logs. *Level implements
// flag.Value; the -v flag is of type Level and should be modified
// only through the flag.Value interface.
type Level int32

// get returns the value of the Level.
func (l *Level) get() Level {
	return Level(atomic.LoadInt32((*int32)(l)))
}

// set sets the value of the Level.
func (l *Level) set(val Level) {
	atomic.StoreInt32((*int32)(l), int32(val))
}

// String is part of the flag.Value interface.
func (l *Level) String() string {
	return strconv.FormatInt(int64(*l), 10)
}

// Get is part of the flag.Getter interface.
func (l *Level) Get() interface{} {
	return *l
}

// Set is part of the flag.Value interface.
func (l *Level) Set(value string) error {
	v, err := strconv.ParseInt(value, 10, 32)
	if err != nil {
		return err
	}
	logging.mu.Lock()
	defer logging.mu.Unlock()
	logging.setVState(Level(v), logging.vmodule.filter, false)
	return nil
}

// moduleSpec represents the setting of the -vmodule flag.
type moduleSpec struct {
	filter []modulePat
}

// modulePat contains a filter for the -vmodule flag.
// It holds a verbosity level and a file pattern to match.
type modulePat struct {
	pattern string
	literal bool // The pattern is a literal string
	level   Level
}

// match reports whether the file matches the pattern. It uses a string
// comparison if the pattern contains no metacharacters.
func (m *modulePat) match(file string) bool {
	if m.literal {
		return file == m.pattern
	}
	match, _ := filepath.Match(m.pattern, file)
	return match
}

func (m *moduleSpec) String() string {
	// Lock because the type is not atomic. TODO: clean this up.
	logging.mu.Lock()
	defer logging.mu.Unlock()
	return m.serialize()
}

func (m *moduleSpec) serialize() string {
	var b bytes.Buffer
	for i, f := range m.filter {
		if i > 0 {
			b.WriteRune(',')
		}
		fmt.Fprintf(&b, "%s=%d", f.pattern, f.level)
	}
	return b.String()
}

// Get is part of the (Go 1.2)  flag.Getter interface. It always returns nil for this flag type since the
// struct is not exported.
func (m *moduleSpec) Get() interface{} {
	return nil
}

var errVmoduleSyntax = errors.New("syntax error: expect comma-separated list of filename=N")

// Set will sets module value
// Syntax: -vmodule=recordio=2,file=1,gfs*=3
func (m *moduleSpec) Set(value string) error {
	filter, err := parseModuleSpec(value)
	if err != nil {
		return err
	}
	logging.mu.Lock()
	defer logging.mu.Unlock()
	logging.setVState(logging.verbosity, filter, true)
	return nil
}

func parseModuleSpec(value string) ([]modulePat, error) {
	var filter []modulePat
	for _, pat := range strings.Split(value, ",") {
		if len(pat) == 0 {
			// Empty strings such as from a trailing comma can be ignored.
			continue
		}
		patLev := strings.Split(pat, "=")
		if len(patLev) != 2 || len(patLev[0]) == 0 || len(patLev[1]) == 0 {
			return nil, errVmoduleSyntax
		}
		pattern := patLev[0]
		v, err := strconv.ParseInt(patLev[1], 10, 32)
		if err != nil {
			return nil, errors.New("syntax error: expect comma-separated list of filename=N")
		}
		if v < 0 {
			return nil, errors.New("negative value for vmodule level")
		}
		if v == 0 {
			continue // Ignore. It's harmless but no point in paying the overhead.
		}
		// TODO: check syntax of filter?
		filter = append(filter, modulePat{pattern, isLiteral(pattern), Level(v)})
	}
	return filter, nil
}

// isLiteral reports whether the pattern is a literal string, that is, has no metacharacters
// that require filepath.Match to be called to match the pattern.
func isLiteral(pattern string) bool {
	return !strings.ContainsAny(pattern, `\*?[]`)
}

// traceLocation represents the setting of the -log_backtrace_at flag.
type traceLocation struct {
	file string
	line int
}

// isSet reports whether the trace location has been specified.
// logging.mu is held.
func (t *traceLocation) isSet() bool {
	return t.line > 0
}

// match reports whether the specified file and line matches the trace location.
// The argument file name is the full path, not the basename specified in the flag.
// logging.mu is held.
func (t *traceLocation) match(file string, line int) bool {
	if t.line != line {
		return false
	}
	if i := strings.LastIndex(file, "/"); i >= 0 {
		file = file[i+1:]
	}
	return t.file == file
}

func (t *traceLocation) String() string {
	// Lock because the type is not atomic. TODO: clean this up.
	logging.mu.Lock()
	defer logging.mu.Unlock()
	return fmt.Sprintf("%s:%d", t.file, t.line)
}

// Get is part of the (Go 1.2) flag.Getter interface. It always returns nil for this flag type since the
// struct is not exported
func (t *traceLocation) Get() interface{} {
	return nil
}

var errTraceSyntax = errors.New("syntax error: expect file.go:234")

// Set will sets backtrace value
// Syntax: -log_backtrace_at=gopherflakes.go:234
// Note that unlike vmodule the file extension is included here.
func (t *traceLocation) Set(value string) error {
	if value == "" {
		// Unset.
		logging.mu.Lock()
		defer logging.mu.Unlock()
		t.line = 0
		t.file = ""
		return nil
	}
	fields := strings.Split(value, ":")
	if len(fields) != 2 {
		return errTraceSyntax
	}
	file, line := fields[0], fields[1]
	if !strings.Contains(file, ".") {
		return errTraceSyntax
	}
	v, err := strconv.Atoi(line)
	if err != nil {
		return errTraceSyntax
	}
	if v <= 0 {
		return errors.New("negative or zero value for level")
	}
	logging.mu.Lock()
	defer logging.mu.Unlock()
	t.line = v
	t.file = file
	return nil
}

// flushSyncWriter is the interface satisfied by logging destinations.
type flushSyncWriter interface {
	Flush() error
	Sync() error
	io.Writer
}

// init sets up the defaults.
func init() {
	logging.stderrThreshold = severityValue{
		Severity: severity.ErrorLog, // Default stderrThreshold is ERROR.
	}
	logging.setVState(0, nil, false)
	logging.logDir = ""
	logging.logFile = ""
	logging.logFileMaxSizeMB = 1800
	logging.toStderr = true
	logging.alsoToStderr = false
	logging.skipHeaders = false
	logging.addDirHeader = false
	logging.skipLogHeaders = false
	logging.oneOutput = false
	logging.flushD = newFlushDaemon(logging.lockAndFlushAll, nil)
}

// InitFlags is for explicitly initializing the flags.
func InitFlags(flagset *flag.FlagSet) {
	if flagset == nil {
		flagset = flag.CommandLine
	}

	flagset.StringVar(&logging.logDir, "log_dir", logging.logDir, "If non-empty, write log files in this directory (no effect when -logtostderr=true)")
	flagset.StringVar(&logging.logFile, "log_file", logging.logFile, "If non-empty, use this log file (no effect when -logtostderr=true)")
	flagset.Uint64Var(&logging.logFileMaxSizeMB, "log_file_max_size", logging.logFileMaxSizeMB,
		"Defines the maximum size a log file can grow to (no effect when -logtostderr=true). Unit is megabytes. "+
			"If the value is 0, the maximum file size is unlimited.")
	flagset.BoolVar(&logging.toStderr, "logtostderr", logging.toStderr, "log to standard error instead of files")
	flagset.BoolVar(&logging.alsoToStderr, "alsologtostderr", logging.alsoToStderr, "log to standard error as well as files (no effect when -logtostderr=true)")
	flagset.Var(&logging.verbosity, "v", "number for the log level verbosity")
	flagset.BoolVar(&logging.addDirHeader, "add_dir_header", logging.addDirHeader, "If true, adds the file directory to the header of the log messages")
	flagset.BoolVar(&logging.skipHeaders, "skip_headers", logging.skipHeaders, "If true, avoid header prefixes in the log messages")
	flagset.BoolVar(&logging.oneOutput, "one_output", logging.oneOutput, "If true, only write logs to their native severity level (vs also writing to each lower severity level; no effect when -logtostderr=true)")
	flagset.BoolVar(&logging.skipLogHeaders, "skip_log_headers", logging.skipLogHeaders, "If true, avoid headers when opening log files (no effect when -logtostderr=true)")
	flagset.Var(&logging.stderrThreshold, "stderrthreshold", "logs at or above this threshold go to stderr when writing to files and stderr (no effect when -logtostderr=true or -alsologtostderr=false)")
	flagset.Var(&logging.vmodule, "vmodule", "comma-separated list of pattern=N settings for file-filtered logging")
	flagset.Var(&logging.traceLocation, "log_backtrace_at", "when logging hits line file:N, emit a stack trace")
}

// Flush flushes all pending log I/O.
func Flush() {
	logging.lockAndFlushAll()
}

// settings collects global settings.
type settings struct {
	// contextualLoggingEnabled controls whether contextual logging is
	// active. Disabling it may have some small performance benefit.
	contextualLoggingEnabled bool

	// logger is the global Logger chosen by users of klog, nil if
	// none is available.
	logger *Logger

	// loggerOptions contains the options that were supplied for
	// globalLogger.
	loggerOptions loggerOptions

	// Boolean flags. Not handled atomically because the flag.Value interface
	// does not let us avoid the =true, and that shorthand is necessary for
	// compatibility. TODO: does this matter enough to fix? Seems unlikely.
	toStderr     bool // The -logtostderr flag.
	alsoToStderr bool // The -alsologtostderr flag.

	// Level flag. Handled atomically.
	stderrThreshold severityValue // The -stderrthreshold flag.

	// Access to all of the following fields must be protected via a mutex.

	// file holds writer for each of the log types.
	file [severity.NumSeverity]flushSyncWriter
	// flushInterval is the interval for periodic flushing. If zero,
	// the global default will be used.
	flushInterval time.Duration

	// filterLength stores the length of the vmodule filter chain. If greater
	// than zero, it means vmodule is enabled. It may be read safely
	// using sync.LoadInt32, but is only modified under mu.
	filterLength int32
	// traceLocation is the state of the -log_backtrace_at flag.
	traceLocation traceLocation
	// These flags are modified only under lock, although verbosity may be fetched
	// safely using atomic.LoadInt32.
	vmodule   moduleSpec // The state of the -vmodule flag.
	verbosity Level      // V logging level, the value of the -v flag/

	// If non-empty, overrides the choice of directory in which to write logs.
	// See createLogDirs for the full list of possible destinations.
	logDir string

	// If non-empty, specifies the path of the file to write logs. mutually exclusive
	// with the log_dir option.
	logFile string

	// When logFile is specified, this limiter makes sure the logFile won't exceeds a certain size. When exceeds, the
	// logFile will be cleaned up. If this value is 0, no size limitation will be applied to logFile.
	logFileMaxSizeMB uint64

	// If true, do not add the prefix headers, useful when used with SetOutput
	skipHeaders bool

	// If true, do not add the headers to log files
	skipLogHeaders bool

	// If true, add the file directory to the header
	addDirHeader bool

	// If true, messages will not be propagated to lower severity log levels
	oneOutput bool

	// If set, all output will be filtered through the filter.
	filter LogFilter
}

// deepCopy creates a copy that doesn't share anything with the original
// instance.
func (s settings) deepCopy() settings {
	// vmodule is a slice and would be shared, so we have copy it.
	filter := make([]modulePat, len(s.vmodule.filter))
	for i := range s.vmodule.filter {
		filter[i] = s.vmodule.filter[i]
	}
	s.vmodule.filter = filter

	return s
}

// loggingT collects all the global state of the logging setup.
type loggingT struct {
	settings

	// bufferCache maintains the free list. It uses its own mutex
	// so buffers can be grabbed and printed to without holding the main lock,
	// for better parallelization.
	bufferCache buffer.Buffers

	// flushD holds a flushDaemon that frequently flushes log file buffers.
	// Uses its own mutex.
	flushD *flushDaemon

	// mu protects the remaining elements of this structure and the fields
	// in settingsT which need a mutex lock.
	mu sync.Mutex

	// pcs is used in V to avoid an allocation when computing the caller's PC.
	pcs [1]uintptr
	// vmap is a cache of the V Level for each V() call site, identified by PC.
	// It is wiped whenever the vmodule flag changes state.
	vmap map[uintptr]Level
}

var logging = loggingT{
	settings: settings{
		contextualLoggingEnabled: true,
	},
}

// setVState sets a consistent state for V logging.
// l.mu is held.
func (l *loggingT) setVState(verbosity Level, filter []modulePat, setFilter bool) {
	// Turn verbosity off so V will not fire while we are in transition.
	l.verbosity.set(0)
	// Ditto for filter length.
	atomic.StoreInt32(&l.filterLength, 0)

	// Set the new filters and wipe the pc->Level map if the filter has changed.
	if setFilter {
		l.vmodule.filter = filter
		l.vmap = make(map[uintptr]Level)
	}

	// Things are consistent now, so enable filtering and verbosity.
	// They are enabled in order opposite to that in V.
	atomic.StoreInt32(&l.filterLength, int32(len(filter)))
	l.verbosity.set(verbosity)
}

var timeNow = time.Now // Stubbed out for testing.

// CaptureState gathers information about all current klog settings.
// The result can be used to restore those settings.
func CaptureState() State {
	logging.mu.Lock()
	defer logging.mu.Unlock()
	return &state{
		settings:      logging.settings.deepCopy(),
		flushDRunning: logging.flushD.isRunning(),
		maxSize:       MaxSize,
	}
}

// State stores a snapshot of klog settings. It gets created with CaptureState
// and can be used to restore the entire state. Modifying individual settings
// is supported via the command line flags.
type State interface {
	// Restore restore the entire state. It may get called more than once.
	Restore()
}

type state struct {
	settings

	flushDRunning bool
	maxSize       uint64
}

func (s *state) Restore() {
	// This needs to be done before mutex locking.
	if s.flushDRunning && !logging.flushD.isRunning() {
		// This is not quite accurate: StartFlushDaemon might
		// have been called with some different interval.
		interval := s.flushInterval
		if interval == 0 {
			interval = flushInterval
		}
		logging.flushD.run(interval)
	} else if !s.flushDRunning && logging.flushD.isRunning() {
		logging.flushD.stop()
	}

	logging.mu.Lock()
	defer logging.mu.Unlock()

	logging.settings = s.settings
	logging.setVState(s.verbosity, s.vmodule.filter, true)
	MaxSize = s.maxSize
}

/*
header formats a log header as defined by the C++ implementation.
It returns a buffer containing the formatted header and the user's file and line number.
The depth specifies how many stack frames above lives the source line to be identified in the log message.

Log lines have this form:
	Lmmdd hh:mm:ss.uuuuuu threadid file:line] msg...
where the fields are defined as follows:
	L                A single character, representing the log level (eg 'I' for INFO)
	mm               The month (zero padded; ie May is '05')
	dd               The day (zero padded)
	hh:mm:ss.uuuuuu  Time in hours, minutes and fractional seconds
	threadid         The space-padded thread ID as returned by GetTID()
	file             The file name
	line             The line number
	msg              The user-supplied message
*/
func (l *loggingT) header(s severity.Severity, depth int) (*buffer.Buffer, string, int) {
	_, file, line, ok := runtime.Caller(3 + depth)
	if !ok {
		file = "???"
		line = 1
	} else {
		if slash := strings.LastIndex(file, "/"); slash >= 0 {
			path := file
			file = path[slash+1:]
			if l.addDirHeader {
				if dirsep := strings.LastIndex(path[:slash], "/"); dirsep >= 0 {
					file = path[dirsep+1:]
				}
			}
		}
	}
	return l.formatHeader(s, file, line), file, line
}

// formatHeader formats a log header using the provided file name and line number.
func (l *loggingT) formatHeader(s severity.Severity, file string, line int) *buffer.Buffer {
	buf := l.bufferCache.GetBuffer()
	if l.skipHeaders {
		return buf
	}
	now := timeNow()
	buf.FormatHeader(s, file, line, now)
	return buf
}

func (l *loggingT) println(s severity.Severity, logger *logr.Logger, filter LogFilter, args ...interface{}) {
	l.printlnDepth(s, logger, filter, 1, args...)
}

func (l *loggingT) printlnDepth(s severity.Severity, logger *logr.Logger, filter LogFilter, depth int, args ...interface{}) {
	buf, file, line := l.header(s, depth)
	// if logger is set, we clear the generated header as we rely on the backing
	// logger implementation to print headers
	if logger != nil {
		l.bufferCache.PutBuffer(buf)
		buf = l.bufferCache.GetBuffer()
	}
	if filter != nil {
		args = filter.Filter(args)
	}
	fmt.Fprintln(buf, args...)
	l.output(s, logger, buf, depth, file, line, false)
}

func (l *loggingT) print(s severity.Severity, logger *logr.Logger, filter LogFilter, args ...interface{}) {
	l.printDepth(s, logger, filter, 1, args...)
}

func (l *loggingT) printDepth(s severity.Severity, logger *logr.Logger, filter LogFilter, depth int, args ...interface{}) {
	buf, file, line := l.header(s, depth)
	// if logr is set, we clear the generated header as we rely on the backing
	// logr implementation to print headers
	if logger != nil {
		l.bufferCache.PutBuffer(buf)
		buf = l.bufferCache.GetBuffer()
	}
	if filter != nil {
		args = filter.Filter(args)
	}
	fmt.Fprint(buf, args...)
	if buf.Len() == 0 || buf.Bytes()[buf.Len()-1] != '\n' {
		buf.WriteByte('\n')
	}
	l.output(s, logger, buf, depth, file, line, false)
}

func (l *loggingT) printf(s severity.Severity, logger *logr.Logger, filter LogFilter, format string, args ...interface{}) {
	l.printfDepth(s, logger, filter, 1, format, args...)
}

func (l *loggingT) printfDepth(s severity.Severity, logger *logr.Logger, filter LogFilter, depth int, format string, args ...interface{}) {
	buf, file, line := l.header(s, depth)
	// if logr is set, we clear the generated header as we rely on the backing
	// logr implementation to print headers
	if logger != nil {
		l.bufferCache.PutBuffer(buf)
		buf = l.bufferCache.GetBuffer()
	}
	if filter != nil {
		format, args = filter.FilterF(format, args)
	}
	fmt.Fprintf(buf, format, args...)
	if buf.Bytes()[buf.Len()-1] != '\n' {
		buf.WriteByte('\n')
	}
	l.output(s, logger, buf, depth, file, line, false)
}

// printWithFileLine behaves like print but uses the provided file and line number.  If
// alsoLogToStderr is true, the log message always appears on standard error; it
// will also appear in the log file unless --logtostderr is set.
func (l *loggingT) printWithFileLine(s severity.Severity, logger *logr.Logger, filter LogFilter, file string, line int, alsoToStderr bool, args ...interface{}) {
	buf := l.formatHeader(s, file, line)
	// if logr is set, we clear the generated header as we rely on the backing
	// logr implementation to print headers
	if logger != nil {
		l.bufferCache.PutBuffer(buf)
		buf = l.bufferCache.GetBuffer()
	}
	if filter != nil {
		args = filter.Filter(args)
	}
	fmt.Fprint(buf, args...)
	if buf.Bytes()[buf.Len()-1] != '\n' {
		buf.WriteByte('\n')
	}
	l.output(s, logger, buf, 2 /* depth */, file, line, alsoToStderr)
}

// if loggr is specified, will call loggr.Error, otherwise output with logging module.
func (l *loggingT) errorS(err error, logger *logr.Logger, filter LogFilter, depth int, msg string, keysAndValues ...interface{}) {
	if filter != nil {
		msg, keysAndValues = filter.FilterS(msg, keysAndValues)
	}
	if logger != nil {
		logger.WithCallDepth(depth+2).Error(err, msg, keysAndValues...)
		return
	}
	l.printS(err, severity.ErrorLog, depth+1, msg, keysAndValues...)
}

// if loggr is specified, will call loggr.Info, otherwise output with logging module.
func (l *loggingT) infoS(logger *logr.Logger, filter LogFilter, depth int, msg string, keysAndValues ...interface{}) {
	if filter != nil {
		msg, keysAndValues = filter.FilterS(msg, keysAndValues)
	}
	if logger != nil {
		logger.WithCallDepth(depth+2).Info(msg, keysAndValues...)
		return
	}
	l.printS(nil, severity.InfoLog, depth+1, msg, keysAndValues...)
}

// printS is called from infoS and errorS if loggr is not specified.
// set log severity by s
func (l *loggingT) printS(err error, s severity.Severity, depth int, msg string, keysAndValues ...interface{}) {
	// Only create a new buffer if we don't have one cached.
	b := l.bufferCache.GetBuffer()
	// The message is always quoted, even if it contains line breaks.
	// If developers want multi-line output, they should use a small, fixed
	// message and put the multi-line output into a value.
	b.WriteString(strconv.Quote(msg))
	if err != nil {
		serialize.KVListFormat(&b.Buffer, "err", err)
	}
	serialize.KVListFormat(&b.Buffer, keysAndValues...)
	l.printDepth(s, logging.logger, nil, depth+1, &b.Buffer)
	// Make the buffer available for reuse.
	l.bufferCache.PutBuffer(b)
}

// redirectBuffer is used to set an alternate destination for the logs
type redirectBuffer struct {
	w io.Writer
}

func (rb *redirectBuffer) Sync() error {
	return nil
}

func (rb *redirectBuffer) Flush() error {
	return nil
}

func (rb *redirectBuffer) Write(bytes []byte) (n int, err error) {
	return rb.w.Write(bytes)
}

// SetOutput sets the output destination for all severities
func SetOutput(w io.Writer) {
	logging.mu.Lock()
	defer logging.mu.Unlock()
	for s := severity.FatalLog; s >= severity.InfoLog; s-- {
		rb := &redirectBuffer{
			w: w,
		}
		logging.file[s] = rb
	}
}

// SetOutputBySeverity sets the output destination for specific severity
func SetOutputBySeverity(name string, w io.Writer) {
	logging.mu.Lock()
	defer logging.mu.Unlock()
	sev, ok := severity.ByName(name)
	if !ok {
		panic(fmt.Sprintf("SetOutputBySeverity(%q): unrecognized severity name", name))
	}
	rb := &redirectBuffer{
		w: w,
	}
	logging.file[sev] = rb
}

// LogToStderr sets whether to log exclusively to stderr, bypassing outputs
func LogToStderr(stderr bool) {
	logging.mu.Lock()
	defer logging.mu.Unlock()

	logging.toStderr = stderr
}

// output writes the data to the log files and releases the buffer.
func (l *loggingT) output(s severity.Severity, log *logr.Logger, buf *buffer.Buffer, depth int, file string, line int, alsoToStderr bool) {
	var isLocked = true
	l.mu.Lock()
	defer func() {
		if isLocked {
			// Unlock before returning in case that it wasn't done already.
			l.mu.Unlock()
		}
	}()

	if l.traceLocation.isSet() {
		if l.traceLocation.match(file, line) {
			buf.Write(dbg.Stacks(false))
		}
	}
	data := buf.Bytes()
	if log != nil {
		// TODO: set 'severity' and caller information as structured log info
		// keysAndValues := []interface{}{"severity", severityName[s], "file", file, "line", line}
		if s == severity.ErrorLog {
			logging.logger.WithCallDepth(depth+3).Error(nil, string(data))
		} else {
			log.WithCallDepth(depth + 3).Info(string(data))
		}
	} else if l.toStderr {
		os.Stderr.Write(data)
	} else {
		if alsoToStderr || l.alsoToStderr || s >= l.stderrThreshold.get() {
			os.Stderr.Write(data)
		}

		if logging.logFile != "" {
			// Since we are using a single log file, all of the items in l.file array
			// will point to the same file, so just use one of them to write data.
			if l.file[severity.InfoLog] == nil {
				if err := l.createFiles(severity.InfoLog); err != nil {
					os.Stderr.Write(data) // Make sure the message appears somewhere.
					l.exit(err)
				}
			}
			l.file[severity.InfoLog].Write(data)
		} else {
			if l.file[s] == nil {
				if err := l.createFiles(s); err != nil {
					os.Stderr.Write(data) // Make sure the message appears somewhere.
					l.exit(err)
				}
			}

			if l.oneOutput {
				l.file[s].Write(data)
			} else {
				switch s {
				case severity.FatalLog:
					l.file[severity.FatalLog].Write(data)
					fallthrough
				case severity.ErrorLog:
					l.file[severity.ErrorLog].Write(data)
					fallthrough
				case severity.WarningLog:
					l.file[severity.WarningLog].Write(data)
					fallthrough
				case severity.InfoLog:
					l.file[severity.InfoLog].Write(data)
				}
			}
		}
	}
	if s == severity.FatalLog {
		// If we got here via Exit rather than Fatal, print no stacks.
		if atomic.LoadUint32(&fatalNoStacks) > 0 {
			l.mu.Unlock()
			isLocked = false
			timeoutFlush(ExitFlushTimeout)
			OsExit(1)
		}
		// Dump all goroutine stacks before exiting.
		// First, make sure we see the trace for the current goroutine on standard error.
		// If -logtostderr has been specified, the loop below will do that anyway
		// as the first stack in the full dump.
		if !l.toStderr {
			os.Stderr.Write(dbg.Stacks(false))
		}

		// Write the stack trace for all goroutines to the files.
		trace := dbg.Stacks(true)
		logExitFunc = func(error) {} // If we get a write error, we'll still exit below.
		for log := severity.FatalLog; log >= severity.InfoLog; log-- {
			if f := l.file[log]; f != nil { // Can be nil if -logtostderr is set.
				f.Write(trace)
			}
		}
		l.mu.Unlock()
		isLocked = false
		timeoutFlush(ExitFlushTimeout)
		OsExit(255) // C++ uses -1, which is silly because it's anded with 255 anyway.
	}
	l.bufferCache.PutBuffer(buf)

	if stats := severityStats[s]; stats != nil {
		atomic.AddInt64(&stats.lines, 1)
		atomic.AddInt64(&stats.bytes, int64(len(data)))
	}
}

// logExitFunc provides a simple mechanism to override the default behavior
// of exiting on error. Used in testing and to guarantee we reach a required exit
// for fatal logs. Instead, exit could be a function rather than a method but that
// would make its use clumsier.
var logExitFunc func(error)

// exit is called if there is trouble creating or writing log files.
// It flushes the logs and exits the program; there's no point in hanging around.
// l.mu is held.
func (l *loggingT) exit(err error) {
	fmt.Fprintf(os.Stderr, "log: exiting because of error: %s\n", err)
	// If logExitFunc is set, we do that instead of exiting.
	if logExitFunc != nil {
		logExitFunc(err)
		return
	}
	l.flushAll()
	OsExit(2)
}

// syncBuffer joins a bufio.Writer to its underlying file, providing access to the
// file's Sync method and providing a wrapper for the Write method that provides log
// file rotation. There are conflicting methods, so the file cannot be embedded.
// l.mu is held for all its methods.
type syncBuffer struct {
	logger *loggingT
	*bufio.Writer
	file     *os.File
	sev      severity.Severity
	nbytes   uint64 // The number of bytes written to this file
	maxbytes uint64 // The max number of bytes this syncBuffer.file can hold before cleaning up.
}

func (sb *syncBuffer) Sync() error {
	return sb.file.Sync()
}

// CalculateMaxSize returns the real max size in bytes after considering the default max size and the flag options.
func CalculateMaxSize() uint64 {
	if logging.logFile != "" {
		if logging.logFileMaxSizeMB == 0 {
			// If logFileMaxSizeMB is zero, we don't have limitations on the log size.
			return math.MaxUint64
		}
		// Flag logFileMaxSizeMB is in MB for user convenience.
		return logging.logFileMaxSizeMB * 1024 * 1024
	}
	// If "log_file" flag is not specified, the target file (sb.file) will be cleaned up when reaches a fixed size.
	return MaxSize
}

func (sb *syncBuffer) Write(p []byte) (n int, err error) {
	if sb.nbytes+uint64(len(p)) >= sb.maxbytes {
		if err := sb.rotateFile(time.Now(), false); err != nil {
			sb.logger.exit(err)
		}
	}
	n, err = sb.Writer.Write(p)
	sb.nbytes += uint64(n)
	if err != nil {
		sb.logger.exit(err)
	}
	return
}

// rotateFile closes the syncBuffer's file and starts a new one.
// The startup argument indicates whether this is the initial startup of klog.
// If startup is true, existing files are opened for appending instead of truncated.
func (sb *syncBuffer) rotateFile(now time.Time, startup bool) error {
	if sb.file != nil {
		sb.Flush()
		sb.file.Close()
	}
	var err error
	sb.file, _, err = create(severity.Name[sb.sev], now, startup)
	if err != nil {
		return err
	}
	if startup {
		fileInfo, err := sb.file.Stat()
		if err != nil {
			return fmt.Errorf("file stat could not get fileinfo: %v", err)
		}
		// init file size
		sb.nbytes = uint64(fileInfo.Size())
	} else {
		sb.nbytes = 0
	}
	sb.Writer = bufio.NewWriterSize(sb.file, bufferSize)

	if sb.logger.skipLogHeaders {
		return nil
	}

	// Write header.
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Log file created at: %s\n", now.Format("2006/01/02 15:04:05"))
	fmt.Fprintf(&buf, "Running on machine: %s\n", host)
	fmt.Fprintf(&buf, "Binary: Built with %s %s for %s/%s\n", runtime.Compiler, runtime.Version(), runtime.GOOS, runtime.GOARCH)
	fmt.Fprintf(&buf, "Log line format: [IWEF]mmdd hh:mm:ss.uuuuuu threadid file:line] msg\n")
	n, err := sb.file.Write(buf.Bytes())
	sb.nbytes += uint64(n)
	return err
}

// bufferSize sizes the buffer associated with each log file. It's large
// so that log records can accumulate without the logging thread blocking
// on disk I/O. The flushDaemon will block instead.
const bufferSize = 256 * 1024

// createFiles creates all the log files for severity from sev down to infoLog.
// l.mu is held.
func (l *loggingT) createFiles(sev severity.Severity) error {
	interval := l.flushInterval
	if interval == 0 {
		interval = flushInterval
	}
	l.flushD.run(interval)
	now := time.Now()
	// Files are created in decreasing severity order, so as soon as we find one
	// has already been created, we can stop.
	for s := sev; s >= severity.InfoLog && l.file[s] == nil; s-- {
		sb := &syncBuffer{
			logger:   l,
			sev:      s,
			maxbytes: CalculateMaxSize(),
		}
		if err := sb.rotateFile(now, true); err != nil {
			return err
		}
		l.file[s] = sb
	}
	return nil
}

const flushInterval = 5 * time.Second

// flushDaemon periodically flushes the log file buffers.
type flushDaemon struct {
	mu       sync.Mutex
	clock    clock.WithTicker
	flush    func()
	stopC    chan struct{}
	stopDone chan struct{}
}

// newFlushDaemon returns a new flushDaemon. If the passed clock is nil, a
// clock.RealClock is used.
func newFlushDaemon(flush func(), tickClock clock.WithTicker) *flushDaemon {
	if tickClock == nil {
		tickClock = clock.RealClock{}
	}
	return &flushDaemon{
		flush: flush,
		clock: tickClock,
	}
}

// run starts a goroutine that periodically calls the daemons flush function.
// Calling run on an already running daemon will have no effect.
func (f *flushDaemon) run(interval time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.stopC != nil { // daemon already running
		return
	}

	f.stopC = make(chan struct{}, 1)
	f.stopDone = make(chan struct{}, 1)

	ticker := f.clock.NewTicker(interval)
	go func() {
		defer ticker.Stop()
		defer func() { f.stopDone <- struct{}{} }()
		for {
			select {
			case <-ticker.C():
				f.flush()
			case <-f.stopC:
				f.flush()
				return
			}
		}
	}()
}

// stop stops the running flushDaemon and waits until the daemon has shut down.
// Calling stop on a daemon that isn't running will have no effect.
func (f *flushDaemon) stop() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.stopC == nil { // daemon not running
		return
	}

	f.stopC <- struct{}{}
	<-f.stopDone

	f.stopC = nil
	f.stopDone = nil
}

// isRunning returns true if the flush daemon is running.
func (f *flushDaemon) isRunning() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.stopC != nil
}

// StopFlushDaemon stops the flush daemon, if running, and flushes once.
// This prevents klog from leaking goroutines on shutdown. After stopping
// the daemon, you can still manually flush buffers again by calling Flush().
func StopFlushDaemon() {
	logging.flushD.stop()
}

// StartFlushDaemon ensures that the flush daemon runs with the given delay
// between flush calls. If it is already running, it gets restarted.
func StartFlushDaemon(interval time.Duration) {
	StopFlushDaemon()
	logging.flushD.run(interval)
}

// lockAndFlushAll is like flushAll but locks l.mu first.
func (l *loggingT) lockAndFlushAll() {
	l.mu.Lock()
	l.flushAll()
	l.mu.Unlock()
}

// flushAll flushes all the logs and attempts to "sync" their data to disk.
// l.mu is held.
func (l *loggingT) flushAll() {
	// Flush from fatal down, in case there's trouble flushing.
	for s := severity.FatalLog; s >= severity.InfoLog; s-- {
		file := l.file[s]
		if file != nil {
			file.Flush() // ignore error
			file.Sync()  // ignore error
		}
	}
	if logging.loggerOptions.flush != nil {
		logging.loggerOptions.flush()
	}
}

// CopyStandardLogTo arranges for messages written to the Go "log" package's
// default logs to also appear in the Google logs for the named and lower
// severities.  Subsequent changes to the standard log's default output location
// or format may break this behavior.
//
// Valid names are "INFO", "WARNING", "ERROR", and "FATAL".  If the name is not
// recognized, CopyStandardLogTo panics.
func CopyStandardLogTo(name string) {
	sev, ok := severity.ByName(name)
	if !ok {
		panic(fmt.Sprintf("log.CopyStandardLogTo(%q): unrecognized severity name", name))
	}
	// Set a log format that captures the user's file and line:
	//   d.go:23: message
	stdLog.SetFlags(stdLog.Lshortfile)
	stdLog.SetOutput(logBridge(sev))
}

// logBridge provides the Write method that enables CopyStandardLogTo to connect
// Go's standard logs to the logs provided by this package.
type logBridge severity.Severity

// Write parses the standard logging line and passes its components to the
// logger for severity(lb).
func (lb logBridge) Write(b []byte) (n int, err error) {
	var (
		file = "???"
		line = 1
		text string
	)
	// Split "d.go:23: message" into "d.go", "23", and "message".
	if parts := bytes.SplitN(b, []byte{':'}, 3); len(parts) != 3 || len(parts[0]) < 1 || len(parts[2]) < 1 {
		text = fmt.Sprintf("bad log format: %s", b)
	} else {
		file = string(parts[0])
		text = string(parts[2][1:]) // skip leading space
		line, err = strconv.Atoi(string(parts[1]))
		if err != nil {
			text = fmt.Sprintf("bad line number: %s", b)
			line = 1
		}
	}
	// printWithFileLine with alsoToStderr=true, so standard log messages
	// always appear on standard error.
	logging.printWithFileLine(severity.Severity(lb), logging.logger, logging.filter, file, line, true, text)
	return len(b), nil
}

// setV computes and remembers the V level for a given PC
// when vmodule is enabled.
// File pattern matching takes the basename of the file, stripped
// of its .go suffix, and uses filepath.Match, which is a little more
// general than the *? matching used in C++.
// l.mu is held.
func (l *loggingT) setV(pc uintptr) Level {
	fn := runtime.FuncForPC(pc)
	file, _ := fn.FileLine(pc)
	// The file is something like /a/b/c/d.go. We want just the d.
	if strings.HasSuffix(file, ".go") {
		file = file[:len(file)-3]
	}
	if slash := strings.LastIndex(file, "/"); slash >= 0 {
		file = file[slash+1:]
	}
	for _, filter := range l.vmodule.filter {
		if filter.match(file) {
			l.vmap[pc] = filter.level
			return filter.level
		}
	}
	l.vmap[pc] = 0
	return 0
}

// Verbose is a boolean type that implements Infof (like Printf) etc.
// See the documentation of V for more information.
type Verbose struct {
	enabled bool
	logr    *logr.Logger
}

func newVerbose(level Level, b bool) Verbose {
	if logging.logger == nil {
		return Verbose{b, nil}
	}
	v := logging.logger.V(int(level))
	return Verbose{b, &v}
}

// V reports whether verbosity at the call site is at least the requested level.
// The returned value is a struct of type Verbose, which implements Info, Infoln
// and Infof. These methods will write to the Info log if called.
// Thus, one may write either
//	if klog.V(2).Enabled() { klog.Info("log this") }
// or
//	klog.V(2).Info("log this")
// The second form is shorter but the first is cheaper if logging is off because it does
// not evaluate its arguments.
//
// Whether an individual call to V generates a log record depends on the setting of
// the -v and -vmodule flags; both are off by default. The V call will log if its level
// is less than or equal to the value of the -v flag, or alternatively if its level is
// less than or equal to the value of the -vmodule pattern matching the source file
// containing the call.
func V(level Level) Verbose {
	// This function tries hard to be cheap unless there's work to do.
	// The fast path is two atomic loads and compares.

	// Here is a cheap but safe test to see if V logging is enabled globally.
	if logging.verbosity.get() >= level {
		return newVerbose(level, true)
	}

	// It's off globally but vmodule may still be set.
	// Here is another cheap but safe test to see if vmodule is enabled.
	if atomic.LoadInt32(&logging.filterLength) > 0 {
		// Now we need a proper lock to use the logging structure. The pcs field
		// is shared so we must lock before accessing it. This is fairly expensive,
		// but if V logging is enabled we're slow anyway.
		logging.mu.Lock()
		defer logging.mu.Unlock()
		if runtime.Callers(2, logging.pcs[:]) == 0 {
			return newVerbose(level, false)
		}
		// runtime.Callers returns "return PCs", but we want
		// to look up the symbolic information for the call,
		// so subtract 1 from the PC. runtime.CallersFrames
		// would be cleaner, but allocates.
		pc := logging.pcs[0] - 1
		v, ok := logging.vmap[pc]
		if !ok {
			v = logging.setV(pc)
		}
		return newVerbose(level, v >= level)
	}
	return newVerbose(level, false)
}

// Enabled will return true if this log level is enabled, guarded by the value
// of v.
// See the documentation of V for usage.
func (v Verbose) Enabled() bool {
	return v.enabled
}

// Info is equivalent to the global Info function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) Info(args ...interface{}) {
	if v.enabled {
		logging.print(severity.InfoLog, v.logr, logging.filter, args...)
	}
}

// InfoDepth is equivalent to the global InfoDepth function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) InfoDepth(depth int, args ...interface{}) {
	if v.enabled {
		logging.printDepth(severity.InfoLog, v.logr, logging.filter, depth, args...)
	}
}

// Infoln is equivalent to the global Infoln function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) Infoln(args ...interface{}) {
	if v.enabled {
		logging.println(severity.InfoLog, v.logr, logging.filter, args...)
	}
}

// InfolnDepth is equivalent to the global InfolnDepth function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) InfolnDepth(depth int, args ...interface{}) {
	if v.enabled {
		logging.printlnDepth(severity.InfoLog, v.logr, logging.filter, depth, args...)
	}
}

// Infof is equivalent to the global Infof function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) Infof(format string, args ...interface{}) {
	if v.enabled {
		logging.printf(severity.InfoLog, v.logr, logging.filter, format, args...)
	}
}

// InfofDepth is equivalent to the global InfofDepth function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) InfofDepth(depth int, format string, args ...interface{}) {
	if v.enabled {
		logging.printfDepth(severity.InfoLog, v.logr, logging.filter, depth, format, args...)
	}
}

// InfoS is equivalent to the global InfoS function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) InfoS(msg string, keysAndValues ...interface{}) {
	if v.enabled {
		logging.infoS(v.logr, logging.filter, 0, msg, keysAndValues...)
	}
}

// InfoSDepth acts as InfoS but uses depth to determine which call frame to log.
// InfoSDepth(0, "msg") is the same as InfoS("msg").
func InfoSDepth(depth int, msg string, keysAndValues ...interface{}) {
	logging.infoS(logging.logger, logging.filter, depth, msg, keysAndValues...)
}

// InfoSDepth is equivalent to the global InfoSDepth function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) InfoSDepth(depth int, msg string, keysAndValues ...interface{}) {
	if v.enabled {
		logging.infoS(v.logr, logging.filter, depth, msg, keysAndValues...)
	}
}

// Deprecated: Use ErrorS instead.
func (v Verbose) Error(err error, msg string, args ...interface{}) {
	if v.enabled {
		logging.errorS(err, v.logr, logging.filter, 0, msg, args...)
	}
}

// ErrorS is equivalent to the global Error function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) ErrorS(err error, msg string, keysAndValues ...interface{}) {
	if v.enabled {
		logging.errorS(err, v.logr, logging.filter, 0, msg, keysAndValues...)
	}
}

// Info logs to the INFO log.
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Info(args ...interface{}) {
	logging.print(severity.InfoLog, logging.logger, logging.filter, args...)
}

// InfoDepth acts as Info but uses depth to determine which call frame to log.
// InfoDepth(0, "msg") is the same as Info("msg").
func InfoDepth(depth int, args ...interface{}) {
	logging.printDepth(severity.InfoLog, logging.logger, logging.filter, depth, args...)
}

// Infoln logs to the INFO log.
// Arguments are handled in the manner of fmt.Println; a newline is always appended.
func Infoln(args ...interface{}) {
	logging.println(severity.InfoLog, logging.logger, logging.filter, args...)
}

// InfolnDepth acts as Infoln but uses depth to determine which call frame to log.
// InfolnDepth(0, "msg") is the same as Infoln("msg").
func InfolnDepth(depth int, args ...interface{}) {
	logging.printlnDepth(severity.InfoLog, logging.logger, logging.filter, depth, args...)
}

// Infof logs to the INFO log.
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Infof(format string, args ...interface{}) {
	logging.printf(severity.InfoLog, logging.logger, logging.filter, format, args...)
}

// InfofDepth acts as Infof but uses depth to determine which call frame to log.
// InfofDepth(0, "msg", args...) is the same as Infof("msg", args...).
func InfofDepth(depth int, format string, args ...interface{}) {
	logging.printfDepth(severity.InfoLog, logging.logger, logging.filter, depth, format, args...)
}

// InfoS structured logs to the INFO log.
// The msg argument used to add constant description to the log line.
// The key/value pairs would be join by "=" ; a newline is always appended.
//
// Basic examples:
// >> klog.InfoS("Pod status updated", "pod", "kubedns", "status", "ready")
// output:
// >> I1025 00:15:15.525108       1 controller_utils.go:116] "Pod status updated" pod="kubedns" status="ready"
func InfoS(msg string, keysAndValues ...interface{}) {
	logging.infoS(logging.logger, logging.filter, 0, msg, keysAndValues...)
}

// Warning logs to the WARNING and INFO logs.
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Warning(args ...interface{}) {
	logging.print(severity.WarningLog, logging.logger, logging.filter, args...)
}

// WarningDepth acts as Warning but uses depth to determine which call frame to log.
// WarningDepth(0, "msg") is the same as Warning("msg").
func WarningDepth(depth int, args ...interface{}) {
	logging.printDepth(severity.WarningLog, logging.logger, logging.filter, depth, args...)
}

// Warningln logs to the WARNING and INFO logs.
// Arguments are handled in the manner of fmt.Println; a newline is always appended.
func Warningln(args ...interface{}) {
	logging.println(severity.WarningLog, logging.logger, logging.filter, args...)
}

// WarninglnDepth acts as Warningln but uses depth to determine which call frame to log.
// WarninglnDepth(0, "msg") is the same as Warningln("msg").
func WarninglnDepth(depth int, args ...interface{}) {
	logging.printlnDepth(severity.WarningLog, logging.logger, logging.filter, depth, args...)
}

// Warningf logs to the WARNING and INFO logs.
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Warningf(format string, args ...interface{}) {
	logging.printf(severity.WarningLog, logging.logger, logging.filter, format, args...)
}

// WarningfDepth acts as Warningf but uses depth to determine which call frame to log.
// WarningfDepth(0, "msg", args...) is the same as Warningf("msg", args...).
func WarningfDepth(depth int, format string, args ...interface{}) {
	logging.printfDepth(severity.WarningLog, logging.logger, logging.filter, depth, format, args...)
}

// Error logs to the ERROR, WARNING, and INFO logs.
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Error(args ...interface{}) {
	logging.print(severity.ErrorLog, logging.logger, logging.filter, args...)
}

// ErrorDepth acts as Error but uses depth to determine which call frame to log.
// ErrorDepth(0, "msg") is the same as Error("msg").
func ErrorDepth(depth int, args ...interface{}) {
	logging.printDepth(severity.ErrorLog, logging.logger, logging.filter, depth, args...)
}

// Errorln logs to the ERROR, WARNING, and INFO logs.
// Arguments are handled in the manner of fmt.Println; a newline is always appended.
func Errorln(args ...interface{}) {
	logging.println(severity.ErrorLog, logging.logger, logging.filter, args...)
}

// ErrorlnDepth acts as Errorln but uses depth to determine which call frame to log.
// ErrorlnDepth(0, "msg") is the same as Errorln("msg").
func ErrorlnDepth(depth int, args ...interface{}) {
	logging.printlnDepth(severity.ErrorLog, logging.logger, logging.filter, depth, args...)
}

// Errorf logs to the ERROR, WARNING, and INFO logs.
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Errorf(format string, args ...interface{}) {
	logging.printf(severity.ErrorLog, logging.logger, logging.filter, format, args...)
}

// ErrorfDepth acts as Errorf but uses depth to determine which call frame to log.
// ErrorfDepth(0, "msg", args...) is the same as Errorf("msg", args...).
func ErrorfDepth(depth int, format string, args ...interface{}) {
	logging.printfDepth(severity.ErrorLog, logging.logger, logging.filter, depth, format, args...)
}

// ErrorS structured logs to the ERROR, WARNING, and INFO logs.
// the err argument used as "err" field of log line.
// The msg argument used to add constant description to the log line.
// The key/value pairs would be join by "=" ; a newline is always appended.
//
// Basic examples:
// >> klog.ErrorS(err, "Failed to update pod status")
// output:
// >> E1025 00:15:15.525108       1 controller_utils.go:114] "Failed to update pod status" err="timeout"
func ErrorS(err error, msg string, keysAndValues ...interface{}) {
	logging.errorS(err, logging.logger, logging.filter, 0, msg, keysAndValues...)
}

// ErrorSDepth acts as ErrorS but uses depth to determine which call frame to log.
// ErrorSDepth(0, "msg") is the same as ErrorS("msg").
func ErrorSDepth(depth int, err error, msg string, keysAndValues ...interface{}) {
	logging.errorS(err, logging.logger, logging.filter, depth, msg, keysAndValues...)
}

// Fatal logs to the FATAL, ERROR, WARNING, and INFO logs,
// prints stack trace(s), then calls OsExit(255).
//
// Stderr only receives a dump of the current goroutine's stack trace. Log files,
// if there are any, receive a dump of the stack traces in all goroutines.
//
// Callers who want more control over handling of fatal events may instead use a
// combination of different functions:
// - some info or error logging function, optionally with a stack trace
//   value generated by github.com/go-logr/lib/dbg.Backtrace
// - Flush to flush pending log data
// - panic, os.Exit or returning to the caller with an error
//
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Fatal(args ...interface{}) {
	logging.print(severity.FatalLog, logging.logger, logging.filter, args...)
}

// FatalDepth acts as Fatal but uses depth to determine which call frame to log.
// FatalDepth(0, "msg") is the same as Fatal("msg").
func FatalDepth(depth int, args ...interface{}) {
	logging.printDepth(severity.FatalLog, logging.logger, logging.filter, depth, args...)
}

// Fatalln logs to the FATAL, ERROR, WARNING, and INFO logs,
// including a stack trace of all running goroutines, then calls OsExit(255).
// Arguments are handled in the manner of fmt.Println; a newline is always appended.
func Fatalln(args ...interface{}) {
	logging.println(severity.FatalLog, logging.logger, logging.filter, args...)
}

// FatallnDepth acts as Fatalln but uses depth to determine which call frame to log.
// FatallnDepth(0, "msg") is the same as Fatalln("msg").
func FatallnDepth(depth int, args ...interface{}) {
	logging.printlnDepth(severity.FatalLog, logging.logger, logging.filter, depth, args...)
}

// Fatalf logs to the FATAL, ERROR, WARNING, and INFO logs,
// including a stack trace of all running goroutines, then calls OsExit(255).
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Fatalf(format string, args ...interface{}) {
	logging.printf(severity.FatalLog, logging.logger, logging.filter, format, args...)
}

// FatalfDepth acts as Fatalf but uses depth to determine which call frame to log.
// FatalfDepth(0, "msg", args...) is the same as Fatalf("msg", args...).
func FatalfDepth(depth int, format string, args ...interface{}) {
	logging.printfDepth(severity.FatalLog, logging.logger, logging.filter, depth, format, args...)
}

// fatalNoStacks is non-zero if we are to exit without dumping goroutine stacks.
// It allows Exit and relatives to use the Fatal logs.
var fatalNoStacks uint32

// Exit logs to the FATAL, ERROR, WARNING, and INFO logs, then calls OsExit(1).
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Exit(args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.print(severity.FatalLog, logging.logger, logging.filter, args...)
}

// ExitDepth acts as Exit but uses depth to determine which call frame to log.
// ExitDepth(0, "msg") is the same as Exit("msg").
func ExitDepth(depth int, args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.printDepth(severity.FatalLog, logging.logger, logging.filter, depth, args...)
}

// Exitln logs to the FATAL, ERROR, WARNING, and INFO logs, then calls OsExit(1).
func Exitln(args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.println(severity.FatalLog, logging.logger, logging.filter, args...)
}

// ExitlnDepth acts as Exitln but uses depth to determine which call frame to log.
// ExitlnDepth(0, "msg") is the same as Exitln("msg").
func ExitlnDepth(depth int, args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.printlnDepth(severity.FatalLog, logging.logger, logging.filter, depth, args...)
}

// Exitf logs to the FATAL, ERROR, WARNING, and INFO logs, then calls OsExit(1).
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Exitf(format string, args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.printf(severity.FatalLog, logging.logger, logging.filter, format, args...)
}

// ExitfDepth acts as Exitf but uses depth to determine which call frame to log.
// ExitfDepth(0, "msg", args...) is the same as Exitf("msg", args...).
func ExitfDepth(depth int, format string, args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.printfDepth(severity.FatalLog, logging.logger, logging.filter, depth, format, args...)
}

// LogFilter is a collection of functions that can filter all logging calls,
// e.g. for sanitization of arguments and prevent accidental leaking of secrets.
type LogFilter interface {
	Filter(args []interface{}) []interface{}
	FilterF(format string, args []interface{}) (string, []interface{})
	FilterS(msg string, keysAndValues []interface{}) (string, []interface{})
}

// SetLogFilter installs a filter that is used for all log calls.
//
// Modifying the filter is not thread-safe and should be done while no other
// goroutines invoke log calls, usually during program initialization.
func SetLogFilter(filter LogFilter) {
	logging.filter = filter
}
