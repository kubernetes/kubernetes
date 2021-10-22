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
)

// severity identifies the sort of log: info, warning etc. It also implements
// the flag.Value interface. The -stderrthreshold flag is of type severity and
// should be modified only through the flag.Value interface. The values match
// the corresponding constants in C++.
type severity int32 // sync/atomic int32

// These constants identify the log levels in order of increasing severity.
// A message written to a high-severity log file is also written to each
// lower-severity log file.
const (
	infoLog severity = iota
	warningLog
	errorLog
	fatalLog
	numSeverity = 4
)

const severityChar = "IWEF"

var severityName = []string{
	infoLog:    "INFO",
	warningLog: "WARNING",
	errorLog:   "ERROR",
	fatalLog:   "FATAL",
}

// get returns the value of the severity.
func (s *severity) get() severity {
	return severity(atomic.LoadInt32((*int32)(s)))
}

// set sets the value of the severity.
func (s *severity) set(val severity) {
	atomic.StoreInt32((*int32)(s), int32(val))
}

// String is part of the flag.Value interface.
func (s *severity) String() string {
	return strconv.FormatInt(int64(*s), 10)
}

// Get is part of the flag.Getter interface.
func (s *severity) Get() interface{} {
	return *s
}

// Set is part of the flag.Value interface.
func (s *severity) Set(value string) error {
	var threshold severity
	// Is it a known name?
	if v, ok := severityByName(value); ok {
		threshold = v
	} else {
		v, err := strconv.ParseInt(value, 10, 32)
		if err != nil {
			return err
		}
		threshold = severity(v)
	}
	logging.stderrThreshold.set(threshold)
	return nil
}

func severityByName(s string) (severity, bool) {
	s = strings.ToUpper(s)
	for i, name := range severityName {
		if name == s {
			return severity(i), true
		}
	}
	return 0, false
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

var severityStats = [numSeverity]*OutputStats{
	infoLog:    &Stats.Info,
	warningLog: &Stats.Warning,
	errorLog:   &Stats.Error,
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

// Syntax: -vmodule=recordio=2,file=1,gfs*=3
func (m *moduleSpec) Set(value string) error {
	var filter []modulePat
	for _, pat := range strings.Split(value, ",") {
		if len(pat) == 0 {
			// Empty strings such as from a trailing comma can be ignored.
			continue
		}
		patLev := strings.Split(pat, "=")
		if len(patLev) != 2 || len(patLev[0]) == 0 || len(patLev[1]) == 0 {
			return errVmoduleSyntax
		}
		pattern := patLev[0]
		v, err := strconv.ParseInt(patLev[1], 10, 32)
		if err != nil {
			return errors.New("syntax error: expect comma-separated list of filename=N")
		}
		if v < 0 {
			return errors.New("negative value for vmodule level")
		}
		if v == 0 {
			continue // Ignore. It's harmless but no point in paying the overhead.
		}
		// TODO: check syntax of filter?
		filter = append(filter, modulePat{pattern, isLiteral(pattern), Level(v)})
	}
	logging.mu.Lock()
	defer logging.mu.Unlock()
	logging.setVState(logging.verbosity, filter, true)
	return nil
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

// init sets up the defaults and runs flushDaemon.
func init() {
	logging.stderrThreshold = errorLog // Default stderrThreshold is ERROR.
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
	go logging.flushDaemon()
}

// InitFlags is for explicitly initializing the flags.
func InitFlags(flagset *flag.FlagSet) {
	if flagset == nil {
		flagset = flag.CommandLine
	}

	flagset.StringVar(&logging.logDir, "log_dir", logging.logDir, "If non-empty, write log files in this directory")
	flagset.StringVar(&logging.logFile, "log_file", logging.logFile, "If non-empty, use this log file")
	flagset.Uint64Var(&logging.logFileMaxSizeMB, "log_file_max_size", logging.logFileMaxSizeMB,
		"Defines the maximum size a log file can grow to. Unit is megabytes. "+
			"If the value is 0, the maximum file size is unlimited.")
	flagset.BoolVar(&logging.toStderr, "logtostderr", logging.toStderr, "log to standard error instead of files")
	flagset.BoolVar(&logging.alsoToStderr, "alsologtostderr", logging.alsoToStderr, "log to standard error as well as files")
	flagset.Var(&logging.verbosity, "v", "number for the log level verbosity")
	flagset.BoolVar(&logging.addDirHeader, "add_dir_header", logging.addDirHeader, "If true, adds the file directory to the header of the log messages")
	flagset.BoolVar(&logging.skipHeaders, "skip_headers", logging.skipHeaders, "If true, avoid header prefixes in the log messages")
	flagset.BoolVar(&logging.oneOutput, "one_output", logging.oneOutput, "If true, only write logs to their native severity level (vs also writing to each lower severity level")
	flagset.BoolVar(&logging.skipLogHeaders, "skip_log_headers", logging.skipLogHeaders, "If true, avoid headers when opening log files")
	flagset.Var(&logging.stderrThreshold, "stderrthreshold", "logs at or above this threshold go to stderr")
	flagset.Var(&logging.vmodule, "vmodule", "comma-separated list of pattern=N settings for file-filtered logging")
	flagset.Var(&logging.traceLocation, "log_backtrace_at", "when logging hits line file:N, emit a stack trace")
}

// Flush flushes all pending log I/O.
func Flush() {
	logging.lockAndFlushAll()
}

// loggingT collects all the global state of the logging setup.
type loggingT struct {
	// Boolean flags. Not handled atomically because the flag.Value interface
	// does not let us avoid the =true, and that shorthand is necessary for
	// compatibility. TODO: does this matter enough to fix? Seems unlikely.
	toStderr     bool // The -logtostderr flag.
	alsoToStderr bool // The -alsologtostderr flag.

	// Level flag. Handled atomically.
	stderrThreshold severity // The -stderrthreshold flag.

	// freeList is a list of byte buffers, maintained under freeListMu.
	freeList *buffer
	// freeListMu maintains the free list. It is separate from the main mutex
	// so buffers can be grabbed and printed to without holding the main lock,
	// for better parallelization.
	freeListMu sync.Mutex

	// mu protects the remaining elements of this structure and is
	// used to synchronize logging.
	mu sync.Mutex
	// file holds writer for each of the log types.
	file [numSeverity]flushSyncWriter
	// pcs is used in V to avoid an allocation when computing the caller's PC.
	pcs [1]uintptr
	// vmap is a cache of the V Level for each V() call site, identified by PC.
	// It is wiped whenever the vmodule flag changes state.
	vmap map[uintptr]Level
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

	// If set, all output will be redirected unconditionally to the provided logr.Logger
	logr logr.Logger

	// If true, messages will not be propagated to lower severity log levels
	oneOutput bool

	// If set, all output will be filtered through the filter.
	filter LogFilter
}

// buffer holds a byte Buffer for reuse. The zero value is ready for use.
type buffer struct {
	bytes.Buffer
	tmp  [64]byte // temporary byte array for creating headers.
	next *buffer
}

var logging loggingT

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

// getBuffer returns a new, ready-to-use buffer.
func (l *loggingT) getBuffer() *buffer {
	l.freeListMu.Lock()
	b := l.freeList
	if b != nil {
		l.freeList = b.next
	}
	l.freeListMu.Unlock()
	if b == nil {
		b = new(buffer)
	} else {
		b.next = nil
		b.Reset()
	}
	return b
}

// putBuffer returns a buffer to the free list.
func (l *loggingT) putBuffer(b *buffer) {
	if b.Len() >= 256 {
		// Let big buffers die a natural death.
		return
	}
	l.freeListMu.Lock()
	b.next = l.freeList
	l.freeList = b
	l.freeListMu.Unlock()
}

var timeNow = time.Now // Stubbed out for testing.

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
func (l *loggingT) header(s severity, depth int) (*buffer, string, int) {
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
func (l *loggingT) formatHeader(s severity, file string, line int) *buffer {
	now := timeNow()
	if line < 0 {
		line = 0 // not a real line number, but acceptable to someDigits
	}
	if s > fatalLog {
		s = infoLog // for safety.
	}
	buf := l.getBuffer()
	if l.skipHeaders {
		return buf
	}

	// Avoid Fprintf, for speed. The format is so simple that we can do it quickly by hand.
	// It's worth about 3X. Fprintf is hard.
	_, month, day := now.Date()
	hour, minute, second := now.Clock()
	// Lmmdd hh:mm:ss.uuuuuu threadid file:line]
	buf.tmp[0] = severityChar[s]
	buf.twoDigits(1, int(month))
	buf.twoDigits(3, day)
	buf.tmp[5] = ' '
	buf.twoDigits(6, hour)
	buf.tmp[8] = ':'
	buf.twoDigits(9, minute)
	buf.tmp[11] = ':'
	buf.twoDigits(12, second)
	buf.tmp[14] = '.'
	buf.nDigits(6, 15, now.Nanosecond()/1000, '0')
	buf.tmp[21] = ' '
	buf.nDigits(7, 22, pid, ' ') // TODO: should be TID
	buf.tmp[29] = ' '
	buf.Write(buf.tmp[:30])
	buf.WriteString(file)
	buf.tmp[0] = ':'
	n := buf.someDigits(1, line)
	buf.tmp[n+1] = ']'
	buf.tmp[n+2] = ' '
	buf.Write(buf.tmp[:n+3])
	return buf
}

// Some custom tiny helper functions to print the log header efficiently.

const digits = "0123456789"

// twoDigits formats a zero-prefixed two-digit integer at buf.tmp[i].
func (buf *buffer) twoDigits(i, d int) {
	buf.tmp[i+1] = digits[d%10]
	d /= 10
	buf.tmp[i] = digits[d%10]
}

// nDigits formats an n-digit integer at buf.tmp[i],
// padding with pad on the left.
// It assumes d >= 0.
func (buf *buffer) nDigits(n, i, d int, pad byte) {
	j := n - 1
	for ; j >= 0 && d > 0; j-- {
		buf.tmp[i+j] = digits[d%10]
		d /= 10
	}
	for ; j >= 0; j-- {
		buf.tmp[i+j] = pad
	}
}

// someDigits formats a zero-prefixed variable-width integer at buf.tmp[i].
func (buf *buffer) someDigits(i, d int) int {
	// Print into the top, then copy down. We know there's space for at least
	// a 10-digit number.
	j := len(buf.tmp)
	for {
		j--
		buf.tmp[j] = digits[d%10]
		d /= 10
		if d == 0 {
			break
		}
	}
	return copy(buf.tmp[i:], buf.tmp[j:])
}

func (l *loggingT) println(s severity, logr logr.Logger, filter LogFilter, args ...interface{}) {
	buf, file, line := l.header(s, 0)
	// if logr is set, we clear the generated header as we rely on the backing
	// logr implementation to print headers
	if logr != nil {
		l.putBuffer(buf)
		buf = l.getBuffer()
	}
	if filter != nil {
		args = filter.Filter(args)
	}
	fmt.Fprintln(buf, args...)
	l.output(s, logr, buf, file, line, false)
}

func (l *loggingT) print(s severity, logr logr.Logger, filter LogFilter, args ...interface{}) {
	l.printDepth(s, logr, filter, 1, args...)
}

func (l *loggingT) printDepth(s severity, logr logr.Logger, filter LogFilter, depth int, args ...interface{}) {
	buf, file, line := l.header(s, depth)
	// if logr is set, we clear the generated header as we rely on the backing
	// logr implementation to print headers
	if logr != nil {
		l.putBuffer(buf)
		buf = l.getBuffer()
	}
	if filter != nil {
		args = filter.Filter(args)
	}
	fmt.Fprint(buf, args...)
	if buf.Bytes()[buf.Len()-1] != '\n' {
		buf.WriteByte('\n')
	}
	l.output(s, logr, buf, file, line, false)
}

func (l *loggingT) printf(s severity, logr logr.Logger, filter LogFilter, format string, args ...interface{}) {
	buf, file, line := l.header(s, 0)
	// if logr is set, we clear the generated header as we rely on the backing
	// logr implementation to print headers
	if logr != nil {
		l.putBuffer(buf)
		buf = l.getBuffer()
	}
	if filter != nil {
		format, args = filter.FilterF(format, args)
	}
	fmt.Fprintf(buf, format, args...)
	if buf.Bytes()[buf.Len()-1] != '\n' {
		buf.WriteByte('\n')
	}
	l.output(s, logr, buf, file, line, false)
}

// printWithFileLine behaves like print but uses the provided file and line number.  If
// alsoLogToStderr is true, the log message always appears on standard error; it
// will also appear in the log file unless --logtostderr is set.
func (l *loggingT) printWithFileLine(s severity, logr logr.Logger, filter LogFilter, file string, line int, alsoToStderr bool, args ...interface{}) {
	buf := l.formatHeader(s, file, line)
	// if logr is set, we clear the generated header as we rely on the backing
	// logr implementation to print headers
	if logr != nil {
		l.putBuffer(buf)
		buf = l.getBuffer()
	}
	if filter != nil {
		args = filter.Filter(args)
	}
	fmt.Fprint(buf, args...)
	if buf.Bytes()[buf.Len()-1] != '\n' {
		buf.WriteByte('\n')
	}
	l.output(s, logr, buf, file, line, alsoToStderr)
}

// if loggr is specified, will call loggr.Error, otherwise output with logging module.
func (l *loggingT) errorS(err error, loggr logr.Logger, filter LogFilter, msg string, keysAndValues ...interface{}) {
	if filter != nil {
		msg, keysAndValues = filter.FilterS(msg, keysAndValues)
	}
	if loggr != nil {
		loggr.Error(err, msg, keysAndValues...)
		return
	}
	l.printS(err, msg, keysAndValues...)
}

// if loggr is specified, will call loggr.Info, otherwise output with logging module.
func (l *loggingT) infoS(loggr logr.Logger, filter LogFilter, msg string, keysAndValues ...interface{}) {
	if filter != nil {
		msg, keysAndValues = filter.FilterS(msg, keysAndValues)
	}
	if loggr != nil {
		loggr.Info(msg, keysAndValues...)
		return
	}
	l.printS(nil, msg, keysAndValues...)
}

// printS is called from infoS and errorS if loggr is not specified.
// if err arguments is specified, will output to errorLog severity
func (l *loggingT) printS(err error, msg string, keysAndValues ...interface{}) {
	b := &bytes.Buffer{}
	b.WriteString(fmt.Sprintf("%q", msg))
	if err != nil {
		b.WriteByte(' ')
		b.WriteString(fmt.Sprintf("err=%q", err.Error()))
	}
	kvListFormat(b, keysAndValues...)
	var s severity
	if err == nil {
		s = infoLog
	} else {
		s = errorLog
	}
	l.printDepth(s, logging.logr, nil, 2, b)
}

const missingValue = "(MISSING)"

func kvListFormat(b *bytes.Buffer, keysAndValues ...interface{}) {
	for i := 0; i < len(keysAndValues); i += 2 {
		var v interface{}
		k := keysAndValues[i]
		if i+1 < len(keysAndValues) {
			v = keysAndValues[i+1]
		} else {
			v = missingValue
		}
		b.WriteByte(' ')

		switch v.(type) {
		case string, error:
			b.WriteString(fmt.Sprintf("%s=%q", k, v))
		default:
			if _, ok := v.(fmt.Stringer); ok {
				b.WriteString(fmt.Sprintf("%s=%q", k, v))
			} else {
				b.WriteString(fmt.Sprintf("%s=%+v", k, v))
			}
		}
	}
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

// SetLogger will set the backing logr implementation for klog.
// If set, all log lines will be suppressed from the regular Output, and
// redirected to the logr implementation.
// All log lines include the 'severity', 'file' and 'line' values attached as
// structured logging values.
// Use as:
//   ...
//   klog.SetLogger(zapr.NewLogger(zapLog))
func SetLogger(logr logr.Logger) {
	logging.logr = logr
}

// SetOutput sets the output destination for all severities
func SetOutput(w io.Writer) {
	logging.mu.Lock()
	defer logging.mu.Unlock()
	for s := fatalLog; s >= infoLog; s-- {
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
	sev, ok := severityByName(name)
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
func (l *loggingT) output(s severity, log logr.Logger, buf *buffer, file string, line int, alsoToStderr bool) {
	l.mu.Lock()
	if l.traceLocation.isSet() {
		if l.traceLocation.match(file, line) {
			buf.Write(stacks(false))
		}
	}
	data := buf.Bytes()
	if log != nil {
		// TODO: set 'severity' and caller information as structured log info
		// keysAndValues := []interface{}{"severity", severityName[s], "file", file, "line", line}
		if s == errorLog {
			l.logr.Error(nil, string(data))
		} else {
			log.Info(string(data))
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
			if l.file[infoLog] == nil {
				if err := l.createFiles(infoLog); err != nil {
					os.Stderr.Write(data) // Make sure the message appears somewhere.
					l.exit(err)
				}
			}
			l.file[infoLog].Write(data)
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
				case fatalLog:
					l.file[fatalLog].Write(data)
					fallthrough
				case errorLog:
					l.file[errorLog].Write(data)
					fallthrough
				case warningLog:
					l.file[warningLog].Write(data)
					fallthrough
				case infoLog:
					l.file[infoLog].Write(data)
				}
			}
		}
	}
	if s == fatalLog {
		// If we got here via Exit rather than Fatal, print no stacks.
		if atomic.LoadUint32(&fatalNoStacks) > 0 {
			l.mu.Unlock()
			timeoutFlush(10 * time.Second)
			os.Exit(1)
		}
		// Dump all goroutine stacks before exiting.
		trace := stacks(true)
		// Write the stack trace for all goroutines to the stderr.
		if l.toStderr || l.alsoToStderr || s >= l.stderrThreshold.get() || alsoToStderr {
			os.Stderr.Write(trace)
		}
		// Write the stack trace for all goroutines to the files.
		logExitFunc = func(error) {} // If we get a write error, we'll still exit below.
		for log := fatalLog; log >= infoLog; log-- {
			if f := l.file[log]; f != nil { // Can be nil if -logtostderr is set.
				f.Write(trace)
			}
		}
		l.mu.Unlock()
		timeoutFlush(10 * time.Second)
		os.Exit(255) // C++ uses -1, which is silly because it's anded with 255 anyway.
	}
	l.putBuffer(buf)
	l.mu.Unlock()
	if stats := severityStats[s]; stats != nil {
		atomic.AddInt64(&stats.lines, 1)
		atomic.AddInt64(&stats.bytes, int64(len(data)))
	}
}

// timeoutFlush calls Flush and returns when it completes or after timeout
// elapses, whichever happens first.  This is needed because the hooks invoked
// by Flush may deadlock when klog.Fatal is called from a hook that holds
// a lock.
func timeoutFlush(timeout time.Duration) {
	done := make(chan bool, 1)
	go func() {
		Flush() // calls logging.lockAndFlushAll()
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(timeout):
		fmt.Fprintln(os.Stderr, "klog: Flush took longer than", timeout)
	}
}

// stacks is a wrapper for runtime.Stack that attempts to recover the data for all goroutines.
func stacks(all bool) []byte {
	// We don't know how big the traces are, so grow a few times if they don't fit. Start large, though.
	n := 10000
	if all {
		n = 100000
	}
	var trace []byte
	for i := 0; i < 5; i++ {
		trace = make([]byte, n)
		nbytes := runtime.Stack(trace, all)
		if nbytes < len(trace) {
			return trace[:nbytes]
		}
		n *= 2
	}
	return trace
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
	os.Exit(2)
}

// syncBuffer joins a bufio.Writer to its underlying file, providing access to the
// file's Sync method and providing a wrapper for the Write method that provides log
// file rotation. There are conflicting methods, so the file cannot be embedded.
// l.mu is held for all its methods.
type syncBuffer struct {
	logger *loggingT
	*bufio.Writer
	file     *os.File
	sev      severity
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
	sb.file, _, err = create(severityName[sb.sev], now, startup)
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
func (l *loggingT) createFiles(sev severity) error {
	now := time.Now()
	// Files are created in decreasing severity order, so as soon as we find one
	// has already been created, we can stop.
	for s := sev; s >= infoLog && l.file[s] == nil; s-- {
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
func (l *loggingT) flushDaemon() {
	for range time.NewTicker(flushInterval).C {
		l.lockAndFlushAll()
	}
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
	for s := fatalLog; s >= infoLog; s-- {
		file := l.file[s]
		if file != nil {
			file.Flush() // ignore error
			file.Sync()  // ignore error
		}
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
	sev, ok := severityByName(name)
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
type logBridge severity

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
	logging.printWithFileLine(severity(lb), logging.logr, logging.filter, file, line, true, text)
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
	logr    logr.Logger
	filter  LogFilter
}

func newVerbose(level Level, b bool) Verbose {
	if logging.logr == nil {
		return Verbose{b, nil, logging.filter}
	}
	return Verbose{b, logging.logr.V(int(level)), logging.filter}
}

// V reports whether verbosity at the call site is at least the requested level.
// The returned value is a struct of type Verbose, which implements Info, Infoln
// and Infof. These methods will write to the Info log if called.
// Thus, one may write either
//	if glog.V(2).Enabled() { klog.Info("log this") }
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
		v, ok := logging.vmap[logging.pcs[0]]
		if !ok {
			v = logging.setV(logging.pcs[0])
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
		logging.print(infoLog, v.logr, v.filter, args...)
	}
}

// Infoln is equivalent to the global Infoln function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) Infoln(args ...interface{}) {
	if v.enabled {
		logging.println(infoLog, v.logr, v.filter, args...)
	}
}

// Infof is equivalent to the global Infof function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) Infof(format string, args ...interface{}) {
	if v.enabled {
		logging.printf(infoLog, v.logr, v.filter, format, args...)
	}
}

// InfoS is equivalent to the global InfoS function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) InfoS(msg string, keysAndValues ...interface{}) {
	if v.enabled {
		logging.infoS(v.logr, v.filter, msg, keysAndValues...)
	}
}

// Deprecated: Use ErrorS instead.
func (v Verbose) Error(err error, msg string, args ...interface{}) {
	if v.enabled {
		logging.errorS(err, v.logr, v.filter, msg, args...)
	}
}

// ErrorS is equivalent to the global Error function, guarded by the value of v.
// See the documentation of V for usage.
func (v Verbose) ErrorS(err error, msg string, keysAndValues ...interface{}) {
	if v.enabled {
		logging.errorS(err, v.logr, v.filter, msg, keysAndValues...)
	}
}

// Info logs to the INFO log.
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Info(args ...interface{}) {
	logging.print(infoLog, logging.logr, logging.filter, args...)
}

// InfoDepth acts as Info but uses depth to determine which call frame to log.
// InfoDepth(0, "msg") is the same as Info("msg").
func InfoDepth(depth int, args ...interface{}) {
	logging.printDepth(infoLog, logging.logr, logging.filter, depth, args...)
}

// Infoln logs to the INFO log.
// Arguments are handled in the manner of fmt.Println; a newline is always appended.
func Infoln(args ...interface{}) {
	logging.println(infoLog, logging.logr, logging.filter, args...)
}

// Infof logs to the INFO log.
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Infof(format string, args ...interface{}) {
	logging.printf(infoLog, logging.logr, logging.filter, format, args...)
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
	logging.infoS(logging.logr, logging.filter, msg, keysAndValues...)
}

// Warning logs to the WARNING and INFO logs.
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Warning(args ...interface{}) {
	logging.print(warningLog, logging.logr, logging.filter, args...)
}

// WarningDepth acts as Warning but uses depth to determine which call frame to log.
// WarningDepth(0, "msg") is the same as Warning("msg").
func WarningDepth(depth int, args ...interface{}) {
	logging.printDepth(warningLog, logging.logr, logging.filter, depth, args...)
}

// Warningln logs to the WARNING and INFO logs.
// Arguments are handled in the manner of fmt.Println; a newline is always appended.
func Warningln(args ...interface{}) {
	logging.println(warningLog, logging.logr, logging.filter, args...)
}

// Warningf logs to the WARNING and INFO logs.
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Warningf(format string, args ...interface{}) {
	logging.printf(warningLog, logging.logr, logging.filter, format, args...)
}

// Error logs to the ERROR, WARNING, and INFO logs.
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Error(args ...interface{}) {
	logging.print(errorLog, logging.logr, logging.filter, args...)
}

// ErrorDepth acts as Error but uses depth to determine which call frame to log.
// ErrorDepth(0, "msg") is the same as Error("msg").
func ErrorDepth(depth int, args ...interface{}) {
	logging.printDepth(errorLog, logging.logr, logging.filter, depth, args...)
}

// Errorln logs to the ERROR, WARNING, and INFO logs.
// Arguments are handled in the manner of fmt.Println; a newline is always appended.
func Errorln(args ...interface{}) {
	logging.println(errorLog, logging.logr, logging.filter, args...)
}

// Errorf logs to the ERROR, WARNING, and INFO logs.
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Errorf(format string, args ...interface{}) {
	logging.printf(errorLog, logging.logr, logging.filter, format, args...)
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
	logging.errorS(err, logging.logr, logging.filter, msg, keysAndValues...)
}

// Fatal logs to the FATAL, ERROR, WARNING, and INFO logs,
// including a stack trace of all running goroutines, then calls os.Exit(255).
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Fatal(args ...interface{}) {
	logging.print(fatalLog, logging.logr, logging.filter, args...)
}

// FatalDepth acts as Fatal but uses depth to determine which call frame to log.
// FatalDepth(0, "msg") is the same as Fatal("msg").
func FatalDepth(depth int, args ...interface{}) {
	logging.printDepth(fatalLog, logging.logr, logging.filter, depth, args...)
}

// Fatalln logs to the FATAL, ERROR, WARNING, and INFO logs,
// including a stack trace of all running goroutines, then calls os.Exit(255).
// Arguments are handled in the manner of fmt.Println; a newline is always appended.
func Fatalln(args ...interface{}) {
	logging.println(fatalLog, logging.logr, logging.filter, args...)
}

// Fatalf logs to the FATAL, ERROR, WARNING, and INFO logs,
// including a stack trace of all running goroutines, then calls os.Exit(255).
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Fatalf(format string, args ...interface{}) {
	logging.printf(fatalLog, logging.logr, logging.filter, format, args...)
}

// fatalNoStacks is non-zero if we are to exit without dumping goroutine stacks.
// It allows Exit and relatives to use the Fatal logs.
var fatalNoStacks uint32

// Exit logs to the FATAL, ERROR, WARNING, and INFO logs, then calls os.Exit(1).
// Arguments are handled in the manner of fmt.Print; a newline is appended if missing.
func Exit(args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.print(fatalLog, logging.logr, logging.filter, args...)
}

// ExitDepth acts as Exit but uses depth to determine which call frame to log.
// ExitDepth(0, "msg") is the same as Exit("msg").
func ExitDepth(depth int, args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.printDepth(fatalLog, logging.logr, logging.filter, depth, args...)
}

// Exitln logs to the FATAL, ERROR, WARNING, and INFO logs, then calls os.Exit(1).
func Exitln(args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.println(fatalLog, logging.logr, logging.filter, args...)
}

// Exitf logs to the FATAL, ERROR, WARNING, and INFO logs, then calls os.Exit(1).
// Arguments are handled in the manner of fmt.Printf; a newline is appended if missing.
func Exitf(format string, args ...interface{}) {
	atomic.StoreUint32(&fatalNoStacks, 1)
	logging.printf(fatalLog, logging.logr, logging.filter, format, args...)
}

// LogFilter is a collection of functions that can filter all logging calls,
// e.g. for sanitization of arguments and prevent accidental leaking of secrets.
type LogFilter interface {
	Filter(args []interface{}) []interface{}
	FilterF(format string, args []interface{}) (string, []interface{})
	FilterS(msg string, keysAndValues []interface{}) (string, []interface{})
}

func SetLogFilter(filter LogFilter) {
	logging.mu.Lock()
	defer logging.mu.Unlock()

	logging.filter = filter
}

// ObjectRef references a kubernetes object
type ObjectRef struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace,omitempty"`
}

func (ref ObjectRef) String() string {
	if ref.Namespace != "" {
		return fmt.Sprintf("%s/%s", ref.Namespace, ref.Name)
	}
	return ref.Name
}

// KMetadata is a subset of the kubernetes k8s.io/apimachinery/pkg/apis/meta/v1.Object interface
// this interface may expand in the future, but will always be a subset of the
// kubernetes k8s.io/apimachinery/pkg/apis/meta/v1.Object interface
type KMetadata interface {
	GetName() string
	GetNamespace() string
}

// KObj returns ObjectRef from ObjectMeta
func KObj(obj KMetadata) ObjectRef {
	return ObjectRef{
		Name:      obj.GetName(),
		Namespace: obj.GetNamespace(),
	}
}

// KRef returns ObjectRef from name and namespace
func KRef(namespace, name string) ObjectRef {
	return ObjectRef{
		Name:      name,
		Namespace: namespace,
	}
}
