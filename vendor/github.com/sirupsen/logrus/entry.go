package logrus

import (
	"bytes"
	"context"
	"fmt"
	"maps"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (

	// qualified package name, cached at first use
	logrusPackage string

	// Positions in the call stack when tracing to report the calling method.
	//
	// Start at the bottom of the stack before the package-name cache is primed.
	minimumCallerDepth = 1

	// Used for caller information initialisation
	callerInitOnce sync.Once
)

const (
	maximumCallerDepth int = 25
	knownLogrusFrames  int = 4
)

// ErrorKey defines the key when adding errors using [WithError], [Logger.WithError].
var ErrorKey = "error"

// Entry represents a single log event. It may be either an intermediate
// entry (created via WithField(s), WithContext, etc.) or a final entry
// that is emitted when one of the level methods (Trace, Debug, Info,
// Warn, Error, Fatal, Panic) is called.
//
// An Entry always belongs to a Logger. A nil Logger is invalid and will
// cause a panic when the entry is logged. Use [NewEntry] or Logger methods
// to construct entries.
//
// Entries are safe to reuse for adding fields and may be passed around
// to avoid field duplication. Each log operation operates on a copy
// of the Entry’s data to avoid mutation during formatting.
//
//nolint:recvcheck // Entry methods intentionally use both pointer and value receivers.
type Entry struct {
	// Logger is the Logger that owns this entry and is responsible for
	// formatting, hooks, and output. It must not be nil. An Entry without
	// a Logger is invalid and will panic when logged.
	Logger *Logger

	// Data contains all user-defined fields attached to this entry.
	Data Fields

	// Time is the timestamp for the log event. If zero when the entry is
	// logged, it defaults to the current time.
	Time time.Time

	// Level is the severity of the log entry. It is set when the entry
	// is fired and reflects the level used for that log call.
	Level Level

	// Caller contains the calling method information.
	//
	// When [Logger.ReportCaller] is enabled, Caller is populated automatically at
	// log time if it is nil. Hooks and formatters may inspect Caller.
	//
	// Applications generally should not modify Caller unless they intentionally
	// want to provide custom caller information.
	Caller *runtime.Frame

	// Message is the log message supplied to one of the logging methods
	// (Trace, Debug, Info, Warn, Error, Fatal, or Panic). It is set when
	// the entry is logged.
	Message string

	// Buffer is a reusable buffer provided to the formatter. It is set
	// before formatting in the normal log path; when nil, formatters
	// allocate their own.
	Buffer *bytes.Buffer

	// Context carries user-provided context for hooks and formatters.
	Context context.Context

	// err contains internal field-formatting errors.
	err string
}

// NewEntry creates a new [Entry] associated with the provided Logger.
// The logger must not be nil. Passing a nil logger results in a
// panic when a logging method (e.g., [Entry.Info], [Entry.Error], etc.)
// is called.
func NewEntry(logger *Logger) *Entry {
	return &Entry{
		Logger: logger,
		// Reserve default predefined fields and a little extra room.
		Data: make(Fields, defaultFields+3),
	}
}

// Dup creates a copy of the entry for further modification.
//
// Data is cloned to avoid mutating the original entry. Other fields
// (Logger, Time, Context, etc.) are copied by value.
func (entry *Entry) Dup() *Entry {
	dup := entry.dup()
	dup.Data = maps.Clone(entry.Data)
	return dup
}

// dup copies the entry fields shared by derived entries except Data, which
// callers must copy or initialize as appropriate for their use.
func (entry *Entry) dup() *Entry {
	return &Entry{
		Logger:  entry.Logger,
		Time:    entry.Time,
		Caller:  entry.Caller,
		Context: entry.Context,
		err:     entry.err,
	}
}

// Bytes returns the bytes representation of this entry from the formatter.
func (entry *Entry) Bytes() ([]byte, error) {
	// Snapshot the formatter under the lock to protect against concurrent
	// SetFormatter calls, then release the lock before formatting.
	// This avoids a data race and prevents a deadlock if Format() triggers
	// reentrant logging (e.g., a field's MarshalJSON calls logrus).
	//
	// See:
	//
	// - https://github.com/sirupsen/logrus/issues/1440
	// - https://github.com/sirupsen/logrus/issues/1448
	entry.Logger.mu.Lock()
	formatter := entry.Logger.Formatter
	entry.Logger.mu.Unlock()

	return formatter.Format(entry)
}

// String returns the string representation from the reader and ultimately the
// formatter.
func (entry *Entry) String() (string, error) {
	serialized, err := entry.Bytes()
	if err != nil {
		return "", err
	}
	str := string(serialized)
	return str, nil
}

// WithError adds an error as single field (using the key defined in [ErrorKey])
// to the Entry.
func (entry *Entry) WithError(err error) *Entry {
	return entry.WithField(ErrorKey, err)
}

// WithContext adds a context to the Entry.
func (entry *Entry) WithContext(ctx context.Context) *Entry {
	dup := entry.dup()
	dup.Data = maps.Clone(entry.Data)
	dup.Context = ctx
	return dup
}

// WithField adds a single field to the Entry.
func (entry *Entry) WithField(key string, value any) *Entry {
	dup := entry.dup()
	dup.Data = maps.Clone(entry.Data)
	dup.addField(key, value)
	return dup
}

// WithFields adds a map of fields to the Entry.
func (entry *Entry) WithFields(fields Fields) *Entry {
	dup := entry.dup()
	dup.Data = make(Fields, len(entry.Data)+len(fields))
	maps.Copy(dup.Data, entry.Data)

	for key, value := range fields {
		dup.addField(key, value)
	}
	return dup
}

// WithTime overrides the time of the Entry.
func (entry *Entry) WithTime(t time.Time) *Entry {
	dup := entry.dup()
	dup.Data = maps.Clone(entry.Data)
	dup.Time = t
	return dup
}

func (entry *Entry) addField(key string, value any) {
	if _, ok := value.(error); !ok {
		t := reflect.TypeOf(value)
		if t != nil && (t.Kind() == reflect.Func || t.Kind() == reflect.Pointer && t.Elem().Kind() == reflect.Func) {
			if entry.err != "" {
				entry.err += ", skipping unsupported field " + strconv.Quote(key)
			} else {
				entry.err = "skipping unsupported field " + strconv.Quote(key)
			}
			return
		}
	}

	if entry.Data == nil {
		entry.Data = make(Fields, 1)
	}
	entry.Data[key] = value
}

// getPackageName reduces a fully qualified function name to the package name
// There really ought to be a better way...
func getPackageName(f string) string {
	for {
		lastPeriod := strings.LastIndex(f, ".")
		lastSlash := strings.LastIndex(f, "/")
		if lastPeriod > lastSlash {
			f = f[:lastPeriod]
		} else {
			break
		}
	}

	return f
}

// getCaller retrieves the name of the first non-logrus calling function
func getCaller() *runtime.Frame {
	// cache this package's fully-qualified name
	callerInitOnce.Do(func() {
		pcs := make([]uintptr, maximumCallerDepth)
		_ = runtime.Callers(0, pcs)

		// dynamic get the package name and the minimum caller depth
		for i := range maximumCallerDepth {
			funcName := runtime.FuncForPC(pcs[i]).Name()
			if strings.Contains(funcName, "getCaller") {
				logrusPackage = getPackageName(funcName)
				break
			}
		}

		minimumCallerDepth = knownLogrusFrames
	})

	// Restrict the lookback frames to avoid runaway lookups
	pcs := make([]uintptr, maximumCallerDepth)
	depth := runtime.Callers(minimumCallerDepth, pcs)
	frames := runtime.CallersFrames(pcs[:depth])

	for f, again := frames.Next(); again; f, again = frames.Next() {
		pkg := getPackageName(f.Function)

		// If the caller isn't part of this package, we're done
		if pkg != logrusPackage {
			return &f
		}
	}

	// if we got here, we failed to find the caller's context
	return nil
}

// HasCaller reports whether this Entry contains caller information.
//
// Caller may be set explicitly, or populated at log time when
// [Logger.ReportCaller] is enabled.
//
// Deprecated: use [Entry.Caller] != nil instead.
//
//go:fix inline
func (entry Entry) HasCaller() bool {
	return entry.Caller != nil
}

func (entry *Entry) logArgs(level Level, panicAfter bool, args ...any) {
	entry.log(level, panicAfter, sprint(args...))
}

func (entry *Entry) logf(level Level, panicAfter bool, format string, args ...any) {
	entry.log(level, panicAfter, fmt.Sprintf(format, args...))
}

// logln uses Sprintln for multiple arguments to preserve Println-style
// spacing between args, then trims the trailing newline.
func (entry *Entry) logln(level Level, panicAfter bool, args ...any) {
	if len(args) <= 1 {
		entry.log(level, panicAfter, sprint(args...))
		return
	}
	msg := fmt.Sprintln(args...)
	msg = msg[:len(msg)-1] // Trim the newline added by Sprintln; logging adds its own.
	entry.log(level, panicAfter, msg)
}

// log writes msg at level. If panicAfter is true, it panics with the fully
// populated entry after hooks and output have completed.
//
// The explicit flag keeps panic behavior limited to Panic, Panicf, and
// Panicln while avoiding a return value used only as the panic value.
// See #1283 and commits f96066e and 5f8c666.
func (entry *Entry) log(level Level, panicAfter bool, msg string) {
	newEntry := entry.dup()
	newEntry.Data = maps.Clone(entry.Data)

	if newEntry.Time.IsZero() {
		newEntry.Time = time.Now()
	}

	newEntry.Level = level
	newEntry.Message = msg

	logger := newEntry.Logger
	logger.mu.Lock()
	reportCaller := logger.ReportCaller
	bufPool := newEntry.getBufferPool()
	logger.mu.Unlock()

	// Preserve explicitly set caller information.
	if reportCaller && newEntry.Caller == nil {
		newEntry.Caller = getCaller()
	}

	// Select hooks based on the level for this log call. Hooks receive the
	// Entry and may mutate it, but that does not affect which hooks are
	// fired for this event.
	hooks := logger.hooksForLevel(level)
	newEntry.fireHooks(hooks)

	buffer := bufPool.Get()
	defer func() {
		newEntry.Buffer = nil
		buffer.Reset()
		bufPool.Put(buffer)
	}()
	buffer.Reset()
	newEntry.Buffer = buffer
	newEntry.write()
	newEntry.Buffer = nil

	// Panic here so the panic value contains the fully populated entry without
	// requiring log to return it to the caller.
	if panicAfter {
		panic(newEntry)
	}
}

func (entry *Entry) getBufferPool() (pool BufferPool) {
	if entry.Logger.BufferPool != nil {
		return entry.Logger.BufferPool
	}
	return bufferPool
}

func (entry *Entry) fireHooks(hooks []Hook) {
	for _, hook := range hooks {
		if err := hook.Fire(entry); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, "Failed to fire hook:", err)
			return
		}
	}
}

func (entry *Entry) write() {
	// Snapshot the formatter under the lock to protect against concurrent
	// SetFormatter calls, then release the lock before formatting.
	// This avoids a deadlock when Format() triggers reentrant logging (e.g.,
	// a field's MarshalJSON calls logrus). See #1448, #1440.
	entry.Logger.mu.Lock()
	formatter := entry.Logger.Formatter
	entry.Logger.mu.Unlock()

	serialized, err := formatter.Format(entry)
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "Failed to format entry:", err)
		return
	}

	// Re-acquire the lock to serialize writes to the underlying io.Writer.
	entry.Logger.mu.Lock()
	defer entry.Logger.mu.Unlock()
	if _, err := entry.Logger.Out.Write(serialized); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "Failed to write to log:", err)
	}
}

// Log logs a message at the specified level.
//
// Using Log with [PanicLevel] or [FatalLevel] intentionally does not
// trigger a panic or exit. Log treats the level as logging severity only;
// use [Entry.Panic] or [Entry.Fatal] when those side effects are desired.
func (entry *Entry) Log(level Level, args ...any) {
	const panicAfter = false
	if entry.Logger.IsLevelEnabled(level) {
		entry.logArgs(level, panicAfter, args...)
	}
}

func (entry *Entry) Trace(args ...any) {
	entry.Log(TraceLevel, args...)
}

func (entry *Entry) Debug(args ...any) {
	entry.Log(DebugLevel, args...)
}

func (entry *Entry) Print(args ...any) {
	entry.Info(args...)
}

func (entry *Entry) Info(args ...any) {
	entry.Log(InfoLevel, args...)
}

func (entry *Entry) Warn(args ...any) {
	entry.Log(WarnLevel, args...)
}

func (entry *Entry) Warning(args ...any) {
	entry.Warn(args...)
}

func (entry *Entry) Error(args ...any) {
	entry.Log(ErrorLevel, args...)
}

func (entry *Entry) Fatal(args ...any) {
	entry.Log(FatalLevel, args...)
	entry.Logger.Exit(1)
}

func (entry *Entry) Panic(args ...any) {
	const panicAfter = true
	if entry.Logger.IsLevelEnabled(PanicLevel) {
		entry.logArgs(PanicLevel, panicAfter, args...)
	}
}

// Entry Printf family functions

// Logf logs a formatted message at the specified level.
//
// Using Logf with [PanicLevel] or [FatalLevel] intentionally does not
// trigger a panic or exit. Logf treats the level as logging severity only;
// use [Entry.Panicf] or [Entry.Fatalf] when those side effects are desired.
func (entry *Entry) Logf(level Level, format string, args ...any) {
	const panicAfter = false
	if entry.Logger.IsLevelEnabled(level) {
		entry.logf(level, panicAfter, format, args...)
	}
}

func (entry *Entry) Tracef(format string, args ...any) {
	entry.Logf(TraceLevel, format, args...)
}

func (entry *Entry) Debugf(format string, args ...any) {
	entry.Logf(DebugLevel, format, args...)
}

func (entry *Entry) Infof(format string, args ...any) {
	entry.Logf(InfoLevel, format, args...)
}

func (entry *Entry) Printf(format string, args ...any) {
	entry.Infof(format, args...)
}

func (entry *Entry) Warnf(format string, args ...any) {
	entry.Logf(WarnLevel, format, args...)
}

func (entry *Entry) Warningf(format string, args ...any) {
	entry.Warnf(format, args...)
}

func (entry *Entry) Errorf(format string, args ...any) {
	entry.Logf(ErrorLevel, format, args...)
}

func (entry *Entry) Fatalf(format string, args ...any) {
	entry.Logf(FatalLevel, format, args...)
	entry.Logger.Exit(1)
}

func (entry *Entry) Panicf(format string, args ...any) {
	const panicAfter = true
	if entry.Logger.IsLevelEnabled(PanicLevel) {
		entry.logf(PanicLevel, panicAfter, format, args...)
	}
}

// Entry Println family functions

// Logln logs a message at the specified level with Println-style spacing.
//
// Using Logln with [PanicLevel] or [FatalLevel] intentionally does not
// trigger a panic or exit. Logln treats the level as logging severity only;
// use [Entry.Panicln] or [Entry.Fatalln] when those side effects are desired.
func (entry *Entry) Logln(level Level, args ...any) {
	const panicAfter = false
	if entry.Logger.IsLevelEnabled(level) {
		entry.logln(level, panicAfter, args...)
	}
}

func (entry *Entry) Traceln(args ...any) {
	entry.Logln(TraceLevel, args...)
}

func (entry *Entry) Debugln(args ...any) {
	entry.Logln(DebugLevel, args...)
}

func (entry *Entry) Infoln(args ...any) {
	entry.Logln(InfoLevel, args...)
}

func (entry *Entry) Println(args ...any) {
	entry.Infoln(args...)
}

func (entry *Entry) Warnln(args ...any) {
	entry.Logln(WarnLevel, args...)
}

func (entry *Entry) Warningln(args ...any) {
	entry.Warnln(args...)
}

func (entry *Entry) Errorln(args ...any) {
	entry.Logln(ErrorLevel, args...)
}

func (entry *Entry) Fatalln(args ...any) {
	entry.Logln(FatalLevel, args...)
	entry.Logger.Exit(1)
}

func (entry *Entry) Panicln(args ...any) {
	const panicAfter = true
	if entry.Logger.IsLevelEnabled(PanicLevel) {
		entry.logln(PanicLevel, panicAfter, args...)
	}
}

// sprint is fmt.Sprint with fast paths for zero or one string argument.
func sprint(args ...any) string {
	switch len(args) {
	case 0:
		return ""
	case 1:
		if msg, ok := args[0].(string); ok {
			return msg
		}
	}
	return fmt.Sprint(args...)
}
