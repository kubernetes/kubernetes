package logrus

import (
	"context"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// LogFunction For big messages, it can be more efficient to pass a function
// and only call it if the log level is actually enables rather than
// generating the log message and then checking if the level is enabled
type LogFunction func() []any

type Logger struct {
	// The logs are `io.Copy`'d to this in a mutex. It's common to set this to a
	// file, or leave it default which is `os.Stderr`. You can also set this to
	// something more adventurous, such as logging to Kafka.
	Out io.Writer

	// Hooks for the logger instance. These allow firing events based on logging
	// levels and log entries. For example, to send errors to an error tracking
	// service, log to StatsD or dump the core on fatal errors.
	Hooks LevelHooks

	// All log entries pass through the formatter before logged to Out. The
	// included formatters are `TextFormatter` and `JSONFormatter` for which
	// TextFormatter is the default. In development (when a TTY is attached) it
	// logs with colors, but to a file it wouldn't. You can easily implement your
	// own that implements the `Formatter` interface, see the `README` or included
	// formatters for examples.
	Formatter Formatter

	// Flag for whether to log caller info (off by default)
	ReportCaller bool

	// The logging level the logger should log at. This is typically (and defaults
	// to) `logrus.Info`, which allows Info(), Warn(), Error() and Fatal() to be
	// logged.
	Level Level

	// Used to sync writing to the log. Locking is enabled by Default
	mu mutexWrap

	// Reusable empty entry
	entryPool sync.Pool

	// Function to exit the application, defaults to `os.Exit()`
	ExitFunc func(int)

	// The buffer pool used to format the log. If it is nil, the default global
	// buffer pool will be used.
	BufferPool BufferPool
}

// MutexWrap is the mutex implementation used by [Logger].
//
// Deprecated: MutexWrap is an implementation detail of Logger and should not be used directly.
type MutexWrap = mutexWrap

type mutexWrap struct {
	lock     sync.Mutex
	disabled bool
}

func (mw *mutexWrap) Lock() {
	if !mw.disabled {
		mw.lock.Lock()
	}
}

func (mw *mutexWrap) Unlock() {
	if !mw.disabled {
		mw.lock.Unlock()
	}
}

func (mw *mutexWrap) Disable() {
	mw.disabled = true
}

// New Creates a new logger. Configuration should be set by changing [Formatter],
// Out and Hooks directly on the default Logger instance. You can also just
// instantiate your own:
//
//	var log = &logrus.Logger{
//	  Out:       os.Stderr,
//	  Formatter: new(logrus.TextFormatter),
//	  Hooks:     make(logrus.LevelHooks),
//	  Level:     logrus.DebugLevel,
//	}
//
// It's recommended to make this a global instance called `log`.
func New() *Logger {
	return &Logger{
		Out:          os.Stderr,
		Formatter:    new(TextFormatter),
		Hooks:        make(LevelHooks),
		Level:        InfoLevel,
		ExitFunc:     os.Exit,
		ReportCaller: false,
	}
}

func (logger *Logger) newEntry() *Entry {
	entry, ok := logger.entryPool.Get().(*Entry)
	if ok {
		return entry
	}
	return NewEntry(logger)
}

func (logger *Logger) releaseEntry(entry *Entry) {
	entry.Data = map[string]any{}
	logger.entryPool.Put(entry)
}

// WithField allocates a new entry and adds a field to it.
// Debug, Print, Info, Warn, Error, Fatal or Panic must be then applied to
// this new returned entry.
// If you want multiple fields, use `WithFields`.
func (logger *Logger) WithField(key string, value any) *Entry {
	entry := logger.newEntry()
	defer logger.releaseEntry(entry)
	return entry.WithField(key, value)
}

// WithFields adds a struct of fields to the log entry. It calls [Entry.WithField]
// for each Field.
func (logger *Logger) WithFields(fields Fields) *Entry {
	entry := logger.newEntry()
	defer logger.releaseEntry(entry)
	return entry.WithFields(fields)
}

// WithError adds an error as single field to the log entry.  It calls
// [Entry.WithError] for the given error.
func (logger *Logger) WithError(err error) *Entry {
	entry := logger.newEntry()
	defer logger.releaseEntry(entry)
	return entry.WithError(err)
}

// WithContext add a context to the log entry.
func (logger *Logger) WithContext(ctx context.Context) *Entry {
	entry := logger.newEntry()
	defer logger.releaseEntry(entry)
	return entry.WithContext(ctx)
}

// WithTime overrides the time of the log entry.
func (logger *Logger) WithTime(t time.Time) *Entry {
	entry := logger.newEntry()
	defer logger.releaseEntry(entry)
	return entry.WithTime(t)
}

// Logf logs a formatted message at the specified level.
//
// Using Logf with [PanicLevel] or [FatalLevel] intentionally does not
// trigger a panic or exit. Logf treats the level as logging severity only;
// use [Logger.Panicf] or [Logger.Fatalf] when those side effects are desired.
func (logger *Logger) Logf(level Level, format string, args ...any) {
	if logger.IsLevelEnabled(level) {
		entry := logger.newEntry()
		entry.Logf(level, format, args...)
		logger.releaseEntry(entry)
	}
}

func (logger *Logger) Tracef(format string, args ...any) {
	logger.Logf(TraceLevel, format, args...)
}

func (logger *Logger) Debugf(format string, args ...any) {
	logger.Logf(DebugLevel, format, args...)
}

func (logger *Logger) Infof(format string, args ...any) {
	logger.Logf(InfoLevel, format, args...)
}

func (logger *Logger) Printf(format string, args ...any) {
	entry := logger.newEntry()
	entry.Printf(format, args...)
	logger.releaseEntry(entry)
}

func (logger *Logger) Warnf(format string, args ...any) {
	logger.Logf(WarnLevel, format, args...)
}

func (logger *Logger) Warningf(format string, args ...any) {
	logger.Warnf(format, args...)
}

func (logger *Logger) Errorf(format string, args ...any) {
	logger.Logf(ErrorLevel, format, args...)
}

func (logger *Logger) Fatalf(format string, args ...any) {
	logger.Logf(FatalLevel, format, args...)
	logger.Exit(1)
}

func (logger *Logger) Panicf(format string, args ...any) {
	if logger.IsLevelEnabled(PanicLevel) {
		entry := logger.newEntry()
		defer logger.releaseEntry(entry)
		entry.Panicf(format, args...)
	}
}

// Log logs a message at the specified level.
//
// Using Log with [PanicLevel] or [FatalLevel] intentionally does not
// trigger a panic or exit. Log treats the level as logging severity only;
// use [Logger.Panic] or [Logger.Fatal] when those side effects are desired.
func (logger *Logger) Log(level Level, args ...any) {
	if logger.IsLevelEnabled(level) {
		entry := logger.newEntry()
		entry.Log(level, args...)
		logger.releaseEntry(entry)
	}
}

// LogFn logs a message returned by fn at the specified level.
//
// Using LogFn with [PanicLevel] or [FatalLevel] intentionally does not
// trigger a panic or exit. LogFn treats the level as logging severity only;
// use [Logger.PanicFn] or [Logger.FatalFn] when those side effects are desired.
func (logger *Logger) LogFn(level Level, fn LogFunction) {
	if logger.IsLevelEnabled(level) {
		entry := logger.newEntry()
		entry.Log(level, fn()...)
		logger.releaseEntry(entry)
	}
}

func (logger *Logger) Trace(args ...any) {
	logger.Log(TraceLevel, args...)
}

func (logger *Logger) Debug(args ...any) {
	logger.Log(DebugLevel, args...)
}

func (logger *Logger) Info(args ...any) {
	logger.Log(InfoLevel, args...)
}

func (logger *Logger) Print(args ...any) {
	entry := logger.newEntry()
	entry.Print(args...)
	logger.releaseEntry(entry)
}

func (logger *Logger) Warn(args ...any) {
	logger.Log(WarnLevel, args...)
}

func (logger *Logger) Warning(args ...any) {
	logger.Warn(args...)
}

func (logger *Logger) Error(args ...any) {
	logger.Log(ErrorLevel, args...)
}

func (logger *Logger) Fatal(args ...any) {
	logger.Log(FatalLevel, args...)
	logger.Exit(1)
}

func (logger *Logger) Panic(args ...any) {
	if logger.IsLevelEnabled(PanicLevel) {
		entry := logger.newEntry()
		defer logger.releaseEntry(entry)
		entry.Panic(args...)
	}
}

func (logger *Logger) TraceFn(fn LogFunction) {
	logger.LogFn(TraceLevel, fn)
}

func (logger *Logger) DebugFn(fn LogFunction) {
	logger.LogFn(DebugLevel, fn)
}

func (logger *Logger) InfoFn(fn LogFunction) {
	logger.LogFn(InfoLevel, fn)
}

func (logger *Logger) PrintFn(fn LogFunction) {
	entry := logger.newEntry()
	entry.Print(fn()...)
	logger.releaseEntry(entry)
}

func (logger *Logger) WarnFn(fn LogFunction) {
	logger.LogFn(WarnLevel, fn)
}

func (logger *Logger) WarningFn(fn LogFunction) {
	logger.WarnFn(fn)
}

func (logger *Logger) ErrorFn(fn LogFunction) {
	logger.LogFn(ErrorLevel, fn)
}

func (logger *Logger) FatalFn(fn LogFunction) {
	logger.LogFn(FatalLevel, fn)
	logger.Exit(1)
}

func (logger *Logger) PanicFn(fn LogFunction) {
	if logger.IsLevelEnabled(PanicLevel) {
		entry := logger.newEntry()
		defer logger.releaseEntry(entry)
		entry.Panic(fn()...)
	}
}

// Logln logs a message at the specified level with Println-style spacing.
//
// Using Logln with [PanicLevel] or [FatalLevel] intentionally does not
// trigger a panic or exit. Logln treats the level as logging severity only;
// use [Logger.Panicln] or [Logger.Fatalln] when those side effects are desired.
func (logger *Logger) Logln(level Level, args ...any) {
	if logger.IsLevelEnabled(level) {
		entry := logger.newEntry()
		entry.Logln(level, args...)
		logger.releaseEntry(entry)
	}
}

func (logger *Logger) Traceln(args ...any) {
	logger.Logln(TraceLevel, args...)
}

func (logger *Logger) Debugln(args ...any) {
	logger.Logln(DebugLevel, args...)
}

func (logger *Logger) Infoln(args ...any) {
	logger.Logln(InfoLevel, args...)
}

func (logger *Logger) Println(args ...any) {
	entry := logger.newEntry()
	entry.Println(args...)
	logger.releaseEntry(entry)
}

func (logger *Logger) Warnln(args ...any) {
	logger.Logln(WarnLevel, args...)
}

func (logger *Logger) Warningln(args ...any) {
	logger.Warnln(args...)
}

func (logger *Logger) Errorln(args ...any) {
	logger.Logln(ErrorLevel, args...)
}

func (logger *Logger) Fatalln(args ...any) {
	logger.Logln(FatalLevel, args...)
	logger.Exit(1)
}

func (logger *Logger) Panicln(args ...any) {
	if logger.IsLevelEnabled(PanicLevel) {
		entry := logger.newEntry()
		defer logger.releaseEntry(entry)
		entry.Panicln(args...)
	}
}

func (logger *Logger) Exit(code int) {
	runHandlers()
	if logger.ExitFunc == nil {
		logger.ExitFunc = os.Exit
	}
	logger.ExitFunc(code)
}

// SetNoLock disables the lock for situations where a file is opened with
// appending mode, and safe for concurrent writes to the file (within 4k
// message on Linux). In these cases user can choose to disable the lock.
func (logger *Logger) SetNoLock() {
	logger.mu.Disable()
}

func (logger *Logger) level() Level {
	return Level(atomic.LoadUint32((*uint32)(&logger.Level)))
}

// SetLevel sets the logger level.
func (logger *Logger) SetLevel(level Level) {
	atomic.StoreUint32((*uint32)(&logger.Level), uint32(level))
}

// GetLevel returns the logger level.
func (logger *Logger) GetLevel() Level {
	return logger.level()
}

// AddHook adds a hook to the logger hooks.
func (logger *Logger) AddHook(hook Hook) {
	logger.mu.Lock()
	defer logger.mu.Unlock()
	logger.Hooks.Add(hook)
}

// hooksForLevel returns a snapshot of the hooks registered for the given level.
// The returned slice is a shallow copy and may be used without holding logger.mu.
func (logger *Logger) hooksForLevel(level Level) []Hook {
	logger.mu.Lock()
	hooks := logger.Hooks[level]
	if len(hooks) == 0 {
		logger.mu.Unlock()
		return nil
	}
	out := make([]Hook, len(hooks))
	copy(out, hooks)
	logger.mu.Unlock()
	return out
}

// IsLevelEnabled checks if logging for the given level is enabled.
func (logger *Logger) IsLevelEnabled(level Level) bool {
	return logger.level() >= level
}

// SetFormatter sets the logger formatter.
func (logger *Logger) SetFormatter(formatter Formatter) {
	logger.mu.Lock()
	defer logger.mu.Unlock()
	logger.Formatter = formatter
}

// SetOutput sets the logger output.
func (logger *Logger) SetOutput(output io.Writer) {
	logger.mu.Lock()
	defer logger.mu.Unlock()
	logger.Out = output
}

// SetReportCaller sets whether the caller stack frame must be logged.
func (logger *Logger) SetReportCaller(reportCaller bool) {
	logger.mu.Lock()
	defer logger.mu.Unlock()
	logger.ReportCaller = reportCaller
}

// ReplaceHooks replaces the logger hooks and returns the old ones
func (logger *Logger) ReplaceHooks(hooks LevelHooks) LevelHooks {
	logger.mu.Lock()
	defer logger.mu.Unlock()
	oldHooks := logger.Hooks
	logger.Hooks = hooks
	return oldHooks
}

// SetBufferPool sets the logger buffer pool.
func (logger *Logger) SetBufferPool(pool BufferPool) {
	logger.mu.Lock()
	defer logger.mu.Unlock()
	logger.BufferPool = pool
}
