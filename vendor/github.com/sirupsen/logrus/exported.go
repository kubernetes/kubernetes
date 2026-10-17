package logrus

import (
	"context"
	"io"
	"time"
)

// std is the package-level standard logger, similar to the default logger
// in the stdlib [log] package.
var std = New()

// StandardLogger returns the package-level standard logger used by
// the top-level logging functions.
func StandardLogger() *Logger {
	return std
}

// SetOutput sets the standard logger output.
func SetOutput(out io.Writer) {
	std.SetOutput(out)
}

// SetFormatter sets the standard logger formatter.
func SetFormatter(formatter Formatter) {
	std.SetFormatter(formatter)
}

// SetReportCaller sets whether the standard logger will include the calling
// method as a field.
func SetReportCaller(include bool) {
	std.SetReportCaller(include)
}

// SetLevel sets the standard logger level.
func SetLevel(level Level) {
	std.SetLevel(level)
}

// GetLevel returns the standard logger level.
func GetLevel() Level {
	return std.GetLevel()
}

// IsLevelEnabled checks if logging for the given level is enabled for the standard logger.
func IsLevelEnabled(level Level) bool {
	return std.IsLevelEnabled(level)
}

// AddHook adds a hook to the standard logger hooks.
func AddHook(hook Hook) {
	std.AddHook(hook)
}

// WithError creates an entry from the standard logger and adds an error to it,
// using the value defined in [ErrorKey] as key.
func WithError(err error) *Entry {
	return std.WithError(err)
}

// WithContext creates an entry from the standard logger and adds a context to it.
func WithContext(ctx context.Context) *Entry {
	return std.WithContext(ctx)
}

// WithField creates an entry from the standard logger and adds a single field.
// For multiple fields, prefer [WithFields] over chaining WithField calls.
func WithField(key string, value any) *Entry {
	return std.WithField(key, value)
}

// WithFields creates an entry from the standard logger and adds the fields to it.
func WithFields(fields Fields) *Entry {
	return std.WithFields(fields)
}

// WithTime creates an entry from the standard logger and overrides the time
// used for logs generated with it.
func WithTime(t time.Time) *Entry {
	return std.WithTime(t)
}

// Trace logs a message at level [TraceLevel] on the standard logger.
func Trace(args ...any) {
	std.Trace(args...)
}

// Debug logs a message at level [DebugLevel] on the standard logger.
func Debug(args ...any) {
	std.Debug(args...)
}

// Print logs a message at level [InfoLevel] on the standard logger.
func Print(args ...any) {
	std.Print(args...)
}

// Info logs a message at level [InfoLevel] on the standard logger.
func Info(args ...any) {
	std.Info(args...)
}

// Warn logs a message at level [WarnLevel] on the standard logger.
func Warn(args ...any) {
	std.Warn(args...)
}

// Warning logs a message at level [WarnLevel] on the standard logger.
func Warning(args ...any) {
	std.Warning(args...)
}

// Error logs a message at level [ErrorLevel] on the standard logger.
func Error(args ...any) {
	std.Error(args...)
}

// Panic logs a message at level [PanicLevel] on the standard logger.
func Panic(args ...any) {
	std.Panic(args...)
}

// Fatal logs a message at level [FatalLevel] on the standard logger,
// then exits the process with status 1.
func Fatal(args ...any) {
	std.Fatal(args...)
}

// TraceFn logs a message from a func at level [TraceLevel] on the standard logger.
func TraceFn(fn LogFunction) {
	std.TraceFn(fn)
}

// DebugFn logs a message from a func at level [DebugLevel] on the standard logger.
func DebugFn(fn LogFunction) {
	std.DebugFn(fn)
}

// PrintFn logs a message from a func at level [InfoLevel] on the standard logger.
func PrintFn(fn LogFunction) {
	std.PrintFn(fn)
}

// InfoFn logs a message from a func at level [InfoLevel] on the standard logger.
func InfoFn(fn LogFunction) {
	std.InfoFn(fn)
}

// WarnFn logs a message from a func at level [WarnLevel] on the standard logger.
func WarnFn(fn LogFunction) {
	std.WarnFn(fn)
}

// WarningFn logs a message from a func at level [WarnLevel] on the standard logger.
func WarningFn(fn LogFunction) {
	std.WarningFn(fn)
}

// ErrorFn logs a message from a func at level [ErrorLevel] on the standard logger.
func ErrorFn(fn LogFunction) {
	std.ErrorFn(fn)
}

// PanicFn logs a message from a func at level [PanicLevel] on the standard logger.
func PanicFn(fn LogFunction) {
	std.PanicFn(fn)
}

// FatalFn logs a message from a func at level [FatalLevel] on the standard logger,
// then exits the process with status 1.
func FatalFn(fn LogFunction) {
	std.FatalFn(fn)
}

// Tracef logs a message at level [TraceLevel] on the standard logger.
func Tracef(format string, args ...any) {
	std.Tracef(format, args...)
}

// Debugf logs a message at level [DebugLevel] on the standard logger.
func Debugf(format string, args ...any) {
	std.Debugf(format, args...)
}

// Printf logs a message at level [InfoLevel] on the standard logger.
func Printf(format string, args ...any) {
	std.Printf(format, args...)
}

// Infof logs a message at level [InfoLevel] on the standard logger.
func Infof(format string, args ...any) {
	std.Infof(format, args...)
}

// Warnf logs a message at level [WarnLevel] on the standard logger.
func Warnf(format string, args ...any) {
	std.Warnf(format, args...)
}

// Warningf logs a message at level [WarnLevel] on the standard logger.
func Warningf(format string, args ...any) {
	std.Warningf(format, args...)
}

// Errorf logs a message at level [ErrorLevel] on the standard logger.
func Errorf(format string, args ...any) {
	std.Errorf(format, args...)
}

// Panicf logs a message at level [PanicLevel] on the standard logger.
func Panicf(format string, args ...any) {
	std.Panicf(format, args...)
}

// Fatalf logs a message at level [FatalLevel] on the standard logger,
// then exits the process with status 1.
func Fatalf(format string, args ...any) {
	std.Fatalf(format, args...)
}

// Traceln logs a message at level [TraceLevel] on the standard logger.
func Traceln(args ...any) {
	std.Traceln(args...)
}

// Debugln logs a message at level [DebugLevel] on the standard logger.
func Debugln(args ...any) {
	std.Debugln(args...)
}

// Println logs a message at level [InfoLevel] on the standard logger.
func Println(args ...any) {
	std.Println(args...)
}

// Infoln logs a message at level [InfoLevel] on the standard logger.
func Infoln(args ...any) {
	std.Infoln(args...)
}

// Warnln logs a message at level [WarnLevel] on the standard logger.
func Warnln(args ...any) {
	std.Warnln(args...)
}

// Warningln logs a message at level [WarnLevel] on the standard logger.
func Warningln(args ...any) {
	std.Warningln(args...)
}

// Errorln logs a message at level [ErrorLevel] on the standard logger.
func Errorln(args ...any) {
	std.Errorln(args...)
}

// Panicln logs a message at level [PanicLevel] on the standard logger.
func Panicln(args ...any) {
	std.Panicln(args...)
}

// Fatalln logs a message at level [FatalLevel] on the standard logger,
// then exits the process with status 1.
func Fatalln(args ...any) {
	std.Fatalln(args...)
}
