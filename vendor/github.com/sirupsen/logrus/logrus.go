package logrus

import (
	"bytes"
	"fmt"
	"log"
)

// Fields type, used to pass to [WithFields].
type Fields map[string]any

// Level type
//
//nolint:recvcheck // the methods of "Entry" use pointer receiver and non-pointer receiver.
type Level uint32

// Convert the Level to a string. E.g. [PanicLevel] becomes "panic".
func (level Level) String() string {
	switch level {
	case TraceLevel:
		return "trace"
	case DebugLevel:
		return "debug"
	case InfoLevel:
		return "info"
	case WarnLevel:
		return "warning"
	case ErrorLevel:
		return "error"
	case FatalLevel:
		return "fatal"
	case PanicLevel:
		return "panic"
	default:
		return "unknown"
	}
}

// ParseLevel takes a string level and returns the Logrus log level constant.
func ParseLevel(lvl string) (Level, error) {
	return parseLevel([]byte(lvl))
}

func parseLevel(b []byte) (Level, error) {
	switch {
	case bytes.EqualFold(b, []byte("panic")):
		return PanicLevel, nil
	case bytes.EqualFold(b, []byte("fatal")):
		return FatalLevel, nil
	case bytes.EqualFold(b, []byte("error")):
		return ErrorLevel, nil
	case bytes.EqualFold(b, []byte("warn")),
		bytes.EqualFold(b, []byte("warning")):
		return WarnLevel, nil
	case bytes.EqualFold(b, []byte("info")):
		return InfoLevel, nil
	case bytes.EqualFold(b, []byte("debug")):
		return DebugLevel, nil
	case bytes.EqualFold(b, []byte("trace")):
		return TraceLevel, nil
	default:
		return 0, fmt.Errorf("not a valid logrus Level: %q", b)
	}
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (level *Level) UnmarshalText(text []byte) error {
	l, err := parseLevel(text)
	if err != nil {
		return err
	}

	*level = l

	return nil
}

func (level Level) MarshalText() ([]byte, error) {
	switch level {
	case TraceLevel, DebugLevel, InfoLevel, WarnLevel, ErrorLevel, FatalLevel, PanicLevel:
		return []byte(level.String()), nil
	default:
		return nil, fmt.Errorf("not a valid logrus level %d", level)
	}
}

// AllLevels exposing all logging levels.
var AllLevels = []Level{
	PanicLevel,
	FatalLevel,
	ErrorLevel,
	WarnLevel,
	InfoLevel,
	DebugLevel,
	TraceLevel,
}

// These are the different logging levels. You can set the logging level to log
// on your instance of logger, obtained with [logrus.New].
const (
	// PanicLevel level, highest level of severity. Logs and then calls panic with the
	// message passed to Debug, Info, ...
	PanicLevel Level = iota
	// FatalLevel level. Logs and then calls `logger.Exit(1)`. It will exit even if the
	// logging level is set to Panic.
	FatalLevel
	// ErrorLevel level. Logs. Used for errors that should definitely be noted.
	// Commonly used for hooks to send errors to an error tracking service.
	ErrorLevel
	// WarnLevel level. Non-critical entries that deserve eyes.
	WarnLevel
	// InfoLevel level. General operational entries about what's going on inside the
	// application.
	InfoLevel
	// DebugLevel level. Usually only enabled when debugging. Very verbose logging.
	DebugLevel
	// TraceLevel level. Designates finer-grained informational events than the Debug.
	TraceLevel
)

// Compile-time interface assertions.
var (
	_ StdLogger = (*log.Logger)(nil)
	_ StdLogger = (*Entry)(nil)
	_ StdLogger = (*Logger)(nil)

	_ FieldLogger = (*Logger)(nil)
	_ FieldLogger = (*Entry)(nil)
	_ FieldLogger = Ext1FieldLogger(nil)

	_ DebugLogger = (*Logger)(nil)
	_ InfoLogger  = (*Logger)(nil)
	_ WarnLogger  = (*Logger)(nil)
	_ ErrorLogger = (*Logger)(nil)
	_ TraceLogger = (*Logger)(nil)

	_ DebugLogger = (*Entry)(nil)
	_ InfoLogger  = (*Entry)(nil)
	_ WarnLogger  = (*Entry)(nil)
	_ ErrorLogger = (*Entry)(nil)
	_ TraceLogger = (*Entry)(nil)

	_ Ext1FieldLogger = (*Logger)(nil)
	_ Ext1FieldLogger = (*Entry)(nil)
)

// StdLogger is what your logrus-enabled library should take, that way
// it'll accept a stdlib logger ([log.Logger]) and a logrus logger.
// There's no standard interface, so this is the closest we get, unfortunately.
type StdLogger interface {
	Print(args ...any)
	Printf(format string, args ...any)
	Println(args ...any)

	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Fatalln(args ...any)

	Panic(args ...any)
	Panicf(format string, args ...any)
	Panicln(args ...any)
}

// FieldLogger extends the [StdLogger] interface, generalizing
// the [Entry] and [Logger] types.
type FieldLogger interface {
	WithField(key string, value any) *Entry
	WithFields(fields Fields) *Entry
	WithError(err error) *Entry

	StdLogger
	DebugLogger
	InfoLogger
	WarnLogger
	ErrorLogger

	// Legacy warning aliases. These are kept on FieldLogger for backwards
	// compatibility, but are intentionally omitted from [WarnLogger].

	Warning(args ...any)
	Warningf(format string, args ...any)
	Warningln(args ...any)
}

// DebugLogger provides convenience functions to log messages at level [DebugLevel].
type DebugLogger interface {
	Debug(args ...any)
	Debugf(format string, args ...any)
	Debugln(args ...any)
}

// InfoLogger provides convenience functions to log messages at level [InfoLevel].
type InfoLogger interface {
	Info(args ...any)
	Infof(format string, args ...any)
	Infoln(args ...any)
}

// WarnLogger provides convenience functions to log messages at level [WarnLevel].
type WarnLogger interface {
	Warn(args ...any)
	Warnf(format string, args ...any)
	Warnln(args ...any)
}

// ErrorLogger provides convenience functions to log messages at level [ErrorLevel].
type ErrorLogger interface {
	Error(args ...any)
	Errorf(format string, args ...any)
	Errorln(args ...any)
}

// TraceLogger provides convenience functions to log messages at level [TraceLevel].
type TraceLogger interface {
	Trace(args ...any)
	Tracef(format string, args ...any)
	Traceln(args ...any)
}

// Ext1FieldLogger is FieldLogger extended with Trace-level methods.
//
// New code should prefer the smallest applicable interface, such as
// [FieldLogger] or [TraceLogger], or use [Logger] or [Entry] directly.
type Ext1FieldLogger interface {
	FieldLogger
	TraceLogger
}
