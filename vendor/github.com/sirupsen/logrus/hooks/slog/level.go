package slog

import (
	"log/slog"

	"github.com/sirupsen/logrus"
)

// slog levels corresponding to Logrus levels.
const (
	slogLevelTrace = slog.LevelDebug - 4
	slogLevelDebug = slog.LevelDebug
	slogLevelInfo  = slog.LevelInfo
	slogLevelWarn  = slog.LevelWarn
	slogLevelError = slog.LevelError
	slogLevelFatal = slog.LevelError + 2
	slogLevelPanic = slog.LevelError + 4
)

// A Leveler provides a [logrus.Level] value.
//
// It is the Logrus counterpart of [slog.Leveler].
//
// As [SlogLevel] itself implements Leveler, clients typically supply
// a SlogLevel value wherever a Leveler is needed.
// Clients who need to vary the level dynamically can provide a more complex
// Leveler implementation.
type Leveler interface {
	Level() logrus.Level
}

// Level adapts a [logrus.Level] to a [slog.Leveler] using the default
// Logrus-to-slog level mapping:
//
//   - [logrus.TraceLevel] -> [slog.LevelDebug] - 4
//   - [logrus.DebugLevel] -> [slog.LevelDebug]
//   - [logrus.InfoLevel] -> [slog.LevelInfo]
//   - [logrus.WarnLevel] -> [slog.LevelWarn]
//   - [logrus.ErrorLevel] -> [slog.LevelError]
//   - [logrus.FatalLevel] -> [slog.LevelError] + 2
//   - [logrus.PanicLevel] -> [slog.LevelError] + 4
//
// Unknown Logrus levels map to [slog.LevelError].
type Level logrus.Level

// Level returns l mapped to the corresponding slog level.
func (l Level) Level() slog.Level {
	return toSlogLevel(logrus.Level(l))
}

var _ slog.Leveler = Level(0)

// SlogLevel adapts a [slog.Level] to a [Leveler] using the default
// slog-to-Logrus level mapping:
//
//   - [slog.LevelDebug] - 4 -> [logrus.TraceLevel]
//   - [slog.LevelDebug] -> [logrus.DebugLevel]
//   - [slog.LevelInfo] -> [logrus.InfoLevel]
//   - [slog.LevelWarn] -> [logrus.WarnLevel]
//   - [slog.LevelError] -> [logrus.ErrorLevel]
//   - [slog.LevelError] + 2 -> [logrus.FatalLevel]
//   - [slog.LevelError] + 4 -> [logrus.PanicLevel]
//
// Levels between these boundaries map to the next lower Logrus severity.
// Levels below [slog.LevelDebug] map to [logrus.TraceLevel], and levels at or
// above the Panic boundary map to [logrus.PanicLevel].
type SlogLevel slog.Level

// Level returns l mapped to the corresponding Logrus level.
func (l SlogLevel) Level() logrus.Level {
	return toLogrusLevel(slog.Level(l))
}

var _ Leveler = SlogLevel(0)

// toLogrusLevel maps a slog level using the default mapping.
func toLogrusLevel(level slog.Level) logrus.Level {
	switch {
	case level >= slogLevelPanic:
		return logrus.PanicLevel
	case level >= slogLevelFatal:
		return logrus.FatalLevel
	case level >= slogLevelError:
		return logrus.ErrorLevel
	case level >= slogLevelWarn:
		return logrus.WarnLevel
	case level >= slogLevelInfo:
		return logrus.InfoLevel
	case level >= slogLevelDebug:
		return logrus.DebugLevel
	case level >= slogLevelTrace:
		return logrus.TraceLevel
	default:
		return logrus.TraceLevel
	}
}

// toSlogLevel maps a Logrus level using the default mapping.
func toSlogLevel(level logrus.Level) slog.Level {
	switch level {
	case logrus.PanicLevel:
		return slogLevelPanic
	case logrus.FatalLevel:
		return slogLevelFatal
	case logrus.ErrorLevel:
		return slogLevelError
	case logrus.WarnLevel:
		return slogLevelWarn
	case logrus.InfoLevel:
		return slogLevelInfo
	case logrus.DebugLevel:
		return slogLevelDebug
	case logrus.TraceLevel:
		return slogLevelTrace
	default:
		// Treat all unknown levels as errors
		return slogLevelError
	}
}
