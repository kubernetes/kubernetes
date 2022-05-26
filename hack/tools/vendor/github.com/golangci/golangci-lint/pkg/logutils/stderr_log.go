package logutils

import (
	"fmt"
	"os"
	"time"

	"github.com/sirupsen/logrus" //nolint:depguard

	"github.com/golangci/golangci-lint/pkg/exitcodes"
)

type StderrLog struct {
	name   string
	logger *logrus.Logger
	level  LogLevel
}

var _ Log = NewStderrLog("")

func NewStderrLog(name string) *StderrLog {
	sl := &StderrLog{
		name:   name,
		logger: logrus.New(),
		level:  LogLevelWarn,
	}

	switch os.Getenv("LOG_LEVEL") {
	case "error", "err":
		sl.logger.SetLevel(logrus.ErrorLevel)
	case "warning", "warn":
		sl.logger.SetLevel(logrus.WarnLevel)
	case "info":
		sl.logger.SetLevel(logrus.InfoLevel)
	default:
		sl.logger.SetLevel(logrus.DebugLevel)
	}

	sl.logger.Out = StdErr
	formatter := &logrus.TextFormatter{
		DisableTimestamp: true, // `INFO[0007] msg` -> `INFO msg`
	}
	if os.Getenv("LOG_TIMESTAMP") == "1" {
		formatter.DisableTimestamp = false
		formatter.FullTimestamp = true
		formatter.TimestampFormat = time.StampMilli
	}
	sl.logger.Formatter = formatter

	return sl
}

func (sl StderrLog) prefix() string {
	prefix := ""
	if sl.name != "" {
		prefix = fmt.Sprintf("[%s] ", sl.name)
	}

	return prefix
}

func (sl StderrLog) Fatalf(format string, args ...interface{}) {
	sl.logger.Errorf("%s%s", sl.prefix(), fmt.Sprintf(format, args...))
	os.Exit(exitcodes.Failure)
}

func (sl StderrLog) Panicf(format string, args ...interface{}) {
	v := fmt.Sprintf("%s%s", sl.prefix(), fmt.Sprintf(format, args...))
	panic(v)
}

func (sl StderrLog) Errorf(format string, args ...interface{}) {
	if sl.level > LogLevelError {
		return
	}

	sl.logger.Errorf("%s%s", sl.prefix(), fmt.Sprintf(format, args...))
	// don't call exitIfTest() because the idea is to
	// crash on hidden errors (warnings); but Errorf MUST NOT be
	// called on hidden errors, see log levels comments.
}

func (sl StderrLog) Warnf(format string, args ...interface{}) {
	if sl.level > LogLevelWarn {
		return
	}

	sl.logger.Warnf("%s%s", sl.prefix(), fmt.Sprintf(format, args...))
}

func (sl StderrLog) Infof(format string, args ...interface{}) {
	if sl.level > LogLevelInfo {
		return
	}

	sl.logger.Infof("%s%s", sl.prefix(), fmt.Sprintf(format, args...))
}

func (sl StderrLog) Debugf(format string, args ...interface{}) {
	if sl.level > LogLevelDebug {
		return
	}

	sl.logger.Debugf("%s%s", sl.prefix(), fmt.Sprintf(format, args...))
}

func (sl StderrLog) Child(name string) Log {
	prefix := ""
	if sl.name != "" {
		prefix = sl.name + "/"
	}

	child := sl
	child.name = prefix + name

	return &child
}

func (sl *StderrLog) SetLevel(level LogLevel) {
	sl.level = level
}
