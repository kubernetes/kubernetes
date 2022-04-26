package report

import (
	"fmt"
	"strings"

	"github.com/golangci/golangci-lint/pkg/logutils"
)

type LogWrapper struct {
	rd      *Data
	tags    []string
	origLog logutils.Log
}

func NewLogWrapper(log logutils.Log, reportData *Data) *LogWrapper {
	return &LogWrapper{
		rd:      reportData,
		origLog: log,
	}
}

func (lw LogWrapper) Fatalf(format string, args ...interface{}) {
	lw.origLog.Fatalf(format, args...)
}

func (lw LogWrapper) Panicf(format string, args ...interface{}) {
	lw.origLog.Panicf(format, args...)
}

func (lw LogWrapper) Errorf(format string, args ...interface{}) {
	lw.origLog.Errorf(format, args...)
	lw.rd.Error = fmt.Sprintf(format, args...)
}

func (lw LogWrapper) Warnf(format string, args ...interface{}) {
	lw.origLog.Warnf(format, args...)
	w := Warning{
		Tag:  strings.Join(lw.tags, "/"),
		Text: fmt.Sprintf(format, args...),
	}

	lw.rd.Warnings = append(lw.rd.Warnings, w)
}

func (lw LogWrapper) Infof(format string, args ...interface{}) {
	lw.origLog.Infof(format, args...)
}

func (lw LogWrapper) Child(name string) logutils.Log {
	c := lw
	c.origLog = lw.origLog.Child(name)
	c.tags = append([]string{}, lw.tags...)
	c.tags = append(c.tags, name)
	return c
}

func (lw LogWrapper) SetLevel(level logutils.LogLevel) {
	lw.origLog.SetLevel(level)
}

func (lw LogWrapper) GoString() string {
	return fmt.Sprintf("lw: %+v, orig log: %#v", lw, lw.origLog)
}
