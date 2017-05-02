//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package utils

import (
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"strings"

	"github.com/lpabon/godbc"
)

type LogLevel int

// Log levels
const (
	LEVEL_NOLOG LogLevel = iota
	LEVEL_CRITICAL
	LEVEL_ERROR
	LEVEL_WARNING
	LEVEL_INFO
	LEVEL_DEBUG
)

var (
	stderr io.Writer = os.Stderr
	stdout io.Writer = os.Stdout
)

type Logger struct {
	critlog, errorlog, infolog *log.Logger
	debuglog, warninglog       *log.Logger

	level LogLevel
}

func logWithLongFile(l *log.Logger, format string, v ...interface{}) {
	_, file, line, _ := runtime.Caller(2)

	// Shorten the path.
	// From
	// /builddir/build/BUILD/heketi-3f4a5b1b6edff87232e8b24533c53b4151ebd9c7/src/github.com/heketi/heketi/apps/glusterfs/volume_entry.go
	// to
	// src/github.com/heketi/heketi/apps/glusterfs/volume_entry.go
	i := strings.Index(file, "/src/")
	if i == -1 {
		i = 0
	}

	l.Print(fmt.Sprintf("%v:%v: ", file[i:], line) +
		fmt.Sprintf(format, v...))
}

// Create a new logger
func NewLogger(prefix string, level LogLevel) *Logger {
	godbc.Require(level >= 0, level)
	godbc.Require(level <= LEVEL_DEBUG, level)

	l := &Logger{}

	if level == LEVEL_NOLOG {
		l.level = LEVEL_DEBUG
	} else {
		l.level = level
	}

	l.critlog = log.New(stderr, prefix+" CRITICAL ", log.LstdFlags)
	l.errorlog = log.New(stderr, prefix+" ERROR ", log.LstdFlags)
	l.warninglog = log.New(stdout, prefix+" WARNING ", log.LstdFlags)
	l.infolog = log.New(stdout, prefix+" INFO ", log.LstdFlags)
	l.debuglog = log.New(stdout, prefix+" DEBUG ", log.LstdFlags)

	godbc.Ensure(l.critlog != nil)
	godbc.Ensure(l.errorlog != nil)
	godbc.Ensure(l.warninglog != nil)
	godbc.Ensure(l.infolog != nil)
	godbc.Ensure(l.debuglog != nil)

	return l
}

// Return current level
func (l *Logger) Level() LogLevel {
	return l.level
}

// Set level
func (l *Logger) SetLevel(level LogLevel) {
	l.level = level
}

// Log critical information
func (l *Logger) Critical(format string, v ...interface{}) {
	if l.level >= LEVEL_CRITICAL {
		logWithLongFile(l.critlog, format, v...)
	}
}

// Log error string
func (l *Logger) LogError(format string, v ...interface{}) error {
	if l.level >= LEVEL_ERROR {
		logWithLongFile(l.errorlog, format, v...)
	}

	return fmt.Errorf(format, v...)
}

// Log error variable
func (l *Logger) Err(err error) error {
	if l.level >= LEVEL_ERROR {
		logWithLongFile(l.errorlog, "%v", err)
	}

	return err
}

// Log warning information
func (l *Logger) Warning(format string, v ...interface{}) {
	if l.level >= LEVEL_WARNING {
		l.warninglog.Printf(format, v...)
	}
}

// Log error variable as a warning
func (l *Logger) WarnErr(err error) error {
	if l.level >= LEVEL_WARNING {
		logWithLongFile(l.warninglog, "%v", err)
	}

	return err
}

// Log string
func (l *Logger) Info(format string, v ...interface{}) {
	if l.level >= LEVEL_INFO {
		l.infolog.Printf(format, v...)
	}
}

// Log string as debug
func (l *Logger) Debug(format string, v ...interface{}) {
	if l.level >= LEVEL_DEBUG {
		logWithLongFile(l.debuglog, format, v...)
	}
}
