package log
// Copyright 2013, CoreOS, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// author: David Fisher <ddf1991@gmail.com>
// based on previous package by: Cong Ding <dinggnu@gmail.com>

import (
	"fmt"
	"os"
)

var BasicFormat = "%s [%9s] %s- %s\n"
var BasicFields = []string{"time", "priority", "prefix", "message"}
var RichFormat = "%s [%9s] %d %s - %s:%s:%d - %s\n"
var RichFields = []string{"full_time", "priority", "seq", "prefix", "filename", "funcname", "lineno", "message"}

// This function has an unusual name to aid in finding it while walking the
// stack. We need to do some dead reckoning from this function to access the
// caller's stack, so there is a consistent call depth above this function.
func (logger *Logger) Log(priority Priority, v ...interface{}) {
	fields := logger.fieldValues()
	fields["priority"] = priority
	fields["message"] = fmt.Sprint(v...)
	for _, sink := range logger.sinks {
		sink.Log(fields)
	}
}

func (logger *Logger) Logf(priority Priority, format string, v ...interface{}) {
	logger.Log(priority, fmt.Sprintf(format, v...))
}


func (logger *Logger) Emergency(v ...interface{}) {
	logger.Log(PriEmerg, v...)
}
func (logger *Logger) Emergencyf(format string, v ...interface{}) {
	logger.Log(PriEmerg, fmt.Sprintf(format, v...))
}

func (logger *Logger) Alert(v ...interface{}) {
	logger.Log(PriAlert, v...)
}
func (logger *Logger) Alertf(format string, v ...interface{}) {
	logger.Log(PriAlert, fmt.Sprintf(format, v...))
}

func (logger *Logger) Critical(v ...interface{}) {
	logger.Log(PriCrit, v...)
}
func (logger *Logger) Criticalf(format string, v ...interface{}) {
	logger.Log(PriCrit, fmt.Sprintf(format, v...))
}

func (logger *Logger) Error(v ...interface{}) {
	logger.Log(PriErr, v...)
}
func (logger *Logger) Errorf(format string, v ...interface{}) {
	logger.Log(PriErr, fmt.Sprintf(format, v...))
}

func (logger *Logger) Warning(v ...interface{}) {
	logger.Log(PriWarning, v...)
}
func (logger *Logger) Warningf(format string, v ...interface{}) {
	logger.Log(PriWarning, fmt.Sprintf(format, v...))
}

func (logger *Logger) Notice(v ...interface{}) {
	logger.Log(PriNotice, v...)
}
func (logger *Logger) Noticef(format string, v ...interface{}) {
	logger.Log(PriNotice, fmt.Sprintf(format, v...))
}

func (logger *Logger) Info(v ...interface{}) {
	logger.Log(PriInfo, v...)
}
func (logger *Logger) Infof(format string, v ...interface{}) {
	logger.Log(PriInfo, fmt.Sprintf(format, v...))
}

func (logger *Logger) Debug(v ...interface{}) {
	logger.Log(PriDebug, v...)
}
func (logger *Logger) Debugf(format string, v ...interface{}) {
	logger.Log(PriDebug, fmt.Sprintf(format, v...))
}


func Emergency(v ...interface{}) {
	defaultLogger.Log(PriEmerg, v...)
}
func Emergencyf(format string, v ...interface{}) {
	defaultLogger.Log(PriEmerg, fmt.Sprintf(format, v...))
}

func Alert(v ...interface{}) {
	defaultLogger.Log(PriAlert, v...)
}
func Alertf(format string, v ...interface{}) {
	defaultLogger.Log(PriAlert, fmt.Sprintf(format, v...))
}

func Critical(v ...interface{}) {
	defaultLogger.Log(PriCrit, v...)
}
func Criticalf(format string, v ...interface{}) {
	defaultLogger.Log(PriCrit, fmt.Sprintf(format, v...))
}

func Error(v ...interface{}) {
	defaultLogger.Log(PriErr, v...)
}
func Errorf(format string, v ...interface{}) {
	defaultLogger.Log(PriErr, fmt.Sprintf(format, v...))
}

func Warning(v ...interface{}) {
	defaultLogger.Log(PriWarning, v...)
}
func Warningf(format string, v ...interface{}) {
	defaultLogger.Log(PriWarning, fmt.Sprintf(format, v...))
}

func Notice(v ...interface{}) {
	defaultLogger.Log(PriNotice, v...)
}
func Noticef(format string, v ...interface{}) {
	defaultLogger.Log(PriNotice, fmt.Sprintf(format, v...))
}

func Info(v ...interface{}) {
	defaultLogger.Log(PriInfo, v...)
}
func Infof(format string, v ...interface{}) {
	defaultLogger.Log(PriInfo, fmt.Sprintf(format, v...))
}

func Debug(v ...interface{}) {
	defaultLogger.Log(PriDebug, v...)
}
func Debugf(format string, v ...interface{}) {
	defaultLogger.Log(PriDebug, fmt.Sprintf(format, v...))
}

// Standard library log functions

func (logger *Logger)Fatalln (v ...interface{}) {
	logger.Log(PriCrit, v...)
	os.Exit(1)
}
func (logger *Logger)Fatalf (format string, v ...interface{}) {
	logger.Logf(PriCrit, format, v...)
	os.Exit(1)
}

func (logger *Logger)Panicln (v ...interface{}) {
	s := fmt.Sprint(v...)
	logger.Log(PriErr, s)
	panic(s)
}
func (logger *Logger)Panicf (format string, v ...interface{}) {
	s := fmt.Sprintf(format, v...)
	logger.Log(PriErr, s)
	panic(s)
}

func (logger *Logger)Println (v ...interface{}) {
	logger.Log(PriInfo, v...)
}
func (logger *Logger)Printf (format string, v ...interface{}) {
	logger.Logf(PriInfo, format, v...)
}


func Fatalln (v ...interface{}) {
	defaultLogger.Log(PriCrit, v...)
	os.Exit(1)
}
func Fatalf (format string, v ...interface{}) {
	defaultLogger.Logf(PriCrit, format, v...)
	os.Exit(1)
}

func Panicln (v ...interface{}) {
	s := fmt.Sprint(v...)
	defaultLogger.Log(PriErr, s)
	panic(s)
}
func Panicf (format string, v ...interface{}) {
	s := fmt.Sprintf(format, v...)
	defaultLogger.Log(PriErr, s)
	panic(s)
}

func Println (v ...interface{}) {
	defaultLogger.Log(PriInfo, v...)
}
func Printf (format string, v ...interface{}) {
	defaultLogger.Logf(PriInfo, format, v...)
}
