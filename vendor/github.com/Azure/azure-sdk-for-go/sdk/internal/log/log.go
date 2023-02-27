//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package log

import (
	"fmt"
	"os"
	"time"
)

///////////////////////////////////////////////////////////////////////////////////////////////////
// NOTE: The following are exported as public surface area from azcore.  DO NOT MODIFY
///////////////////////////////////////////////////////////////////////////////////////////////////

// Event is used to group entries.  Each group can be toggled on or off.
type Event string

// SetEvents is used to control which events are written to
// the log.  By default all log events are writen.
func SetEvents(cls ...Event) {
	log.cls = cls
}

// SetListener will set the Logger to write to the specified listener.
func SetListener(lst func(Event, string)) {
	log.lst = lst
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// END PUBLIC SURFACE AREA
///////////////////////////////////////////////////////////////////////////////////////////////////

// Should returns true if the specified log event should be written to the log.
// By default all log events will be logged.  Call SetEvents() to limit
// the log events for logging.
// If no listener has been set this will return false.
// Calling this method is useful when the message to log is computationally expensive
// and you want to avoid the overhead if its log event is not enabled.
func Should(cls Event) bool {
	if log.lst == nil {
		return false
	}
	if log.cls == nil || len(log.cls) == 0 {
		return true
	}
	for _, c := range log.cls {
		if c == cls {
			return true
		}
	}
	return false
}

// Write invokes the underlying listener with the specified event and message.
// If the event shouldn't be logged or there is no listener then Write does nothing.
func Write(cls Event, message string) {
	if !Should(cls) {
		return
	}
	log.lst(cls, message)
}

// Writef invokes the underlying listener with the specified event and formatted message.
// If the event shouldn't be logged or there is no listener then Writef does nothing.
func Writef(cls Event, format string, a ...interface{}) {
	if !Should(cls) {
		return
	}
	log.lst(cls, fmt.Sprintf(format, a...))
}

// TestResetEvents is used for TESTING PURPOSES ONLY.
func TestResetEvents() {
	log.cls = nil
}

// logger controls which events to log and writing to the underlying log.
type logger struct {
	cls []Event
	lst func(Event, string)
}

// the process-wide logger
var log logger

func init() {
	initLogging()
}

// split out for testing purposes
func initLogging() {
	if cls := os.Getenv("AZURE_SDK_GO_LOGGING"); cls == "all" {
		// cls could be enhanced to support a comma-delimited list of log events
		log.lst = func(cls Event, msg string) {
			// simple console logger, it writes to stderr in the following format:
			// [time-stamp] Event: message
			fmt.Fprintf(os.Stderr, "[%s] %s: %s\n", time.Now().Format(time.StampMicro), cls, msg)
		}
	}
}
