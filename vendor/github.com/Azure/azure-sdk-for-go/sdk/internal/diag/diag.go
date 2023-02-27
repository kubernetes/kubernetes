//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package diag

import (
	"fmt"
	"runtime"
	"strings"
)

// Caller returns the file and line number of a frame on the caller's stack.
// If the funtion fails an empty string is returned.
// skipFrames - the number of frames to skip when determining the caller.
//  Passing a value of 0 will return the immediate caller of this function.
func Caller(skipFrames int) string {
	if pc, file, line, ok := runtime.Caller(skipFrames + 1); ok {
		// the skipFrames + 1 is to skip ourselves
		frame := runtime.FuncForPC(pc)
		return fmt.Sprintf("%s()\n\t%s:%d", frame.Name(), file, line)
	}
	return ""
}

// StackTrace returns a formatted stack trace string.
// If the funtion fails an empty string is returned.
// skipFrames - the number of stack frames to skip before composing the trace string.
// totalFrames - the maximum number of stack frames to include in the trace string.
func StackTrace(skipFrames, totalFrames int) string {
	pcCallers := make([]uintptr, totalFrames)
	if frames := runtime.Callers(skipFrames, pcCallers); frames == 0 {
		return ""
	}
	frames := runtime.CallersFrames(pcCallers)
	sb := strings.Builder{}
	for {
		frame, more := frames.Next()
		sb.WriteString(frame.Function)
		sb.WriteString("()\n\t")
		sb.WriteString(frame.File)
		sb.WriteRune(':')
		sb.WriteString(fmt.Sprintf("%d\n", frame.Line))
		if !more {
			break
		}
	}
	return sb.String()
}
