// +build windows

package internal

import "os"

// dupStdout returns the current os.Stdout. On Windows, the output interceptor
// uses osGlobalReassigning which only changes the Go variable, so capturing
// the current os.Stdout before interception starts is sufficient.
func dupStdout() *os.File {
	return os.Stdout
}

func NewOutputInterceptor() OutputInterceptor {
	return NewOSGlobalReassigningOutputInterceptor()
}
